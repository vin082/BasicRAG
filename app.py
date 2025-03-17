import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler

# Set up OpenAI API key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit app title and description
st.set_page_config(page_title="RAG Chatbot Demo", page_icon="🤖", layout="wide")
st.title("RAG-Based Conversational Chatbot")
st.write("Upload multiple files (PDF, DOCX, or Markdown) and ask questions about their content.")

# Sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload one or more files (PDF, DOCX, or Markdown).
    2. Ask questions about the content of the uploaded files.
    3. The chatbot will respond based on the top 2 retrieved documents and show their similarity scores for relevant queries.
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

# Function to process an uploaded file
def process_file(uploaded_file):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if "." not in file_name:
            st.error(f"Invalid file '{file_name}': No file extension found.")
            return None

        file_type = file_name.split(".")[-1].lower()
        if file_type not in ["pdf", "docx", "md"]:
            st.error(f"Unsupported file type: .{file_type}")
            return None

        file_path = f"./temp_{file_name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_type == "md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            st.error("Unsupported file type!")
            return None

        try:
            documents = loader.load()
            st.success(f"File '{file_name}' uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to load the file '{file_name}': {e}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        if os.path.exists(file_path):
            os.remove(file_path)

        return texts
    return None

# File uploader for multiple files with a dynamic key
uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "docx", "md"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)
st.session_state.uploaded_files = uploaded_files

# Process uploaded files and add them to the knowledge base
if st.session_state.uploaded_files:
    with st.spinner("Processing files..."):
        new_files_processed = False
        progress_bar = st.progress(0)
        #batch_size=5
        for i, uploaded_file in enumerate(st.session_state.uploaded_files):
            file_name = uploaded_file.name
            if file_name not in st.session_state.processed_file_names:
                texts = process_file(uploaded_file)
                if texts:
                    embeddings = OpenAIEmbeddings()
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
                    else:
                        st.session_state.vectorstore.add_documents(texts)
                    st.session_state.processed_file_names.add(file_name)
                    new_files_processed = True
            progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))
        
        if new_files_processed:
            time.sleep(0.5)
            st.success("New files processed successfully!")
        else:
            st.info("All uploaded files were already processed.")

# Clear chat history and knowledge base
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.processed_file_names = set()
    st.session_state.uploaded_files = None
    st.session_state.uploader_key = f"uploader_{int(time.time())}"
    st.success("Chat history and knowledge base cleared!")
    st.rerun()  # Force re-run to ensure UI updates immediately

# Custom callback handler for streaming responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# List of basic greetings to bypass RAG
GREETINGS = {"hi", "hello", "hey", "greetings"}

# Initialize the QA chain and display retrieved documents with scores
if st.session_state.vectorstore:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Top 2 Retrieved Documents"):
                    for i, (doc, score) in enumerate(message["sources"], 1):
                        st.markdown(f"**Document {i}** (from {doc.metadata.get('source', 'unknown')}, Similarity Score: {score:.4f}):")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    if prompt := st.chat_input("Ask a question about the documents:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                streaming=True,
                callbacks=[stream_handler],
            )

            # Check if the prompt is a basic greeting
            prompt_lower = prompt.lower().strip()
            if prompt_lower in GREETINGS:
                response = "Hello! How can I assist you today?"
                stream_handler.on_llm_new_token(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Process with RAG for non-greeting queries
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template="You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:.",
                            input_variables=["context", "question"],
                        )
                    },
                )

                # Run the chain
                result = qa_chain({"query": prompt})
                answer = result["result"]
                source_docs = result["source_documents"]

                # Get similarity scores
                docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(prompt, k=2)
                sources_with_scores = [(doc, score) for doc, score in docs_with_scores]

                # Store the answer and sources with scores
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": stream_handler.text,
                    "sources": sources_with_scores
                })

                # Display retrieved documents with scores
                with st.expander("Top 2 Retrieved Documents"):
                    for i, (doc, score) in enumerate(sources_with_scores, 1):
                        st.markdown(f"**Document {i}** (from {doc.metadata.get('source', 'unknown')}, Similarity Score(smaller is better): {score:.4f}):")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
else:
    st.info("Please upload files to get started.")