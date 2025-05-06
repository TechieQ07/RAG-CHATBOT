import os
import tempfile
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Hello! Ask me anything about your documents."}
        ]
    if 'hf_api_key' not in st.session_state:
        st.session_state['hf_api_key'] = None

# Handle conversation
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Chat display using Streamlit native chat UI
def display_chat_history(chain):
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your documents...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            response = conversation_chat(user_input, chain, st.session_state["history"])
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Create conversational retrieval chain
def create_conversational_chain(vector_store, hf_api_key):
    generator_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.01, "max_length": 500},
        huggingfacehub_api_token=hf_api_key,
        task="text2text-generation"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=generator_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

# Main app function
def main():
    initialize_session_state()
    st.title("📚 Multi-Document RAG ChatBot")

    # Hugging Face API Key
    if not st.session_state['hf_api_key']:
        st.session_state['hf_api_key'] = st.text_input("Enter your Hugging Face API token:", type="password")
        if not st.session_state['hf_api_key']:
            st.warning("Please enter your Hugging Face API token to continue.")
            return

    st.sidebar.title("📂 Document Upload")
    uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = None
            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_path)
            elif file_ext == ".txt":
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            try:
                if loader:
                    text.extend(loader.load())
            finally:
                os.remove(tmp_path)

        if not text:
            st.error("No valid text extracted.")
            return

        # Split and embed documents
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(text)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)

        # Create QA chain and start chat
        chain = create_conversational_chain(vector_store, st.session_state['hf_api_key'])
        display_chat_history(chain)

if __name__ == "__main__":
    main()
