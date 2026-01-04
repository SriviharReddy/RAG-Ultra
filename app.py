import streamlit as st
import os
import tempfile
from src.graph import RAGGraph

st.set_page_config(page_title="RAG-Ultra", layout="wide")

st.title("RAG-Ultra: Multimodal Intelligent Chatbot")
st.markdown("""
_High-accuracy RAG optimized for scanned PDFs, tables, and visual content._
""")

# Initialize Session State
if "graph" not in st.session_state:
    st.session_state.graph = RAGGraph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Configuration & Ingestion
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Processing with Amazon Textract..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    status = st.session_state.graph.ingest_document(tmp_path)
                    st.success(status)
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    os.remove(tmp_path)

    st.markdown("---")
    st.subheader("System Status")
    st.caption("✅ Textract Pipeline: Ready")
    st.caption("✅ Parent Retriever: Active")
    st.caption("✅ LangSmith Tracing: Enabled")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Retrieving & Reasoning..."):
            try:
                # Run LangGraph Agent
                result = st.session_state.graph.run(prompt)
                full_response = result["answer"]
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"I encountered an error: {e}"
                message_placeholder.error(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
