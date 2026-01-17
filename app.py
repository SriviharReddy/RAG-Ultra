"""
RAG-Ultra: Multimodal RAG System
Streamlit Frontend with Modern UI
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any
import uuid

from src.graph import rag_workflow, document_manager
from src.ingestion import DocumentIngester
from src.config import config


# Page Configuration
st.set_page_config(
    page_title="RAG-Ultra | Multimodal RAG System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #00d9ff, #00ff88, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d2d44 0%, #1a1a2e 100%);
        color: #e0e0e0;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        background: rgba(0, 217, 255, 0.2);
        color: #00d9ff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 0.25rem;
        border: 1px solid rgba(0, 217, 255, 0.3);
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Upload area */
    .upload-area {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: white;
        padding: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(26, 26, 46, 0.6);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 12px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 10px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = 0
    
    if "ingester" not in st.session_state:
        st.session_state.ingester = DocumentIngester()


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>🔮 RAG-Ultra</h1>
        <p>Multimodal RAG System with OCR, Vision-Language Models & Structured Retrieval</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with document upload and stats"""
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload scanned PDFs, documents with tables, formulas, or images"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                process_documents(uploaded_files)
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 System Stats")
        
        try:
            stats = document_manager.get_stats()
            doc_count = stats.get("count", 0)
        except Exception:
            doc_count = st.session_state.documents_loaded
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{doc_count}</div>
                <div class="stats-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.messages)}</div>
                <div class="stats-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        with st.expander("Model Settings"):
            st.text_input(
                "Model ID",
                value=config.aws.bedrock_model_id,
                disabled=True,
                help="Amazon Bedrock model for generation"
            )
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.model.temperature,
                step=0.1
            )
            st.number_input(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=config.retrieval.top_k_final
            )
        
        with st.expander("Retrieval Settings"):
            st.number_input(
                "Chunk Size",
                min_value=200,
                max_value=2000,
                value=config.retrieval.chunk_size
            )
            st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=config.retrieval.chunk_overlap
            )
        
        st.markdown("---")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                document_manager.clear_all()
                st.session_state.messages = []
                st.session_state.documents_loaded = 0
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()


def process_documents(uploaded_files):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_docs = []
    
    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Processing: {file.name}")
        
        try:
            # Read file bytes
            file_bytes = file.read()
            
            # Ingest document
            docs = st.session_state.ingester.ingest_pdf_bytes(
                file_bytes,
                file.name
            )
            total_docs.extend(docs)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    if total_docs:
        status_text.text("Adding to vector store...")
        result = document_manager.add_documents(total_docs)
        
        st.session_state.documents_loaded = result.get("total_documents", 0)
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        st.success(f"✅ Processed {len(uploaded_files)} file(s), added {result.get('added', 0)} document chunks")


def render_chat_message(role: str, content: str, sources: List[str] = None):
    """Render a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message animate-fade-in">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message animate-fade-in">
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            source_html = "".join([
                f'<span class="source-badge">📄 {src}</span>'
                for src in sources[:5]
            ])
            st.markdown(f"""
            <div style="margin-top: 0.5rem;">
                {source_html}
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface"""
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
            <div class="chat-container animate-fade-in" style="text-align: center; padding: 3rem;">
                <h2 style="color: #ffffff; margin-bottom: 1rem;">👋 Welcome to RAG-Ultra</h2>
                <p style="color: rgba(255,255,255,0.7); max-width: 600px; margin: 0 auto;">
                    Upload your PDF documents using the sidebar, then ask questions about their content.
                    I can understand scanned documents, tables, formulas, and visual content.
                </p>
                <div style="margin-top: 2rem;">
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">
                        💡 Try asking about specific data in tables, explaining diagrams, or extracting information from scanned pages.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat history
            for message in st.session_state.messages:
                render_chat_message(
                    message["role"],
                    message["content"],
                    message.get("sources", [])
                )
    
    # Input area
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input(
        "Ask a question about your documents...",
        key="chat_input"
    )
    
    if user_input:
        handle_user_input(user_input)


def handle_user_input(user_input: str):
    """Handle user input and generate response"""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    render_chat_message("user", user_input)
    
    # Generate response
    with st.spinner("🔍 Searching documents and generating response..."):
        try:
            # Build chat history for context
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]
            
            # Invoke RAG workflow
            result = rag_workflow.invoke(
                query=user_input,
                chat_history=chat_history,
                session_id=st.session_state.session_id
            )
            
            response = result.get("response", "I couldn't generate a response.")
            sources = result.get("sources", [])
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })
    
    st.rerun()


def render_features_section():
    """Render the features section"""
    with st.expander("🌟 Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📝 OCR Processing**
            - Amazon Textract integration
            - Scanned PDF support
            - Table extraction
            - Form field detection
            """)
        
        with col2:
            st.markdown("""
            **🎯 Smart Retrieval**
            - Parent document retrieval
            - Semantic search
            - AWS Cohere reranking
            - Query expansion
            """)
        
        with col3:
            st.markdown("""
            **🔮 Vision-Language**
            - Image understanding
            - Diagram analysis
            - Formula extraction
            - Chart interpretation
            """)


def main():
    """Main application entry point"""
    initialize_session_state()
    render_header()
    render_sidebar()
    render_features_section()
    render_chat_interface()


if __name__ == "__main__":
    main()
