"""
app.py
Ultra-Premium Streamlit Web Interface for RLHF Chatbot with Groq API
API key configuration via Streamlit interface
"""

import streamlit as st
import sys
import os
from datetime import datetime
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Flowbotics AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Ultra-Premium CSS Design with Glassmorphism & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    
    /* Clean solid background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Main container */
    .main {
        padding: 0 !important;
        position: relative;
        z-index: 1;
    }
    
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    
    /* Clean header */
    .header-section {
        text-align: center;
        padding: 2.5rem 2rem 2rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.25);
        border-radius: 0 0 20px 20px;
    }
    
    .header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.8rem;
        letter-spacing: -0.5px;
    }
    
    .header-title .emoji {
        font-size: 2.5rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.7rem;
        letter-spacing: 0.5px;
    }
    
    .status-badge {
        margin-top: 1rem;
        display: inline-block;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.85rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
        }
        50% {
            opacity: 0.8;
            box-shadow: 0 0 0 8px rgba(74, 222, 128, 0);
        }
    }
    
    /* Chat wrapper */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        height: 100%;
        max-width: 950px;
        margin: 0 auto;
        padding: 0 1.5rem;
    }
    
    /* Clean empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #475569;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .empty-state h2 {
        color: #1e3a8a;
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .empty-state p {
        font-size: 1rem;
        margin: 0.8rem 0 2rem;
        color: #64748b;
        font-weight: 400;
    }
    
    .empty-state-features {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 2rem;
        text-align: left;
    }
    
    .feature-item {
        padding: 1.2rem;
        background: linear-gradient(135deg, #f0f9ff 0%, #f0f4f8 100%);
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-item:hover {
        transform: translateY(-4px);
        background: linear-gradient(135deg, #e0f2fe 0%, #e8edf4 100%);
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.15);
        border-color: #1e3a8a;
    }
    
    .feature-item span {
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .feature-item strong {
        color: #1e3a8a;
        font-weight: 600;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    .feature-item small {
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Enhanced message styling */
    .message-wrapper {
        margin: 1.5rem 0;
        display: flex;
        animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .user-wrapper {
        justify-content: flex-end;
    }
    
    .assistant-wrapper {
        justify-content: flex-start;
    }
    
    /* User message bubble */
    .user-bubble {
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
        color: #ffffff;
        padding: 1rem 1.3rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25);
        font-size: 0.95rem;
        line-height: 1.6;
        word-wrap: break-word;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .user-bubble:hover {
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.35);
        transform: translateY(-2px);
    }
    
    /* Assistant message bubble */
    .assistant-bubble {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1e293b;
        padding: 1rem 1.3rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        font-size: 0.95rem;
        line-height: 1.6;
        word-wrap: break-word;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        font-weight: 400;
    }
    
    .assistant-bubble:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        border-color: #cbd5e1;
    }
    
    /* Chat input area */
    .chat-input-wrapper {
        margin: 0 auto 2.5rem;
        width: 100%;
        max-width: 950px;
        padding: 0 1.5rem;
        position: relative;
    }
    
    /* Streamlit chat input with glassmorphism */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 25px !important;
        border: 2px solid #cbd5e1 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        padding: 0.9rem 1.8rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #1e3a8a !important;
        background: #ffffff !important;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.2) !important;
        transform: translateY(-2px);
    }
    
    [data-testid="stChatInput"] input {
        color: #1e293b !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stChatInput"] input::placeholder {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }
    
    /* Premium send button */
    .stChatInput button {
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 15px !important;
        width: 45px !important;
        height: 45px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3) !important;
        font-size: 1.2rem !important;
    }
    
    .stChatInput button:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.5) !important;
    }
    
    .stChatInput button:active {
        transform: scale(0.95);
    }
    
    /* Enhanced typing indicator */
    .typing-dots {
        display: flex;
        gap: 6px;
        padding: 1.2rem 1.5rem;
    }
    
    .typing-dots span {
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3);
    }
    
    .typing-dots span:nth-child(1) {
        animation-delay: 0s;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typingBounce {
        0%, 60%, 100% {
            opacity: 0.3;
            transform: translateY(0) scale(1);
        }
        30% {
            opacity: 1;
            transform: translateY(-12px) scale(1.2);
        }
    }
    
    /* Premium footer */
    .footer-section {
        text-align: center;
        padding: 2rem 1.5rem;
        color: #64748b;
        font-size: 0.9rem;
        background: rgba(30, 58, 138, 0.05);
        border-top: 1px solid rgba(30, 58, 138, 0.1);
        margin: 0 1.5rem;
        border-radius: 12px;
    }
    
    .footer-section a {
        color: #1e3a8a;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .footer-section a:hover {
        color: #2c5282;
        text-decoration: underline;
    }
    
    /* API Key Setup Box */
    .api-key-box {
        max-width: 600px;
        margin: 3rem auto;
        padding: 2.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.15);
        border: 2px solid #cbd5e1;
    }
    
    .api-key-box h3 {
        color: #1e3a8a;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .api-key-box p {
        color: #64748b;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }
        
        .header-title .emoji {
            font-size: 2rem;
        }
        
        .empty-state-features {
            grid-template-columns: 1fr;
        }
        
        .user-bubble, .assistant-bubble {
            max-width: 85%;
            padding: 0.9rem 1.1rem;
            font-size: 0.9rem;
        }
        
        .api-key-box {
            margin: 2rem 1rem;
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'total_interactions' not in st.session_state:
    st.session_state.total_interactions = 0

if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""

if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False

# Premium Header with Animation
st.markdown("""
<div class="header-section">
    <h1 class="header-title">
        <span class="emoji">🤖</span>
        Flowbotics AI
    </h1>
    <p class="header-subtitle">Next-Gen Intelligence • Powered by Groq & Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

# Safe initialization function
@st.cache_resource
def initialize_chatbot(api_key):
    """Initialize chatbot with Groq API and error handling"""
    try:
        # Import the Groq-based chatbot
        from chatbot_llm import FlowboticsChatbotOptimized
        
        # Initialize chatbot with Groq - pass API key directly
        chatbot = FlowboticsChatbotOptimized(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            enable_rlhf=True
        )
        return chatbot, True, None
    except Exception as e:
        return None, False, str(e)

# Check if API key is set
if not st.session_state.groq_api_key or not st.session_state.chatbot_initialized:
    st.markdown("""
    <div class="api-key-box">
        <h3>🔑 API Key Required</h3>
        <p>Please enter your Groq API key to start using the chatbot. You can get your API key from <a href="https://console.groq.com/keys" target="_blank">Groq Console</a>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input
    api_key_input = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        placeholder="gsk_...",
        help="Get your API key from https://console.groq.com/keys"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Initialize Chatbot", type="primary", use_container_width=True):
            if api_key_input.strip():
                with st.spinner("Initializing chatbot..."):
                    chatbot, initialized, error = initialize_chatbot(api_key_input.strip())
                    
                    if initialized:
                        st.session_state.groq_api_key = api_key_input.strip()
                        st.session_state.chatbot_initialized = True
                        st.success("✅ Chatbot initialized successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Initialization failed: {error}")
            else:
                st.warning("⚠️ Please enter a valid API key")
    
    st.stop()

# Initialize chatbot if key is available
if st.session_state.chatbot_initialized:
    chatbot, initialized, error = initialize_chatbot(st.session_state.groq_api_key)
    
    if not initialized:
        st.error("❌ Chatbot initialization failed. Please check your API key.")
        if error:
            with st.expander("⚙️ Error Details"):
                st.code(error)
        
        if st.button("🔄 Reset API Key"):
            st.session_state.groq_api_key = ""
            st.session_state.chatbot_initialized = False
            st.rerun()
        
        st.stop()

# Enhanced empty state
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="empty-state">
        <h2>Welcome to the Future of AI</h2>
        <p>Your intelligent assistant is ready to revolutionize your workflow</p>
        <div class="empty-state-features">
            <div class="feature-item">
                <span>✨</span>
                <strong>Business Automation</strong>
                <small>Streamline operations with AI</small>
            </div>
            <div class="feature-item">
                <span>🚀</span>
                <strong>AI Solutions</strong>
                <small>Custom intelligent systems</small>
            </div>
            <div class="feature-item">
                <span>💼</span>
                <strong>Lead Generation</strong>
                <small>Smart customer acquisition</small>
            </div>
            <div class="feature-item">
                <span>📊</span>
                <strong>Data Analytics</strong>
                <small>Insights that drive growth</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display messages with animations
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="message-wrapper user-wrapper">
                <div class="user-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-wrapper assistant-wrapper">
                <div class="assistant-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input with glassmorphism
st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
user_input = st.chat_input("Ask me anything...", key="user_input")
st.markdown('</div>', unsafe_allow_html=True)

# Process input
if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    st.rerun()

# Generate response
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    # Show enhanced typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="message-wrapper assistant-wrapper">
        <div class="assistant-bubble">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        user_message = st.session_state.messages[-1]["content"]
        
        # Collect response
        full_response = ""
        for token in chatbot.stream_chat(user_message, use_rag=True):
            full_response += token
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        st.session_state.total_interactions += 1
        st.rerun()
        
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"Error: {str(e)}")
        st.info("💡 Tip: Make sure your Groq API key is valid and has available credits.")

# Premium Footer
st.markdown("""
<div class="footer-section">
    <p>⚡ Powered by <strong>Groq API</strong> with Llama 3.3 & RLHF Training | © 2025 <a href="#">Flowbotics</a></p>
</div>
""", unsafe_allow_html=True)