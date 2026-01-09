import streamlit as st
from multimodal.ocr_processor import OCRProcessor
from multimodal.audio_processor import AudioProcessor
from rag.vectorstore.vectorstore import RAGPipeline
from agents.parser_agent import ParserAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent
from memory.store import MemoryStore
from PIL import Image
import os
from io import BytesIO


# Try to import audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="Math Mentor - AI Math Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .mode-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Auto-setup on first run
def auto_setup():
    """Auto setup for Streamlit Cloud or first-time local run"""
    
    # Create necessary directories
    os.makedirs("memory", exist_ok=True)
    os.makedirs("rag/knowledge_base", exist_ok=True)
    os.makedirs("rag/vectorstore", exist_ok=True)
    
    # Check if vectorstore exists
    if not os.path.exists("rag/vectorstore/index.faiss"):
        st.info("🔄 First time setup... Building knowledge base and vector store...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Create knowledge base
            status_text.text("📚 Creating knowledge base...")
            progress_bar.progress(25)
            
            if os.path.exists("create_knowledge_base.py"):
                import subprocess
                result = subprocess.run(
                    ["python", "create_knowledge_base.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✅ Knowledge base created!")
                else:
                    st.warning("⚠️ Knowledge base creation skipped")
            
            progress_bar.progress(50)
            
            # Step 2: Build vectorstore
            status_text.text("🔨 Building vector store...")
            progress_bar.progress(75)
            
            rag = RAGPipeline()
            rag.build_vectorstore()
            
            progress_bar.progress(100)
            status_text.text("✅ Setup complete!")
            
            st.success("✅ Vector store built successfully!")
            st.info("🔄 Reloading application...")
            
            # Clear cache and rerun
            st.cache_resource.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Setup failed: {e}")
            st.exception(e)
            st.warning("Please check your configuration and try again.")
            st.stop()


# Run auto setup
auto_setup()


# Initialize components
@st.cache_resource
def init_components():
    rag = RAGPipeline()
    rag.load_vectorstore()
    
    return {
        "ocr": OCRProcessor(),
        "audio": AudioProcessor(),
        "rag": rag,
        "parser": ParserAgent(),
        "solver": SolverAgent(rag),
        "verifier": VerifierAgent(),
        "explainer": ExplainerAgent(),
        "memory": MemoryStore()
    }


components = init_components()


# Initialize session state
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None
if 'audio_processed' not in st.session_state:
    st.session_state.audio_processed = False
if 'current_audio_file' not in st.session_state:
    st.session_state.current_audio_file = None


# Header
st.markdown("<h1 class='main-header'>🧮 Math Mentor - AI Math Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Upload an image, record audio, or type your JEE-style math problem</p>", unsafe_allow_html=True)

st.divider()

# Input mode selector with better styling
col1, col2, col3 = st.columns(3)

with col1:
    text_selected = st.button("📝 Text Input", use_container_width=True, type="primary" if st.session_state.get("input_mode", "text") == "text" else "secondary")
    if text_selected:
        st.session_state.input_mode = "text"

with col2:
    image_selected = st.button("📷 Image Upload", use_container_width=True, type="primary" if st.session_state.get("input_mode", "text") == "image" else "secondary")
    if image_selected:
        st.session_state.input_mode = "image"

with col3:
    audio_selected = st.button("🎤 Audio Recording", use_container_width=True, type="primary" if st.session_state.get("input_mode", "text") == "audio" else "secondary")
    if audio_selected:
        st.session_state.input_mode = "audio"

# Default to text if not set
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"

input_mode = st.session_state.input_mode

st.divider()

raw_input = None
input_type = None
confidence = 1.0


# Input handling with beautiful cards
if input_mode == "text":
    st.markdown("### 📝 Enter Your Math Problem")
    raw_input = st.text_area(
        "Type your problem here:",
        placeholder="Example: Solve the equation x² - 4x + 4 = 0\nOr: Find the derivative of sin(x²)",
        height=150,
        help="Enter any JEE-level math problem"
    )
    input_type = "text"
    
    if raw_input:
        st.info(f"📊 Characters: {len(raw_input)} | Words: {len(raw_input.split())}")


elif input_mode == "image":
    st.markdown("### 📷 Upload Math Problem Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your math problem"
        )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Extracting text from image..."):
                raw_input, confidence = components["ocr"].process_image(image)
            
            confidence_color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
            st.metric("OCR Confidence", f"{confidence_color} {confidence*100:.1f}%")
            
            if confidence < 0.7:
                st.warning("⚠️ Low confidence detected! Please verify the extracted text below.")
            else:
                st.success("✅ Text extracted successfully!")
        
        st.markdown("---")
        raw_input = st.text_area(
            "📝 Extracted Text (You can edit if needed):",
            value=raw_input,
            height=120,
            help="Edit the text if OCR made any mistakes"
        )
        
        input_type = "image"


elif input_mode == "audio":
    st.markdown("### 🎤 Record or Upload Audio")
    
    # Reset audio state when switching modes
    if 'last_input_mode' not in st.session_state or st.session_state.last_input_mode != "audio":
        st.session_state.transcribed_text = None
        st.session_state.audio_processed = False
        st.session_state.current_audio_file = None
    st.session_state.last_input_mode = "audio"
    
    # Audio options tabs
    audio_tab1, audio_tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio File"])
    
    audio_data = None
    audio_file_name = None
    
    with audio_tab1:
        if AUDIO_RECORDER_AVAILABLE:
            st.info("🎤 Click the button below to start/stop recording")
            
            # Audio recorder component
            audio_bytes = audio_recorder(
                text="Click to Record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="3x",
            )
            
            if audio_bytes:
                st.success("✅ Audio recorded successfully!")
                st.audio(audio_bytes, format="audio/wav")
                audio_data = BytesIO(audio_bytes)
                audio_file_name = "recorded_audio.wav"
        else:
            st.warning("🎙️ Audio recording is not available in this deployment.")
            st.info("👉 Please use the **Upload Audio File** tab instead.")
    
    with audio_tab2:
        st.markdown("""
        **📋 Instructions:**
        1. Record audio on your device's voice recorder
        2. Upload the file below (MP3, WAV, M4A, WEBM, OGG)
        3. We'll transcribe it automatically!
        """)
        
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "webm", "ogg"],
            key="audio_uploader",
            help="Upload your recorded audio file"
        )
        
        if audio_file:
            audio_data = audio_file
            audio_file_name = audio_file.name
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.audio(audio_file)
            with col2:
                st.success("✅ Uploaded!")
                st.caption(f"📁 {audio_file_name}")
    
    # Process audio if available
    if audio_data and audio_file_name:
        # Check if this is a new file
        if st.session_state.current_audio_file != audio_file_name or not st.session_state.audio_processed:
            st.session_state.current_audio_file = audio_file_name
            
            with st.spinner("🔄 Transcribing audio... Please wait"):
                try:
                    transcribed = components["audio"].process_audio(audio_data)
                    st.session_state.transcribed_text = transcribed
                    st.session_state.audio_processed = True
                    st.success("✅ Transcription complete!")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}")
                    st.exception(e)
                    st.session_state.transcribed_text = None
    
    # Show editable transcribed text
    if st.session_state.transcribed_text:
        st.markdown("---")
        st.markdown("### 📝 Transcribed Text")
        raw_input = st.text_area(
            "Edit if needed:",
            value=st.session_state.transcribed_text,
            height=120,
            key="audio_text_area",
            help="✏️ You can edit the transcribed text if there are any errors"
        )
        
        st.session_state.transcribed_text = raw_input
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Record/Upload New", use_container_width=True):
                st.session_state.transcribed_text = None
                st.session_state.audio_processed = False
                st.session_state.current_audio_file = None
                st.rerun()
        
        input_type = "audio"


# Solve button
if raw_input:
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solve_button = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
    
    if solve_button:
        with st.spinner("🤖 AI is processing your problem..."):
            
            try:
                # Step 1: Parse
                st.markdown("### 🔍 Step 1: Parsing Problem")
                parsed = components["parser"].parse(raw_input)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    with st.expander("📋 Parsed Data", expanded=False):
                        st.json(parsed)
                with col2:
                    topic = parsed.get('topic', 'algebra')
                    difficulty = parsed.get('difficulty', 'medium')
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("📚 Topic", topic.title() if topic else 'Unknown')
                    with metric_col2:
                        st.metric("⭐ Difficulty", difficulty.title() if difficulty else 'Medium')
                    
                    if parsed.get('needs_clarification'):
                        st.warning(f"⚠️ {parsed.get('clarification_reason', 'Clarification needed')}")
                
                # Check for similar problems
                similar_problems = components["memory"].get_similar_problems(
                    parsed.get("problem_text", raw_input)
                )
                
                if similar_problems:
                    with st.expander("💡 Similar Problems from Memory"):
                        for sp in similar_problems:
                            problem_text = sp.get('parsed_problem', {}).get('problem_text', 'Unknown')
                            st.markdown(f"- {problem_text}")
                
                st.divider()
                
                # Step 2: Solve
                st.markdown("### 🧮 Step 2: Solving")
                solution = components["solver"].solve(parsed)
                
                # Show retrieved context
                with st.expander("📚 Retrieved Knowledge Base Context"):
                    if solution.get("retrieved_context"):
                        for i, ctx in enumerate(solution["retrieved_context"], 1):
                            st.markdown(f"**📖 Source {i}** (Relevance: {ctx.get('score', 0):.3f})")
                            st.text(ctx.get("content", "No content"))
                            if i < len(solution["retrieved_context"]):
                                st.divider()
                    else:
                        st.info("No relevant context found in knowledge base")
                
                st.divider()
                
                # Step 3: Verify
                st.markdown("### ✅ Step 3: Verification")
                verification = components["verifier"].verify(
                    parsed.get("problem_text", raw_input),
                    solution.get("llm_solution", "No solution generated")
                )
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if verification.get("is_correct"):
                        st.success("✅ Solution Verified")
                    else:
                        st.error("❌ Issues Detected")
                    
                    confidence_val = verification.get('confidence', 0)
                    st.metric("🎯 Confidence", f"{confidence_val*100:.0f}%")
                
                with col2:
                    if verification.get("issues"):
                        st.markdown("**⚠️ Verification Issues:**")
                        for issue in verification["issues"]:
                            st.warning(f"• {issue}")
                    else:
                        st.success("✅ No issues found!")
                
                st.divider()
                
                # Step 4: Explain
                st.markdown("### 📖 Step 4: Detailed Explanation")
                explanation = components["explainer"].explain(
                    parsed.get("problem_text", raw_input),
                    solution.get("llm_solution", "No solution available")
                )
                
                st.markdown(explanation)
                
                # SymPy result if available
                if solution.get("sympy_result", {}).get("success"):
                    with st.expander("🔬 SymPy Symbolic Solution"):
                        st.code(f"Solution: {solution['sympy_result']['solution']}", language="python")
                
                st.divider()
                
                # Feedback section
                st.markdown("### 💬 Rate This Solution")
                st.markdown("*Help us improve by providing feedback*")
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("✅ Correct", use_container_width=True, type="primary", key="correct_btn"):
                        memory_id = components["memory"].store_interaction({
                            "original_input": raw_input,
                            "input_type": input_type,
                            "parsed_problem": parsed,
                            "solution": solution.get("llm_solution"),
                            "verification": verification,
                            "feedback": "correct"
                        })
                        st.success(f"✅ Thanks! Feedback saved (ID: {memory_id[:8]})")
                
                with col2:
                    if st.button("❌ Incorrect", use_container_width=True, key="incorrect_btn"):
                        st.session_state["show_feedback_form"] = True
                
                if st.session_state.get("show_feedback_form"):
                    st.markdown("---")
                    user_comment = st.text_input("💭 What's wrong? (optional)", placeholder="Describe the issue...")
                    if st.button("📤 Submit Feedback", type="primary", key="submit_feedback_btn"):
                        memory_id = components["memory"].store_interaction({
                            "original_input": raw_input,
                            "input_type": input_type,
                            "parsed_problem": parsed,
                            "solution": solution.get("llm_solution"),
                            "verification": verification,
                            "feedback": "incorrect",
                            "user_comment": user_comment
                        })
                        st.success(f"✅ Thanks for your feedback! (ID: {memory_id[:8]})")
                        st.session_state["show_feedback_form"] = False
            
            except Exception as e:
                st.error(f"❌ An error occurred while processing: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)


# Sidebar
with st.sidebar:
    st.markdown("## 📊 System Statistics")
    
    # Reload memories to get current count
    try:
        current_count = len(components["memory"].load_memories())
        st.metric("Problems Solved", current_count, delta="+1" if current_count > 0 else None)
    except Exception as e:
        st.metric("Problems Solved", 0)
        st.caption(f"⚠️ {str(e)}")
    
    st.divider()
    
    st.markdown("## ℹ️ About Math Mentor")
    st.markdown("""
    **Powered by:**
    - 🤖 Multi-agent AI System
    - 📚 RAG Knowledge Retrieval
    - 🧮 SymPy Symbolic Math
    - ✅ Human-in-the-loop Verification
    - 🧠 Memory-based Learning
    
    **Version:** 1.0.0
    """)
    
    st.divider()
    
    st.markdown("## 🛠️ Actions")
    
    if st.button("🔄 Reset Memory", use_container_width=True, type="secondary"):
        if st.session_state.get("confirm_reset"):
            try:
                components["memory"].clear_memories()
                st.success("✅ Memory cleared!")
                st.session_state["confirm_reset"] = False
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.session_state["confirm_reset"] = True
            st.warning("⚠️ Click again to confirm reset")
    
    st.divider()
    
    # Storage info
    with st.expander("🗂️ Storage Information"):
        st.caption(f"**Path:** `{components['memory'].storage_path}`")
        if os.path.exists(components['memory'].storage_path):
            file_size = os.path.getsize(components['memory'].storage_path)
            st.caption(f"**Size:** {file_size:,} bytes")
        else:
            st.caption("**Status:** Not created yet")
    
    st.divider()
    st.caption("Made with ❤️ using Streamlit")
