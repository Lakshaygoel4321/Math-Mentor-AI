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


# Page config
st.set_page_config(
    page_title="Math Mentor - AI Math Solver",
    page_icon="🧮",
    layout="wide"
)


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
st.title("🧮 Math Mentor - AI-Powered Math Solver")
st.markdown("Upload an image, record audio, or type your JEE-style math problem")


# Input mode selector
input_mode = st.radio(
    "Choose Input Mode:",
    ["📝 Text", "📷 Image", "🎤 Audio"],
    horizontal=True
)


raw_input = None
input_type = None
confidence = 1.0


# Input handling
if input_mode == "📝 Text":
    raw_input = st.text_area(
        "Enter your math problem:",
        placeholder="Example: Solve x² - 4x + 4 = 0",
        height=100
    )
    input_type = "text"


elif input_mode == "📷 Image":
    uploaded_file = st.file_uploader(
        "Upload image of math problem",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("Extracting text from image..."):
                raw_input, confidence = components["ocr"].process_image(image)
                
            st.metric("OCR Confidence", f"{confidence*100:.1f}%")
            
            if confidence < 0.7:
                st.warning("⚠️ Low confidence! Please verify extracted text.")
            
            raw_input = st.text_area(
                "Extracted Text (editable):",
                value=raw_input,
                height=100
            )
        
        input_type = "image"


elif input_mode == "🎤 Audio":
    # Reset audio state when switching modes
    if 'last_input_mode' not in st.session_state or st.session_state.last_input_mode != "🎤 Audio":
        st.session_state.transcribed_text = None
        st.session_state.audio_processed = False
        st.session_state.current_audio_file = None
    st.session_state.last_input_mode = "🎤 Audio"
    
    # Professional upload interface
    st.info("🎤 **Record audio on your device, then upload here for transcription**")
    
    st.markdown("""
    **How to use:**
    1. 📱 Use your phone/computer's voice recorder
    2. 📤 Upload the audio file below
    3. ✨ We'll transcribe it automatically!
    """)
    
    audio_data = None
    audio_file_name = None
    
    audio_file = st.file_uploader(
        "Upload your audio recording",
        type=["mp3", "wav", "m4a", "webm", "ogg"],
        key="audio_uploader",
        help="Supported formats: MP3, WAV, M4A, WEBM, OGG"
    )
    
    if audio_file:
        audio_data = audio_file
        audio_file_name = audio_file.name
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.audio(audio_file)
        with col2:
            st.success("✅ File uploaded!")
            st.caption(f"📁 {audio_file_name}")
    
    # Process audio if it's new
    if audio_data and audio_file_name:
        # Check if this is a new file
        if st.session_state.current_audio_file != audio_file_name or not st.session_state.audio_processed:
            st.session_state.current_audio_file = audio_file_name
            
            with st.spinner("🔄 Transcribing audio... This may take a moment"):
                try:
                    transcribed = components["audio"].process_audio(audio_data)
                    st.session_state.transcribed_text = transcribed
                    st.session_state.audio_processed = True
                    st.success("✅ Transcription complete!")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}")
                    st.exception(e)
                    st.session_state.transcribed_text = None
    
    # Show editable text area if transcription exists
    if st.session_state.transcribed_text:
        st.markdown("---")
        raw_input = st.text_area(
            "📝 Transcribed Text (editable):",
            value=st.session_state.transcribed_text,
            height=120,
            key="audio_text_area",
            help="✏️ Edit the text if transcription is incorrect"
        )
        
        # Update session state with edited text
        st.session_state.transcribed_text = raw_input
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Upload New Audio", use_container_width=True):
                st.session_state.transcribed_text = None
                st.session_state.audio_processed = False
                st.session_state.current_audio_file = None
                st.rerun()
        
        input_type = "audio"


# Solve button - show if there's input
if raw_input:
    st.divider()
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        with st.spinner("Processing your problem..."):
            
            try:
                # Step 1: Parse
                st.write("### 🔍 Step 1: Parsing Problem")
                parsed = components["parser"].parse(raw_input)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.json(parsed, expanded=False)
                with col2:
                    topic = parsed.get('topic', 'algebra')
                    st.info(f"**Topic**: {topic.title() if topic else 'Unknown'}")
                    if parsed.get('needs_clarification'):
                        st.warning(f"⚠️ {parsed.get('clarification_reason', 'Clarification needed')}")
                
                # Check for similar problems in memory
                similar_problems = components["memory"].get_similar_problems(
                    parsed.get("problem_text", raw_input)
                )
                
                if similar_problems:
                    with st.expander("💡 Similar Problems Found in Memory"):
                        for sp in similar_problems:
                            problem_text = sp.get('parsed_problem', {}).get('problem_text', 'Unknown problem')
                            st.write(f"- {problem_text}")
                
                # Step 2: Solve
                st.write("### 🧮 Step 2: Solving")
                solution = components["solver"].solve(parsed)
                
                # Show retrieved context
                with st.expander("📚 Retrieved Knowledge"):
                    if solution.get("retrieved_context"):
                        for i, ctx in enumerate(solution["retrieved_context"]):
                            st.markdown(f"**Source {i+1}** (Score: {ctx.get('score', 0):.3f})")
                            st.text(ctx.get("content", "No content"))
                            st.divider()
                    else:
                        st.info("No relevant context retrieved")
                
                # Step 3: Verify
                st.write("### ✅ Step 3: Verification")
                verification = components["verifier"].verify(
                    parsed.get("problem_text", raw_input),
                    solution.get("llm_solution", "No solution generated")
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if verification.get("is_correct"):
                        st.success("✅ Solution Verified")
                    else:
                        st.error("❌ Issues Found")
                    st.metric("Confidence", f"{verification.get('confidence', 0)*100:.0f}%")
                
                with col2:
                    if verification.get("issues"):
                        for issue in verification["issues"]:
                            st.warning(f"⚠️ {issue}")
                
                # Step 4: Explain
                st.write("### 📖 Step 4: Explanation")
                explanation = components["explainer"].explain(
                    parsed.get("problem_text", raw_input),
                    solution.get("llm_solution", "No solution available")
                )
                
                st.markdown(explanation)
                
                # SymPy result if available
                if solution.get("sympy_result", {}).get("success"):
                    st.code(f"SymPy Solution: {solution['sympy_result']['solution']}", language="python")
                
                # Feedback section
                st.write("### 💬 Feedback")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Correct", use_container_width=True, key="correct_btn"):
                        memory_id = components["memory"].store_interaction({
                            "original_input": raw_input,
                            "input_type": input_type,
                            "parsed_problem": parsed,
                            "solution": solution.get("llm_solution"),
                            "verification": verification,
                            "feedback": "correct"
                        })
                        st.success(f"✅ Feedback saved! (ID: {memory_id[:8]})")
                
                with col2:
                    if st.button("❌ Incorrect", use_container_width=True, key="incorrect_btn"):
                        st.session_state["show_feedback_form"] = True
                
                if st.session_state.get("show_feedback_form"):
                    user_comment = st.text_input("What's wrong? (optional)")
                    if st.button("Submit Feedback", key="submit_feedback_btn"):
                        memory_id = components["memory"].store_interaction({
                            "original_input": raw_input,
                            "input_type": input_type,
                            "parsed_problem": parsed,
                            "solution": solution.get("llm_solution"),
                            "verification": verification,
                            "feedback": "incorrect",
                            "user_comment": user_comment
                        })
                        st.success(f"✅ Feedback saved! (ID: {memory_id[:8]})")
                        st.session_state["show_feedback_form"] = False
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)


# Sidebar
with st.sidebar:
    st.header("📊 System Stats")
    
    # Reload memories to get current count
    try:
        current_count = len(components["memory"].load_memories())
        st.metric("Total Problems Solved", current_count)
    except Exception as e:
        st.metric("Total Problems Solved", 0)
        st.caption(f"⚠️ Error loading memory: {str(e)}")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    **Math Mentor** uses:
    - 🤖 Multi-agent AI system
    - 📚 RAG for knowledge retrieval
    - 🧮 SymPy for symbolic math
    - 🎯 Human-in-the-loop verification
    - 🧠 Memory-based learning
    """)
    
    st.divider()
    
    if st.button("🔄 Reset Memory", use_container_width=True, key="reset_memory_btn"):
        try:
            components["memory"].clear_memories()
            st.success("✅ Memory cleared successfully!")
            # Force reload to update count
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing memory: {str(e)}")
    
    st.divider()
    
    # Show storage location
    with st.expander("🗂️ Storage Info"):
        st.caption(f"**Memory File:** `{components['memory'].storage_path}`")
        if os.path.exists(components['memory'].storage_path):
            file_size = os.path.getsize(components['memory'].storage_path)
            st.caption(f"**File Size:** {file_size} bytes")
        else:
            st.caption("**Status:** Not created yet")
