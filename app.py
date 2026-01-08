import streamlit as st
from multimodal.ocr_processor import OCRProcessor
from multimodal.audio_processor import AudioProcessor
from rag.vectorstore import RAGPipeline
from agents.parser_agent import ParserAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent
from memory.store import MemoryStore
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Math Mentor - AI Math Solver",
    page_icon="🧮",
    layout="wide"
)

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
        col1, col2 = st.columns()[1]
        
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
    audio_file = st.file_uploader(
        "Upload audio recording",
        type=["mp3", "wav", "m4a"]
    )
    
    if audio_file:
        st.audio(audio_file)
        
        with st.spinner("Transcribing audio..."):
            raw_input = components["audio"].process_audio(audio_file)
        
        st.success("✅ Transcription complete!")
        raw_input = st.text_area(
            "Transcribed Text (editable):",
            value=raw_input,
            height=100
        )
        
        input_type = "audio"

# Solve button
if raw_input and st.button("🚀 Solve Problem", type="primary"):
    with st.spinner("Processing your problem..."):
        
        # Step 1: Parse
        st.write("### 🔍 Step 1: Parsing Problem")
        parsed = components["parser"].parse(raw_input)
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(parsed, expanded=False)
        with col2:
            st.info(f"**Topic**: {parsed['topic'].title()}")
            if parsed.get('needs_clarification'):
                st.warning(f"⚠️ {parsed['clarification_reason']}")
        
        # Check for similar problems in memory
        similar_problems = components["memory"].get_similar_problems(
            parsed["problem_text"]
        )
        
        if similar_problems:
            with st.expander("💡 Similar Problems Found in Memory"):
                for sp in similar_problems:
                    st.write(f"- {sp['parsed_problem']['problem_text']}")
        
        # Step 2: Solve
        st.write("### 🧮 Step 2: Solving")
        solution = components["solver"].solve(parsed)
        
        # Show retrieved context
        with st.expander("📚 Retrieved Knowledge"):
            for i, ctx in enumerate(solution["retrieved_context"]):
                st.markdown(f"**Source {i+1}** (Score: {ctx['score']:.3f})")
                st.text(ctx["content"])
                st.divider()
        
        # Step 3: Verify
        st.write("### ✅ Step 3: Verification")
        verification = components["verifier"].verify(
            parsed["problem_text"],
            solution["llm_solution"]
        )
        
        col1, col2 = st.columns()[12][1]
        with col1:
            if verification["is_correct"]:
                st.success("✅ Solution Verified")
            else:
                st.error("❌ Issues Found")
            st.metric("Confidence", f"{verification['confidence']*100:.0f}%")
        
        with col2:
            if verification.get("issues"):
                for issue in verification["issues"]:
                    st.warning(f"⚠️ {issue}")
        
        # Step 4: Explain
        st.write("### 📖 Step 4: Explanation")
        explanation = components["explainer"].explain(
            parsed["problem_text"],
            solution["llm_solution"]
        )
        
        st.markdown(explanation)
        
        # SymPy result if available
        if solution["sympy_result"].get("success"):
            st.code(f"SymPy Solution: {solution['sympy_result']['solution']}", language="python")
        
        # Feedback section
        st.write("### 💬 Feedback")
        col1, col2, col3 = st.columns()[12][1]
        
        with col1:
            if st.button("✅ Correct", use_container_width=True):
                memory_id = components["memory"].store_interaction({
                    "original_input": raw_input,
                    "input_type": input_type,
                    "parsed_problem": parsed,
                    "solution": solution["llm_solution"],
                    "verification": verification,
                    "feedback": "correct"
                })
                st.success(f"✅ Feedback saved! (ID: {memory_id[:8]})")
        
        with col2:
            if st.button("❌ Incorrect", use_container_width=True):
                st.session_state["show_feedback_form"] = True
        
        if st.session_state.get("show_feedback_form"):
            user_comment = st.text_input("What's wrong? (optional)")
            if st.button("Submit Feedback"):
                memory_id = components["memory"].store_interaction({
                    "original_input": raw_input,
                    "input_type": input_type,
                    "parsed_problem": parsed,
                    "solution": solution["llm_solution"],
                    "verification": verification,
                    "feedback": "incorrect",
                    "user_comment": user_comment
                })
                st.success(f"✅ Feedback saved! (ID: {memory_id[:8]})")
                st.session_state["show_feedback_form"] = False

# Sidebar
with st.sidebar:
    st.header("📊 System Stats")
    st.metric("Total Problems Solved", len(components["memory"].memories))
    
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
    
    if st.button("🔄 Reset Memory"):
        components["memory"].memories = []
        components["memory"].save_memories()
        st.success("Memory cleared!")
