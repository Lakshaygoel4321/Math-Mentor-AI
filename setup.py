"""
Automatic setup script that runs on Streamlit Cloud deployment
"""
import os
import subprocess

def setup():
    print("🚀 Running setup for Streamlit Cloud...")
    
    # Create directories
    os.makedirs("memory", exist_ok=True)
    os.makedirs("rag/vectorstore", exist_ok=True)
    
    # Run knowledge base setup
    if os.path.exists("create_knowledge_base.py"):
        subprocess.run(["python", "create_knowledge_base.py"])
    
    # Build vectorstore
    if os.path.exists("rag/vectorstore/vectorstore.py"):
        subprocess.run(["python", "rag/vectorstore/vectorstore.py"])
    
    print("✅ Setup complete!")

if __name__ == "__main__":
    setup()
