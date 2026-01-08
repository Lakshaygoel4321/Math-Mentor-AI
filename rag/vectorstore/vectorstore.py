from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, knowledge_base_path="rag/knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = None
        
    def build_vectorstore(self):
        """Load documents and create vector store"""
        # Load all text files from knowledge base
        loader = DirectoryLoader(
            self.knowledge_base_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local("rag/vectorstore")
        print(f"✅ Vector store created with {len(chunks)} chunks")
        
    def load_vectorstore(self):
        """Load existing vector store"""
        if os.path.exists("rag/vectorstore"):
            self.vectorstore = FAISS.load_local(
                "rag/vectorstore",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded")
        else:
            print("⚠️ Vector store not found, building new one...")
            self.build_vectorstore()
    
    def retrieve_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        if not self.vectorstore:
            self.load_vectorstore()
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        context = [{"content": doc.page_content, "score": score} 
                   for doc, score in results]
        return context

# Initialize RAG pipeline
if __name__ == "__main__":
    rag = RAGPipeline()
    rag.build_vectorstore()
