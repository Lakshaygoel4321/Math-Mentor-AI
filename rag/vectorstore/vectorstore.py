from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Define embedding model name as constant
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster model

# Custom TextLoader with UTF-8 encoding
class UTF8TextLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding='utf-8')

class RAGPipeline:
    def __init__(self, knowledge_base_path="rag/knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        # Use consistent embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        
    def build_vectorstore(self):
        """Load documents and create vector store"""
        print("📚 Loading knowledge base...")
        
        # Load all text files with UTF-8 encoding
        loader = DirectoryLoader(
            self.knowledge_base_path,
            glob="**/*.txt",
            loader_cls=UTF8TextLoader,
            show_progress=True
        )
        
        try:
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} documents")
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return
        
        if not documents:
            print("⚠️ No documents found! Please add .txt files to rag/knowledge_base/")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")
        
        # Create vector store
        print("🔨 Building vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save to disk
        os.makedirs("rag/vectorstore", exist_ok=True)
        self.vectorstore.save_local("rag/vectorstore")
        print(f"✅ Vector store created with {len(chunks)} chunks!")
        print(f"💾 Saved to: rag/vectorstore")
        print(f"📐 Using embedding model: {EMBEDDING_MODEL}")
        
    def load_vectorstore(self):
        """Load existing vector store"""
        if os.path.exists("rag/vectorstore/index.faiss"):
            try:
                print("📂 Loading existing vector store...")
                self.vectorstore = FAISS.load_local(
                    "rag/vectorstore",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading vector store: {e}")
                print("🔄 Rebuilding vector store...")
                self.build_vectorstore()
        else:
            print("⚠️ Vector store not found, building new one...")
            self.build_vectorstore()
    
    def retrieve_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        if not self.vectorstore:
            self.load_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            context = [{"content": doc.page_content, "score": float(score)} 
                       for doc, score in results]
            return context
        except Exception as e:
            print(f"⚠️ Error during retrieval: {e}")
            return []

# Initialize RAG pipeline
if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline Initialization...\n")
    rag = RAGPipeline()
    rag.build_vectorstore()
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    test_query = "quadratic formula"
    results = rag.retrieve_context(test_query, k=2)
    
    if results:
        print(f"\n📝 Query: '{test_query}'")
        print("📊 Top results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
    else:
        print("⚠️ No results found")
