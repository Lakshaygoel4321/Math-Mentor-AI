import json
import os
from datetime import datetime
import uuid

class MemoryStore:
    def __init__(self, storage_path="memory/storage.json"):
        self.storage_path = storage_path
        self.memories = self.load_memories()
    
    def load_memories(self):
        """Load memories from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading memories: {e}")
                return []
        return []
    
    def save_memories(self):
        """Save memories to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(self.memories)} memories to {self.storage_path}")
        except Exception as e:
            print(f"❌ Error saving memories: {e}")
    
    def store_interaction(self, data):
        """Store a complete interaction"""
        memory = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "original_input": data.get("original_input"),
            "input_type": data.get("input_type"),  # text/image/audio
            "parsed_problem": data.get("parsed_problem"),
            "solution": data.get("solution"),
            "verification": data.get("verification"),
            "feedback": data.get("feedback"),  # correct/incorrect
            "user_comment": data.get("user_comment", "")
        }
        
        self.memories.append(memory)
        self.save_memories()
        return memory["id"]
    
    def get_similar_problems(self, problem_text, limit=3):
        """Retrieve similar past problems"""
        if not self.memories:
            return []
        
        # Simple keyword matching (in production, use embeddings)
        similar = []
        for memory in self.memories:
            parsed = memory.get("parsed_problem")
            if parsed and isinstance(parsed, dict):
                stored_text = parsed.get("problem_text", "")
                if stored_text:
                    similarity = self._simple_similarity(problem_text, stored_text)
                    if similarity > 0.3:
                        similar.append((memory, similarity))
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N memories (without similarity scores)
        return [m[0] for m in similar[:limit]]
    
    def _simple_similarity(self, text1, text2):
        """Simple word overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_all_memories(self):
        """Get all stored memories"""
        return self.memories
    
    def clear_memories(self):
        """Clear all memories"""
        self.memories = []
        self.save_memories()
        print("✅ All memories cleared")
