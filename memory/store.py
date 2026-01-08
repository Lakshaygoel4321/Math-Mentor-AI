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
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []
    
    def save_memories(self):
        """Save memories to disk"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
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
        # Simple keyword matching (in production, use embeddings)
        similar = []
        for memory in self.memories:
            if memory.get("parsed_problem", {}).get("problem_text"):
                similarity = self._simple_similarity(
                    problem_text,
                    memory["parsed_problem"]["problem_text"]
                )
                if similarity > 0.3:
                    similar.append((memory, similarity))
        
        similar.sort(key=lambda x: x, reverse=True)[1]
        return [m for m in similar[:limit]]
    
    def _simple_similarity(self, text1, text2):
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / len(words1 | words2)
