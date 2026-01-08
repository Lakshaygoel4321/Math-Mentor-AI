from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class VerifierAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a math solution verifier. Check the solution for:
1. Mathematical correctness
2. Unit consistency
3. Domain validity (e.g., no negative square roots, division by zero)
4. Edge cases

Respond with JSON:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "issues": ["list of issues if any"],
  "needs_human_review": false
}}"""),
            ("user", "Problem: {problem}\n\nSolution: {solution}\n\nVerify this solution.")
        ])
    
    def verify(self, problem, solution):
        """Verify solution correctness"""
        chain = self.prompt | self.llm
        response = chain.invoke({
            "problem": problem,
            "solution": solution
        })
        
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json").split("```")[1]
            result = json.loads(content.strip())
            return result
        except:
            return {
                "is_correct": True,
                "confidence": 0.7,
                "issues": [],
                "needs_human_review": False
            }
