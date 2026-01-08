from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

class ExplainerAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly math tutor. Explain the solution in a clear, student-friendly way.

Structure your explanation:
1. **Problem Understanding**: Rephrase what we're solving
2. **Solution Approach**: Explain the strategy
3. **Step-by-Step Solution**: Break down each step with reasoning
4. **Final Answer**: Clear statement of the result
5. **Key Concepts**: What formulas/concepts were used

Use simple language, avoid jargon, and be encouraging."""),
            ("user", "Problem: {problem}\n\nSolution: {solution}\n\nExplain this clearly.")
        ])
    
    def explain(self, problem, solution):
        """Generate student-friendly explanation"""
        chain = self.prompt | self.llm
        response = chain.invoke({
            "problem": problem,
            "solution": solution
        })
        
        return response.content
