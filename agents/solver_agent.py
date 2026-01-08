from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

class SolverAgent:
    def __init__(self, rag_pipeline):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("⚠️ GROQ_API_KEY not found in .env file")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key
        )
        self.rag = rag_pipeline
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert math solver. Use the provided context and solve step-by-step.

Context from knowledge base:
{context}

Provide:
1. Solution approach
2. Step-by-step solution
3. Final answer
4. Python/SymPy code if applicable"""),
            ("user", "Problem: {problem}\nTopic: {topic}")
        ])
    
    def solve(self, parsed_problem):
        """Solve the math problem using RAG + tools"""
        problem_text = parsed_problem["problem_text"]
        topic = parsed_problem["topic"]
        
        # Retrieve relevant context
        try:
            context = self.rag.retrieve_context(f"{topic} {problem_text}", k=3)
            context_text = "\n\n".join([c["content"] for c in context]) if context else "No relevant context found."
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            context = []
            context_text = "No context available."
        
        # Try symbolic solving with SymPy
        sympy_result = self.try_sympy_solve(problem_text, parsed_problem.get("variables", []))
        
        # Get LLM solution
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({
                "problem": problem_text,
                "topic": topic,
                "context": context_text
            })
            
            return {
                "llm_solution": response.content,
                "sympy_result": sympy_result,
                "retrieved_context": context,
                "confidence": 0.85
            }
        except Exception as e:
            print(f"❌ Error in LLM solution: {e}")
            return {
                "llm_solution": f"Error: {str(e)}",
                "sympy_result": sympy_result,
                "retrieved_context": context,
                "confidence": 0.0
            }
    
    def try_sympy_solve(self, problem_text, variables):
        """Attempt to solve using SymPy"""
        try:
            # Simple equation detection
            if "=" in problem_text and any(v in problem_text.lower() for v in ['x', 'y', 'z']):
                # Clean the text
                problem_text = problem_text.replace("^", "**")
                parts = problem_text.split("=")
                
                if len(parts) == 2:
                    lhs = parse_expr(parts[0].strip())
                    rhs = parse_expr(parts[1].strip())
                    equation = sp.Eq(lhs, rhs)
                    
                    # Detect variables
                    vars_in_eq = list(equation.free_symbols)
                    solution = sp.solve(equation, vars_in_eq)
                    
                    return {
                        "success": True,
                        "solution": str(solution),
                        "equation": str(equation)
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        
        return {"success": False, "error": "Could not parse as equation"}
