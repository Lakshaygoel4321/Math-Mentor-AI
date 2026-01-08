from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import traceback

class SolverAgent:
    def __init__(self, rag_pipeline):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
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
        context = self.rag.retrieve_context(f"{topic} {problem_text}", k=3)
        context_text = "\n\n".join([c["content"] for c in context])
        
        # Try symbolic solving with SymPy
        sympy_result = self.try_sympy_solve(problem_text, parsed_problem.get("variables", []))
        
        # Get LLM solution
        chain = self.prompt | self.llm
        response = chain.invoke({
            "problem": problem_text,
            "topic": topic,
            "context": context_text
        })
        
        return {
            "llm_solution": response.content,
            "sympy_result": sympy_result,
            "retrieved_context": context,
            "confidence": 0.85  # Placeholder
        }
    
    def try_sympy_solve(self, problem_text, variables):
        """Attempt to solve using SymPy"""
        try:
            # Simple equation detection
            if "=" in problem_text and any(v in problem_text for v in ['x', 'y', 'z']):
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
