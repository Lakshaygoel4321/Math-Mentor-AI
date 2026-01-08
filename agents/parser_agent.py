from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv

load_dotenv()

class ParserAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("⚠️ GROQ_API_KEY not found in .env file")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a math problem parser. Your job is to:
1. Clean and structure the input problem
2. Identify the math topic (algebra, calculus, probability, linear_algebra)
3. Extract variables and constraints
4. Detect if clarification is needed

Return ONLY a valid JSON object with this exact structure:
{
  "problem_text": "cleaned problem statement",
  "topic": "algebra",
  "variables": ["x", "y"],
  "constraints": ["x > 0"],
  "needs_clarification": false,
  "clarification_reason": ""
}

Topic must be one of: algebra, calculus, probability, linear_algebra"""),
            ("user", "Parse this math problem: {input}")
        ])
    
    def parse(self, raw_input):
        """Parse raw input into structured format"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"input": raw_input})
            
            # Extract JSON from response
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json").split("```").strip()[1]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Validate required fields
            if not parsed.get("problem_text"):
                parsed["problem_text"] = raw_input
            if not parsed.get("topic"):
                parsed["topic"] = "algebra"
            if not isinstance(parsed.get("variables"), list):
                parsed["variables"] = []
            if not isinstance(parsed.get("constraints"), list):
                parsed["constraints"] = []
            if "needs_clarification" not in parsed:
                parsed["needs_clarification"] = False
            if not parsed.get("clarification_reason"):
                parsed["clarification_reason"] = ""
            
            return parsed
            
        except Exception as e:
            print(f"⚠️ Parser error: {e}")
            # Return fallback structure
            return {
                "problem_text": raw_input,
                "topic": "algebra",
                "variables": [],
                "constraints": [],
                "needs_clarification": False,
                "clarification_reason": ""
            }
