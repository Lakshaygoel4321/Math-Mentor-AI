from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

class ParserAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a math problem parser. Your job is to:
1. Clean and structure the input problem
2. Identify the math topic (algebra, calculus, probability, linear algebra)
3. Extract variables and constraints
4. Detect if clarification is needed

Return a JSON with this structure:
{{
  "problem_text": "cleaned problem statement",
  "topic": "algebra|calculus|probability|linear_algebra",
  "variables": ["x", "y"],
  "constraints": ["x > 0"],
  "needs_clarification": false,
  "clarification_reason": ""
}}"""),
            ("user", "Parse this math problem: {input}")
        ])
    
    def parse(self, raw_input):
        """Parse raw input into structured format"""
        chain = self.prompt | self.llm
        response = chain.invoke({"input": raw_input})
        
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json").split("```")[1]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            return parsed
        except:
            # Fallback parsing
            return {
                "problem_text": raw_input,
                "topic": "algebra",
                "variables": [],
                "constraints": [],
                "needs_clarification": False,
                "clarification_reason": ""
            }
