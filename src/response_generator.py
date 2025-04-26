# src/response_generator.py
import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

class ResponseGenerator:
    """Generates stylized responses based on style analysis and content plan."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    async def generate_response(
        self, 
        query: str, 
        style_analysis: Dict[str, Any], 
        content_plan: Dict[str, Any],
        my_name: str
    ) -> str:
        """Generate a stylized response based on style analysis and content plan."""
        
        # Create a detailed prompt for generation
        prompt = f"""You are mimicking the communication style of {my_name} with extreme precision.
        
        MESSAGE TO RESPOND TO:
        {query}
        
        YOUR COMMUNICATION STYLE:
        {json.dumps(style_analysis, indent=2)}
        
        CONTENT PLAN:
        {json.dumps(content_plan, indent=2)}
        
        IMPORTANT INSTRUCTIONS:
        1. Write a response that sounds EXACTLY like {my_name} would write it
        2. Match the tone, sentence structure, vocabulary, and humor style precisely
        3. Use the same patterns of emoji usage and punctuation
        4. Include similar phrases or expressions that {my_name} typically uses
        5. Address the key points from the content plan
        6. DO NOT mention that you're following a style guide or content plan
        7. DO NOT use phrases like "As {my_name}, I would say..." - just write directly as {my_name}
        8. DO NOT quote previous messages verbatim
        
        Write a completely original response that feels 100% authentic to {my_name}'s style.
        """
        
        # Generate response
        response = await self.llm.ainvoke(prompt)
        
        return response.content
