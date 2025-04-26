# src/style_analyzer.py
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.document import Document

class StyleAnalyzer:
    """Analyzes communication style from retrieved messages."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    async def extract_style_features(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract style characteristics from retrieved messages."""
        if not documents:
            return {
                "tone": "neutral",
                "sentence_structure": "varied",
                "vocabulary": "standard",
                "humor_style": "balanced",
                "emoji_usage": "moderate"
            }
        
        # Create prompt for style analysis
        messages_text = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = f"""Analyze the following messages to extract communication style characteristics.
        Do NOT focus on the content or topics, only on HOW the person communicates.
        
        MESSAGES:
        {messages_text}
        
        Extract ONLY the following style characteristics:
        1. Tone (formal, casual, friendly, serious, etc.)
        2. Sentence structure (short/long, simple/complex, fragmented, etc.)
        3. Vocabulary (sophistication level, slang usage, technical terms, etc.)
        4. Humor style (sarcastic, dry, silly, wordplay, etc.)
        5. Emoji usage (frequency and types)
        6. Common phrases or expressions
        7. Punctuation patterns
        8. Response length tendency
        
        Format your response as a JSON object with these categories as keys.
        """
        
        # Get style analysis
        response = await self.llm.ainvoke(prompt)
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            json_str = response.content
            # If the response contains markdown code blocks, extract the JSON
            if "```":
                json_str = json_str.split("```json")[1].split("```")
            elif "```" in json_str:
                json_str = json_str.split("``````")[0].strip()
                
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing style analysis: {e}")
            print(f"Raw response: {response.content}")
            return {
                "tone": "neutral",
                "sentence_structure": "varied",
                "vocabulary": "standard",
                "humor_style": "balanced",
                "emoji_usage": "moderate",
                "error": str(e)
            }
