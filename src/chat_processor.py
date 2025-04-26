# src/chat_processor.py
import asyncio
from typing import Dict, Any, Optional
from rag_model import PersonalRagModel, RagResponse

class ChatProcessor:
    """Processes chat messages and handles response generation"""
    
    def __init__(self, rag_model: PersonalRagModel):
        self.rag_model = rag_model
    
    async def process_message(self, message: str, contact: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single message and generate a response
        
        Args:
            message: The incoming message text
            contact: Optional contact name for context
            
        Returns:
            Dictionary containing the original message, response, and metadata
        """
        # Generate response using RAG model
        response: RagResponse = await self.rag_model.generate_response(message, contact)
        
        # Format result
        result = {
            "original_message": message,
            "response": response.response,
            "contact": response.contact,
            "retrieval_metadata": {
                "num_contexts": len(response.retrieved_context),
                "contexts": [ctx["content"][:100] + "..." for ctx in response.retrieved_context[:3]]
            }
        }
        
        # Include style analysis and content plan if available
        if hasattr(response, "style_analysis") and response.style_analysis:
            result["style_analysis"] = response.style_analysis
        
        if hasattr(response, "content_plan") and response.content_plan:
            result["content_plan"] = response.content_plan
            
        return result
    
    def detect_intent(self, message: str) -> str:
        """Simple intent detection for message categorization"""
        message = message.lower()
        
        # Basic intent detection
        if any(q in message for q in ["?", "who", "what", "when", "where", "why", "how"]):
            return "question"
        if any(g in message for g in ["hello", "hi", "hey", "morning", "afternoon", "evening"]):
            return "greeting"
        if any(t in message for t in ["thanks", "thank you", "appreciated", "grateful"]):
            return "thanks"
        
        return "statement"
