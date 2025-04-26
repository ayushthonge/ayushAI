# src/quote_detector.py
from typing import List, Dict, Any
from difflib import SequenceMatcher

class QuoteDetector:
    """Detects direct quotes in generated responses"""
    
    def __init__(self, similarity_threshold: float = 0.7, min_phrase_length: int = 15):
        self.similarity_threshold = similarity_threshold
        self.min_phrase_length = min_phrase_length
    
    def contains_quotes(self, response: str, contexts: List[Dict[str, Any]]) -> bool:
        """
        Check if a response contains direct quotes from the contexts
        
        Args:
            response: The generated response text
            contexts: List of context dictionaries with 'content' field
            
        Returns:
            True if direct quotes are detected, False otherwise
        """
        # Extract content from contexts
        context_texts = [ctx["content"] for ctx in contexts if "content" in ctx]
        
        # 1. Check for exact phrase matches
        for context in context_texts:
            words = context.split()
            for i in range(len(words) - 3):  # Look for 4+ word phrases
                phrase = " ".join(words[i:i+4])
                if len(phrase) >= self.min_phrase_length and phrase in response:
                    return True
        
        # 2. Check for high similarity in sentence fragments
        response_sentences = self._split_into_sentences(response)
        for context in context_texts:
            context_sentences = self._split_into_sentences(context)
            
            for resp_sent in response_sentences:
                if len(resp_sent) < self.min_phrase_length:
                    continue
                    
                for ctx_sent in context_sentences:
                    if len(ctx_sent) < self.min_phrase_length:
                        continue
                        
                    similarity = SequenceMatcher(None, resp_sent, ctx_sent).ratio()
                    if similarity > self.similarity_threshold:
                        return True
        
        return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting on punctuation
        for char in ['.', '!', '?']:
            text = text.replace(char, char + '|')
        return [s.strip() for s in text.split('|') if s.strip()]
