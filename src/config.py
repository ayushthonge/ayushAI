# src/config.py
import os
import json
from typing import Dict, Any

class Config:
    """Manages configuration settings for the RAG system"""
    
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with safe defaults"""
        defaults = {
            "my_name": "Ayush",
            "vector_db_dir": "vector_db",
            "processed_dir": "data/processed",
            "raw_dir": "data/raw",
            "chunk_size": 350,
            "chunk_overlap": 70,
            "gemini_model": "gemini-2.0-flash",
            "temperature": 0.7,
            "top_k": 5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        try:
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
            return {**defaults, **user_config}
        except FileNotFoundError:
            return defaults
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
