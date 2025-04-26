# src/run_vectorization.py
import os
import json
from vectorize_chats import ChatVectorizer

def main():
    """Entry point for vectorization process"""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create vectorizer with configuration
    vectorizer = ChatVectorizer({
        'vector_db_dir': config['vector_db_dir'],
        'processed_dir': config['processed_dir'],
        'chunk_size': config.get('chunk_size', 350),
        'chunk_overlap': config.get('chunk_overlap', 70),
        'my_name': config['my_name']
    })
    
    # Run vectorization process
    print("Starting vectorization process...")
    vectorizer.vectorize_all_messages()
    print("Vectorization completed!")

if __name__ == "__main__":
    main()
