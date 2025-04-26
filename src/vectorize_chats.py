# src/vectorize_chats.py
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

class ChatVectorizer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vectorizer with configuration
        
        Args:
            config: Should contain:
                - vector_db_dir: Path to store vector databases
                - processed_dir: Path to processed CSV files
                - chunk_size: Text chunk size (300-500 recommended)
                - chunk_overlap: Chunk overlap size (20% of chunk_size)
                - my_name: Your name as it appears in chat data
        """
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", "(Media omitted)", "|", ". ", "! ", "? "]
        )
        
        # Ensure directories exist
        os.makedirs(config['vector_db_dir'], exist_ok=True)

    def _process_message(self, row: pd.Series) -> List[Document]:
        """Convert a single message row into chunked documents"""
        message = str(row['message']).strip()
        if not message or '(Media omitted)' in message:
            return []

        # Base metadata
        metadata = {
            'sender': str(row.get('sender', 'Unknown')),
            'timestamp': str(row.get('timestamp', '')),
            'contact': str(row.get('contact', 'Unknown')),
            'is_my_message': row['sender'] == self.config['my_name'],
            'original_length': len(message),
            'tone': self._detect_tone(message)
        }

        # Split message into chunks
        chunks = self.text_splitter.split_text(message)
        
        # Create documents with chunk metadata
        documents = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': idx,
                'total_chunks': len(chunks),
                'chunk_text': chunk  # Store original chunk text for verification
            })
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            
        return documents

    def _detect_tone(self, text: str) -> str:
        """Simple tone detection for style analysis"""
        text = text.lower()
        tone_keywords = {
            'humorous': ['haha', 'lol', '😂', '😆'],
            'polite': ['please', 'thank you', 'appreciate'],
            'urgent': ['asap', 'urgent', 'important'],
            'question': ['?', 'what', 'when', 'where', 'why', 'how']
        }
        
        for tone, keywords in tone_keywords.items():
            if any(kw in text for kw in keywords):
                return tone
        return 'neutral'

    def _process_dataframe(self, df: pd.DataFrame, store_name: str) -> bool:
        """Process a DataFrame and create vector store"""
        if df.empty:
            print(f"⚠️ Skipping {store_name} - empty DataFrame")
            return False

        # Clean data
        df = df.dropna(subset=['message'])
        df['message'] = df['message'].astype(str).str.strip()
        df = df[df['message'] != '']
        
        if df.empty:
            print(f"⚠️ Skipping {store_name} - no valid messages after cleaning")
            return False

        # Process all messages into documents
        all_documents = []
        for _, row in df.iterrows():
            all_documents.extend(self._process_message(row))
            
        if not all_documents:
            print(f"⚠️ Skipping {store_name} - no valid chunks generated")
            return False

        # Create and persist vector store
        try:
            vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=os.path.join(self.config['vector_db_dir'], store_name)
            )
            vectorstore.persist()
            print(f"✅ Created {store_name} with {len(all_documents)} chunks")
            return True
        except Exception as e:
            print(f"❌ Failed to create {store_name}: {str(e)}")
            return False

    def vectorize_all_messages(self):
        """Main vectorization pipeline"""
        print("\n" + "="*50)
        print("Starting Vectorization Process".center(50))
        print("="*50)
        
        # Vectorize complete chat history
        all_chats_path = os.path.join(self.config['processed_dir'], 'all_chats.csv')
        if os.path.exists(all_chats_path):
            print("\nVectorizing complete chat history...")
            df = pd.read_csv(all_chats_path)
            self._process_dataframe(df, 'all_messages')
        else:
            print("⚠️ all_chats.csv not found")

        # Vectorize user's personal messages
        my_messages_path = os.path.join(self.config['processed_dir'], 'my_messages.csv')
        if os.path.exists(my_messages_path):
            print("\nVectorizing personal messages for style analysis...")
            df = pd.read_csv(my_messages_path)
            self._process_dataframe(df, 'my_messages')
        else:
            print("⚠️ my_messages.csv not found")

        # Vectorize contact-specific messages
        print("\nProcessing contact-specific messages...")
        contacts_dir = os.path.join(self.config['processed_dir'], 'by_contact')
        if os.path.exists(contacts_dir):
            for file in os.listdir(contacts_dir):
                if file.endswith('.csv'):
                    contact_name = os.path.splitext(file)[0]
                    file_path = os.path.join(contacts_dir, file)
                    print(f"\nProcessing {contact_name}...")
                    df = pd.read_csv(file_path)
                    self._process_dataframe(df, f'contact_{contact_name}')
        else:
            print("⚠️ by_contact directory not found")

        print("\n" + "="*50)
        print("Vectorization Complete!".center(50))
        print("="*50)

if __name__ == "__main__":
    # Configuration - Update these values according to your setup
    CONFIG = {
        'vector_db_dir': r'C:\ayushai\vector_db',
        'processed_dir': r'C:\ayushai\data\processed',
        'chunk_size': 350,       # Optimal for style analysis
        'chunk_overlap': 70,     # ~20% of chunk_size
        'my_name': 'Ayush'       # Your name as it appears in chats
    }
    
    vectorizer = ChatVectorizer(CONFIG)
    vectorizer.vectorize_all_messages()
