# src/chat_interface.py
import asyncio
import os
from typing import Optional, List
from .rag_model import PersonalRagModel, RagConfig
from .config import Config

class ChatInterface:
    def __init__(self):
        self.config = Config().config
        rag_config = RagConfig(
            my_name=self.config['my_name'],
            vector_db_dir=self.config['vector_db_dir'],
            gemini_model=self.config['gemini_model'],
            temperature=self.config['temperature'],
            top_k=self.config['top_k']
        )
        self.rag_model = PersonalRagModel(rag_config)
        self.available_contacts = self._get_available_contacts()
        print(f"Initialized chat interface for {self.config['my_name']}")
        print(f"Available contacts: {', '.join(self.available_contacts) if self.available_contacts else 'None'}")

    def _get_available_contacts(self) -> List[str]:
        """Get available contacts from vector stores"""
        vector_db_dir = self.config['vector_db_dir']
        if not os.path.exists(vector_db_dir):
            return []
            
        contacts = []
        for item in os.listdir(vector_db_dir):
            if item.startswith('contact_'):
                contact_name = item[8:]  # Remove 'contact_' prefix
                contacts.append(contact_name)
                
        return contacts

    async def process_message(self, message: str, contact: Optional[str] = None):
        """Process a message and get a response"""
        # Check if contact is valid
        if contact and contact not in self.available_contacts:
            print(f"Warning: Contact '{contact}' not found in available contacts")
        
        # Process message
        return await self.rag_model.generate_response(message, contact)

    async def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "="*50)
        print("WhatsApp AI Chat Interface".center(50))
        print("="*50)
        print("Type 'exit' to quit, 'contact [name]' to set contact, 'contacts' to list contacts")
        
        current_contact = None
        
        while True:
            # Get user input
            prompt = f"[{current_contact}] You: " if current_contact else "You: "
            user_input = input(prompt)
            
            # Check for commands
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'contacts':
                print(f"Available contacts: {', '.join(self.available_contacts)}")
                continue
            elif user_input.lower().startswith('contact '):
                contact_name = user_input[8:].strip()
                if contact_name.lower() == 'none':
                    current_contact = None
                    print("Contact cleared")
                elif contact_name in self.available_contacts:
                    current_contact = contact_name
                    print(f"Contact set to {current_contact}")
                else:
                    print(f"Contact '{contact_name}' not found. Available contacts: {', '.join(self.available_contacts)}")
                continue
            
            # Process message
            try:
                response = await self.process_message(user_input, current_contact)
                print(f"\nResponse: {response.response}")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    interface = ChatInterface()
    asyncio.run(interface.interactive_chat())
