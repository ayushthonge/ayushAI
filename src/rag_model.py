# src/rag_model.py
import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.document import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Relative imports
from .style_analyzer import StyleAnalyzer
from .content_planner import ContentPlanner
from .response_generator import ResponseGenerator
from .quote_detector import QuoteDetector

load_dotenv()

class RagConfig(BaseModel):
    my_name: str = Field(description="Your name as it appears in the WhatsApp chats")
    vector_db_dir: str = Field(default="vector_db", description="Directory containing vector databases")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    
class RagResponse(BaseModel):
    query: str = Field(description="Original query")
    contact: Optional[str] = Field(default=None, description="Contact name if specified")
    response: str = Field(description="Generated response")
    retrieved_context: List[Dict[str, Any]] = Field(description="Retrieved context used for generation")
    style_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Style analysis results")
    content_plan: Optional[Dict[str, Any]] = Field(default=None, description="Content plan")

class PersonalRagModel:
    def __init__(self, config: RagConfig):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize vector stores
        self.global_vectorstore = self._load_vectorstore('all_messages')
        self.my_messages_vectorstore = self._load_vectorstore('my_messages')
        self.contact_vectorstores = {}
        self._load_contact_vectorstores()
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        # Initialize components
        self.style_analyzer = StyleAnalyzer(self.llm)
        self.content_planner = ContentPlanner(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        self.quote_detector = QuoteDetector()

    def _load_vectorstore(self, name: str):
        store_path = os.path.join(self.config.vector_db_dir, name)
        if not os.path.exists(store_path):
            print(f"Warning: Vector store {name} does not exist at {store_path}")
            return None
        return Chroma(
            persist_directory=store_path,
            embedding_function=self.embedding_model
        )

    def _load_contact_vectorstores(self):
        for item in os.listdir(self.config.vector_db_dir):
            if item.startswith('contact_'):
                contact_name = item.replace('contact_', '')
                store_path = os.path.join(self.config.vector_db_dir, item)
                self.contact_vectorstores[contact_name] = Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embedding_model
                )
        print(f"Loaded {len(self.contact_vectorstores)} contact-specific vector stores")

    async def _retrieve_relevant_contexts(self, query: str, contact: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts from vector stores based on query and optional contact."""
        all_contexts = []
        
        # Retrieve from my_messages for style reference
        if self.my_messages_vectorstore:
            docs = self.my_messages_vectorstore.similarity_search(query, k=self.config.top_k)
            all_contexts.extend([{
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "my_messages"
            } for doc in docs])
        
        # Retrieve from contact-specific store if provided
        if contact and contact in self.contact_vectorstores:
            docs = self.contact_vectorstores[contact].similarity_search(query, k=self.config.top_k)
            all_contexts.extend([{
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": f"contact_{contact}"
            } for doc in docs])
        
        # Retrieve from global store as fallback
        if self.global_vectorstore:
            docs = self.global_vectorstore.similarity_search(query, k=self.config.top_k)
            all_contexts.extend([{
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "all_messages"
            } for doc in docs])
        
        return all_contexts

    async def generate_response(self, query: str, contact: Optional[str] = None) -> RagResponse:
        contexts = await self._retrieve_relevant_contexts(query, contact)
        if not contexts:
            return RagResponse(
                query=query,
                contact=contact,
                response="Error: No relevant contexts found",
                retrieved_context=[],
                style_analysis={},
                content_plan={}
            )
        
        my_messages_docs = [
            Document(page_content=ctx["content"], metadata=ctx["metadata"])
            for ctx in contexts if ctx["source"] == "my_messages"
        ]
        style_analysis = await self.style_analyzer.extract_style_features(my_messages_docs)
        content_plan = await self.content_planner.create_content_plan(query, contexts)
        response_content = await self.response_generator.generate_response(
            query, style_analysis, content_plan, self.config.my_name
        )
        
        if self.quote_detector.contains_quotes(response_content, contexts):
            original_temp = self.llm.temperature
            self.llm.temperature = min(original_temp + 0.2, 0.95)
            content_plan["avoid_quoting"] = True
            response_content = await self.response_generator.generate_response(
                query, style_analysis, content_plan, self.config.my_name
            )
            self.llm.temperature = original_temp
        
        return RagResponse(
            query=query,
            contact=contact,
            response=response_content,
            retrieved_context=contexts,
            style_analysis=style_analysis,
            content_plan=content_plan
        )
