# src/content_planner.py
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

class ContentPlanner:
    """Plans content for responses based on query and context."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    async def create_content_plan(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a content plan for responding to the query."""
        # Extract relevant information from contexts
        context_texts = [ctx["content"] for ctx in contexts]
        context_combined = "\n\n".join(context_texts)
        
        prompt = f"""Based on the following message and retrieved context, create a brief content plan for a response.

MESSAGE TO RESPOND TO:
{query}

RETRIEVED CONTEXT:
{context_combined}

Create a content plan with the following:
1. Key points to address (2-4 bullet points)
2. Relevant facts or information to include
3. Appropriate tone for this specific message
4. Any specific references that would be natural to mention

Format your response as a JSON object with these categories as keys.
"""
        try:
            # Get content plan
            response = await self.llm.ainvoke(prompt)
            json_str = response.content.strip()

            # If wrapped in markdown-style code block
            if "```" in json_str:
                json_str = json_str.split("```")[1].strip()

            return json.loads(json_str)

        except Exception as e:
            print(f"Error parsing content plan: {e}")
            print(f"Raw response: {response.content}")
            return {
                "key_points": ["Respond to the query"],
                "relevant_facts": [],
                "tone": "neutral",
                "references": [],
                "error": str(e)
            }
