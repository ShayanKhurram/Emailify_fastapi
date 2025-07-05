from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Optional
from google.genai import Client
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.mcp import MCPServerStdio, MCPServerSSE

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Lead Outreach Assistant API")

# Your existing setup
llm = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
model = GeminiModel(llm)
gemini_client = genai.Client()

@dataclass
class PydanticAIDeps:
    supabase: Client
    gemini_client: Client

system_prompt = """
You are a lead outreach assistant. You receive a list of lead records, each including:

id: Unique lead identifier
email: The lead's email address
company: The organization the lead works at
job_title: The lead's position or title
industry: The industry in which the lead's company operates
lead_source: Where the lead was acquired from (e.g., LinkedIn, event, website)
pain_point: A problem or challenge the lead is facing

Your job is to write a personalized cold or warm outreach email for each lead. The email should:
- Begin with a natural, personalized greeting
- Reference their job_title and company to show relevance
- Address the specific pain_point they are facing
- Offer a solution or value proposition
- Match the tone with the lead_source
- End with a clear call to action
Respond only with the email content (including subject line and body).
Keep each email under 150 words.
"""
mail_server = MCPServerSSE(url='https://email-mcp.onrender.com/sse')  

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    mcp_servers=[mail_server]
)

# Request model for FastAPI
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Response model
class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    """Endpoint to interact with the lead outreach assistant"""
    try:
        result = await pydantic_ai_expert.run(request.message)
        return {
            "response": result.output,
            "session_id": request.session_id or "new_session",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Your existing tools
async def get_embedding(text: str, gemini_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from Gemini."""
    try:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text)
        return response.embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """Retrieve relevant documentation chunks with RAG."""
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.gemini_client)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'the_charles_nyc_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"# {doc['title']}\n\n{doc['content']}"
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """Retrieve list of all available documentation pages."""
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'the_charles_nyc_docs') \
            .execute()
        return sorted(set(doc['url'] for doc in result.data))
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """Retrieve full content of a specific documentation page."""
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'the_charles_nyc_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        formatted_content.extend(chunk['content'] for chunk in result.data)
        return "\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

# Server management
async def start_mcp_servers():
    """Start MCP servers in background"""
    async with pydantic_ai_expert.run_mcp_servers():
        print("MCP servers started in background")
        while True:
            await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    """Initialize services when app starts"""
    asyncio.create_task(start_mcp_servers())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": llm}

# Run with: uvicorn your_filename:app --reload
