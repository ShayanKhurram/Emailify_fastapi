from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
import asyncio
import httpx
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from google.genai import Client as GeminiClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.mcp import MCPServerSSE

load_dotenv()

# Global variables for agent and dependencies
agent_instance = None
deps_instance = None

mail_server = MCPServerSSE(url='https://email-mcp.onrender.com/sse')  

class PydanticAIDeps(BaseModel):
    """Dependencies for PydanticAI agent with arbitrary types allowed"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    supabase: Optional[Client] = None
    gemini_client: Optional[GeminiClient] = None

# Pydantic models for API requests/responses
class AgentRequest(BaseModel):
    message: str
    use_message_history: bool = False
    message_history: Optional[List[Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    output: str
    success: bool
    error: Optional[str] = None
    new_messages: Optional[List[Dict[str, Any]]] = None

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    dependencies_ready: bool

# Initialize agent and dependencies
async def initialize_agent():
    global agent_instance, deps_instance
    
    try:
        # Initialize model
        llm = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
        model = GeminiModel(llm)
        
        # Initialize clients
        gemini_client = None
        supabase = None
        
        # Try to initialize Gemini client
        try:
            gemini_client = genai.Client()
            print("Gemini client initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini client: {e}")
        
        # Try to initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if supabase_url and supabase_key:
            try:
                supabase = create_client(supabase_url, supabase_key)
                print("Supabase client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Supabase client: {e}")
        else:
            print("Warning: SUPABASE_URL and SUPABASE_ANON_KEY not set")
        
        # Create dependencies
        deps_instance = PydanticAIDeps(
            supabase=supabase,
            gemini_client=gemini_client
        )
        
        # System prompt
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
- End with a clear, polite call to action

Keep each email under 150 words, and ensure it feels custom-written.
"""
        
        # Create agent
        agent_instance = Agent(
            model,
            system_prompt=system_prompt,
            mcp_servers=["mail_server"],
            deps_type=PydanticAIDeps,
            retries=2
        )
        
        # Add tools to agent
        add_tools_to_agent()
        
        print("Agent initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_tools_to_agent():
    """Add tools to the agent instance"""
    
    @agent_instance.tool
    async def get_embedding(ctx: RunContext[PydanticAIDeps],text: str) -> List[float]:
        """Get embedding vector from Gemini."""
        try:
            if not ctx.deps.gemini_client:
                print("Gemini client not available")
                return [0] * 768  # Default embedding size
                
            response = ctx.deps.gemini_client.models.embed_content(
                model="gemini-embedding-exp-03-0",
                contents=text
            )
            return response.embeddings
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 768

    @agent_instance.tool
    async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
        """Retrieve relevant website chunks based on the query with RAG."""
        try:
            if not ctx.deps.supabase:
                return "Documentation retrieval not available (Supabase not configured)."
                
            query_embedding = await get_embedding(ctx,user_query)
            
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
                chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                formatted_chunks.append(chunk_text)
                
            return "\n\n---\n\n".join(formatted_chunks)
            
        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"

    @agent_instance.tool
    async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
        """Retrieve a list of all available website pages."""
        try:
            if not ctx.deps.supabase:
                return ["Documentation not available (Supabase not configured)"]
                
            result = ctx.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'the_charles_nyc_docs') \
                .execute()
            
            if not result.data:
                return []
                
            urls = sorted(set(doc['url'] for doc in result.data))
            return urls
            
        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return []

    @agent_instance.tool
    async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
        """Retrieve the full content of a specific website page."""
        try:
            if not ctx.deps.supabase:
                return "Page content not available (Supabase not configured)."
                
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
            
            for chunk in result.data:
                formatted_content.append(chunk['content'])
                
            return "\n\n".join(formatted_content)
            
        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI server...")
    success = await initialize_agent()
    if not success:
        print("Warning: Agent initialization failed, but continuing...")
    
    yield
    
    # Shutdown
    print("Shutting down FastAPI server...")

# Create FastAPI app
app = FastAPI(
    title="PydanticAI Agent API",
    description="FastAPI server for running PydanticAI agent with lead outreach capabilities",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent_ready=agent_instance is not None,
        dependencies_ready=deps_instance is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy" if agent_instance else "unhealthy",
        agent_ready=agent_instance is not None,
        dependencies_ready=deps_instance is not None
    )

@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """Main endpoint to interact with the PydanticAI agent"""
    
    if not agent_instance:
        return AgentResponse(
            output="Agent not available. Please check environment variables and initialization.",
            success=False,
            error="Agent not initialized"
        )
    
    try:
        # Run the agent with the provided message
        if request.use_message_history and request.message_history:
            result = await agent_instance.run(
                request.message,
                message_history=request.message_history,
                deps=deps_instance
            )
        else:
            result = await agent_instance.run(
                request.message,
                deps=deps_instance
            )
        
        return AgentResponse(
            output=result.output,
            success=True,
            new_messages=result.new_messages() if hasattr(result, 'new_messages') else None
        )
        
    except Exception as e:
        print(f"Error running agent: {e}")
        return AgentResponse(
            output="",
            success=False,
            error=str(e)
        )

@app.post("/chat-simple")
async def chat_simple(message: str):
    """Simplified endpoint that just takes a message string"""
    
    if not agent_instance:
        return {
            "response": "Agent not available. Please check environment variables and initialization.",
            "error": "Agent not initialized"
        }
    
    try:
        result = await agent_instance.run(message, deps=deps_instance)
        return {"response": result.output}
        
    except Exception as e:
        print(f"Error running agent: {e}")
        return {"response": "Error occurred", "error": str(e)}

# Run the server
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting server on port {port}")
    
    uvicorn.run(
        app,  # Pass the app directly
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
