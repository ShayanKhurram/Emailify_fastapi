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
from typing import List
from google.genai import Client
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.mcp import MCPServerStdio,MCPServerSSE

load_dotenv()

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

email: The lead’s email address

company: The organization the lead works at

job_title: The lead’s position or title

industry: The industry in which the lead’s company operates

lead_source: Where the lead was acquired from (e.g., LinkedIn, event, website)

pain_point: A problem or challenge the lead is facing

Your job is to write a personalized cold or warm outreach email for each lead. The email should:

Begin with a natural, personalized greeting (e.g., "Hi Sarah," or "Hello Mr. Khan,")

Reference their job_title and company to show relevance.

Address the specific pain_point they are facing, demonstrating understanding.

Offer a solution or value proposition that relates directly to the pain point.

Match the tone of the email with the lead_source:

Friendly/informal for social sources (e.g., social media, website)

Formal/professional for business sources (e.g., LinkedIn, conferences)

End with a clear, polite call to action (e.g., suggesting a quick call or reply).

Respond only with the email content (including subject line and body), not with explanations or lead details.

Do not generate emails for leads missing key fields (like email, job_title, or pain_point).
Keep each email under 150 words, and ensure it feels custom-written.
"""
mail_server = MCPServerSSE(url='https://email-mcp.onrender.com/sse')  

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    mcp_servers=[mail_server]
)

async def get_embedding(text: str, gemini_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from Gemini."""
    try:
       response = gemini_client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text)

       return response.embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and Gemini client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.gemini_client)
        
        # Query Supabase for relevant documents
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
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'the_charles_nyc_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'the_charles_nyc_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
# async def main():
#     result = await pydantic_ai_expert.run("Hi, can you help me with my lead outreach emails?")
#     print(result.output)


# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\nSession interrupted. Goodbye!")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         print("Cleaning up...")
#         asyncio.run(asyncio.sleep(0.2))  # Allow cleanup time   


async def main():
    print("Starting MCP servers...")
    
    try:
        async with pydantic_ai_expert.run_mcp_servers():
            print("MCP servers started successfully!")
            
            # Initial query
            print("Sending initial query...")
        async with pydantic_ai_expert.run_mcp_servers():
            result = await pydantic_ai_expert.run('Hi can you help me with my lead outreach emails?')
            print("Agent response:")
            print(result.output)
            
            print("\n" + "="*50)
            print("Starting interactive session...")
            print("Type 'exit', 'quit', or 'bye' to end the session")
            print("="*50)
            
            # Interactive loop
            while True:
                try:
                    user_input = input("\n> ")
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("Goodbye!")
                        break
                    
                    if user_input.strip() == "":
                        print("Please enter a question or command.")
                        continue
                    
                    print("Processing your request...")
                    result = await pydantic_ai_expert.run(user_input, message_history=result.new_messages())
                    print("Agent response:")
                    print(result.output)
                    
                except KeyboardInterrupt:
                    print("\nSession interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"Error processing request: {e}")
                    print("Please try again or type 'exit' to quit.")
                    
    except Exception as e:
        print(f"Failed to start MCP servers: {e}")
        print("Make sure you have the required dependencies installed:")
        print("- npm/npx should be available")
        print("- Check your internet connection")
        
    finally:
        print("Cleaning up...")
        await asyncio.sleep(0.2)  # Allow cleanup time


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("Program ended.")