import asyncio
import logging

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import agent_tool
from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
langfuse = get_client()
GoogleADKInstrumentor().instrument()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TravelWorkflow")

# --- Constants ---
# Using the same model as in test.py
MODEL_NAME = LiteLlm("openai/gpt-4o-mini")
GoogleADKInstrumentor().instrument()

# --- Agents Definition ---

# Agent 1: Tells best cities in the respective countries
# This agent receives a country name and suggests cities.
city_agent = LlmAgent(
    name="CityAgent",
    model=MODEL_NAME,
    instruction=(
        "You are a knowledgeable travel guide specializing in cities. "
        "When given a country name, suggest the top 3 cities to visit in that country. "
        "For each city, mention one key attraction."
    ),
    output_key="city_recommendations"
)

# Wrap CityAgent as a tool for CountryAgent to use
city_agent_tool = agent_tool.AgentTool(city_agent)

# Agent 2: Tells best countries to visit (and delegates to CityAgent)
# This agent decides on countries and delegates to CityAgent for specific cities.
# country_agent = LlmAgent(
#     name="CountryAgent",
#     model=MODEL_NAME,
#     instruction=(
#         "You are a travel expert specialized in selecting destinations. "
#         "Based on the user's travel preferences, suggest 2 best countries to visit. "
#         "After selecting the countries, you MUST delegate to Key 'CityAgent' to get specific city recommendations for EACH of those countries. "
#         "Present the final itinerary with countries and their respective cities."
#     ),
#     sub_agents=[city_agent]
# )

# Agent 2: Tells best countries to visit (and has a tool for city recommendations which it can call)
country_agent = LlmAgent(
    name="CountryAgent",
    model=MODEL_NAME,
    instruction=(
        "You are a travel expert specialized in selecting destinations. "
        "Based on the user's travel preferences, suggest 2 best countries to visit."
    ),
    output_key="country_recommendations"
)

# Orchestrator Agent
# This agent is the entry point. It manages the high-level request.
orchestrator_agent = SequentialAgent(
    name="OrchestratorAgent",
    description="Orchestrates the travel recommendation workflow.",
    sub_agents=[country_agent, city_agent]
)

# --- Runner Setup & Execution ---
async def main():
    # Setup Session Service
    session_service = InMemorySessionService()
    
    # Create a session
    app_name = "TravelApp"
    user_id = "traveler_1"
    session_id = "session_trip_1"
    
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    
    # Initialize Runner with the Orchestrator
    runner = Runner(
        agent=orchestrator_agent,
        app_name=app_name,
        session_service=session_service
    )

    print("\n--- Travel Workflow Test ---")
    # A generic travel request
    travel_query = "I want to visit a place in Europe that has great beaches and historical ruins."
    
    print(f"User Query: {travel_query}")
    print("-" * 30)

    # Run the workflow
    events = runner.run_async(
        user_id=user_id, 
        session_id=session_id, 
        new_message=types.Content(role="user", parts=[types.Part(text=travel_query)])
    )
    
    # Stream and print the response
    async for event in events:
        if event.content and event.content.parts:
            # Print the text part of the event
            print(f"{event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
