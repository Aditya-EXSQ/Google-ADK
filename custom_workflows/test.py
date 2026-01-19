import asyncio
import logging
import datetime
from typing_extensions import override
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from langfuse import get_client
import python_weather

from dotenv import load_dotenv
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HierarchicalSystem")

# --- Constants ---
MODEL_NAME = LiteLlm("openai/gpt-4o-mini")
langfuse = get_client()
GoogleADKInstrumentor().instrument()

# --- Tools ---
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtracts the second integer from the first."""
    return a - b

def get_current_time() -> str:
    """Returns the current system time as a string."""
    return datetime.datetime.now().strftime("%H:%M:%S")

async def tell_weather(city: str) -> str:
    """Fetches the current weather for a given city."""
    client = python_weather.Client(unit=python_weather.IMPERIAL)
    weather = await client.get(city)
    await client.close()
    return f"The current temperature in {city} is {weather.temperature}°F."
# --- Level 3: Specialist Agents ---
# Each performs a single task and writes to session state via output_key.

add_agent = LlmAgent(
    name="AddAgent",
    model=MODEL_NAME,
    instruction="You are an addition specialist. Use the `add` tool to process the request. Output the result into the session state.",
    tools=[add],
    output_key="calc_result"
)

subtract_agent = LlmAgent(
    name="SubtractAgent",
    model=MODEL_NAME,
    instruction="You are a subtraction specialist. Use the `subtract` tool to process the request. Output the result into the session state.",
    tools=[subtract],
    output_key="calc_result"
)

time_agent = LlmAgent(
    name="TimeAgent",
    model=MODEL_NAME,
    instruction="You are a time specialist. Use the `get_current_time` tool to get the current time. Output the result into the session state.",
    tools=[get_current_time],
    output_key="time_result"
)

arithmetic_agent = LlmAgent(
    name="ArithmeticAgent",
    model=MODEL_NAME,
    instruction="You are an arithmetic specialist. You can perform addition and subtraction. Use the appropriate tool based on the user's request. Output the result into the session state.",
    tools=[add, subtract],
    output_key="calc_result"
)

time_weather_agent = LlmAgent(
    name="TimeWeatherAgent",
    model=MODEL_NAME,
    instruction="You are a time and weather specialist. You can provide the current time and weather information. Use the appropriate tool based on the user's request. Output the result into the session state.",
    tools=[get_current_time, tell_weather],
    output_key="time_result"
)
# --- Level 2: Branch Orchestrators ---
# LlmAgents with sub_agents for automatic delegation.

math_orchestrator = LlmAgent(
    name="MathOrchestrator",
    model=MODEL_NAME,
    # instruction="You are a math manager. You have two sub-agents: AddAgent and SubtractAgent. Delegate the user's math request to the appropriate agent.",
    # sub_agents=[add_agent, subtract_agent]
    instruction="You are a math manager. You have a sub-agent ArithmeticAgent. Delegate the user's math request to the ArithmeticAgent.",
    sub_agents=[arithmetic_agent]
)

# time_orchestrator = LlmAgent(
#     name="TimeOrchestrator",
#     model=MODEL_NAME,
#     instruction="You are a time manager. You have a sub-agent TimeAgent. Delegate the user's time request to the TimeAgent.",
#     sub_agents=[time_agent]
# )

time_orchestrator = LlmAgent(
    name="TimeOrchestrator",
    model=MODEL_NAME,
    instruction="You are a time and weather manager. You have a sub-agent TimeWeatherAgent. Delegate the user's time or weather request to the TimeWeatherAgent.",
    sub_agents=[time_weather_agent]
)


# --- Level 1: Master Orchestrator ---
class MasterOrchestrator(BaseAgent):
    """
    Master Orchestrator that deterministically routes requests to branches
    and emits a final response.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    math_branch: LlmAgent
    time_branch: LlmAgent

    def __init__(self, name: str, math_branch: LlmAgent, time_branch: LlmAgent):
        super().__init__(
            name=name,
            math_branch=math_branch,
            time_branch=time_branch,
            sub_agents=[math_branch, time_branch]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 1. Inspect Input for Deterministic Routing
        # Use simple keyword matching as "Explicitly decides" logic
        
        user_input = ""
        if ctx.user_content and ctx.user_content.parts:
             user_input = ctx.user_content.parts[0].text or ""
        
        user_input_lower = user_input.lower()
        
        logger.info(f"[{self.name}] User Input: {user_input}")

        if "time" in user_input_lower:
            logger.info(f"[{self.name}] Routing to Time Branch")
            async for event in self.time_branch.run_async(ctx):
                yield event
            
            # Inspect session state
            result = ctx.session.state.get("time_result", "Unknown Time")
            final_text = f"The time is: {result}"

        elif "add" in user_input_lower or "subtract" in user_input_lower or "+" in user_input_lower or "-" in user_input_lower:
            logger.info(f"[{self.name}] Routing to Math Branch")
            async for event in self.math_branch.run_async(ctx):
                yield event
            
            # Inspect session state
            result = ctx.session.state.get("calc_result", "Unknown Result")
            final_text = f"The calculated result is: {result}"
        
        else:
            final_text = "I can only help with Math or Time requests."

        # 2. Emit Final Response
        logger.info(f"[{self.name}] Final Response: {final_text}")
        
        # Construct content
        content = types.Content(role="model", parts=[types.Part(text=final_text)])
        
        yield Event(author=self.name, content=content)

# --- Runner Setup & Execution ---
async def main():
    # Initialize Master Orchestrator
    master = MasterOrchestrator(
        name="MasterOrchestrator",
        math_branch=math_orchestrator,
        time_branch=time_orchestrator
    )

    # Setup Session
    session_service = InMemorySessionService()
    # Create sessions explicity
    await session_service.create_session(app_name="HierarchicalApp", user_id="user1", session_id="session1")
    await session_service.create_session(app_name="HierarchicalApp", user_id="user1", session_id="session2")
    await session_service.create_session(app_name="HierarchicalApp", user_id="user2", session_id="session3")
    
    runner = Runner(
        agent=master,
        app_name="HierarchicalApp",
        session_service=session_service
    )

    # Test Case 1: Math (Addition)
    print("\n--- Test Case 1: Math (Addition) ---")
    math_query = "What is 9 added to 5 and then 13 subtracted from the result?"
    events = runner.run_async(
        user_id="user1", 
        session_id="session1", 
        new_message=types.Content(role="user", parts=[types.Part(text=math_query)])
    )
    async for event in events:
        if event.content and event.content.parts:
            print(f"{event.content.parts[0].text}")

    # Test Case 2: Time
    print("\n--- Test Case 2: Time ---")
    time_query = "What is the time?"
    events = runner.run_async(
        user_id="user1", 
        session_id="session2", 
        new_message=types.Content(role="user", parts=[types.Part(text=time_query)])
    )
    async for event in events:
        if event.content and event.content.parts:
            print(f"{event.content.parts[0].text}")

    # Test Case 3: Weather
    print("\n--- Test Case 3: Weather ---")
    weather_query = "What is the time in and weather in New York. After fetching time and weather, add the temperature in fahreinheit and minutes of the times fetched.?"
    events = runner.run_async(
        user_id="user2",
        session_id="session3", 
        new_message=types.Content(role="user", parts=[types.Part(text=weather_query)])
    )
    async for event in events:
        if event.content and event.content.parts:
            print(f"{event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
