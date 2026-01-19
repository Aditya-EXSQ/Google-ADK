import asyncio
import logging

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
langfuse = get_client()
GoogleADKInstrumentor().instrument()
MODEL_ID = LiteLlm("openai/gpt-4o-mini")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParallelWorkflow")


PROS_Agent = LlmAgent(
    name="PROSAgent",
    model=MODEL_ID,
    instruction="""
        You job will be to find the positive aspects of a given input.
        It can advantages, benefits, good points, strengths, etc.
        Write the result to session state under 'PROS'.
    """,
    output_key="PROS",
)

CONS_Agent = LlmAgent(
    name="CONSAgent",
    model=MODEL_ID,
    instruction="""
        You job will be to find the negative aspects of a given input.
        It can disadvantages, drawbacks, bad points, weaknesses, etc.
        Write the result to session state under 'CONS'.
    """,
    output_key="CONS",
)


parallel_stage = ParallelAgent(
    name="ParallelStage",
    sub_agents=[PROS_Agent, CONS_Agent],
)

final_agent = LlmAgent(
    name="FinalSynthesizer",
    model=MODEL_ID,
    instruction="""
    You have the PROS and CONS of a given input from session state.
    Your job is to finalize what will be the overall assessment based on the PROS and CONS.
    Provide a concise summary of the overall assessment."
    """,
)

orchestrator = SequentialAgent(
    name="Orchestrator",
    description="""
    Run the parallel stage, then synthesize the final answer.
    """,
    sub_agents=[parallel_stage, final_agent],
)

async def main():
    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name="ParallelApp",
        user_id="user1",
        session_id="session1",
    )

    runner = Runner(
        agent=orchestrator,
        app_name="ParallelApp",
        session_service=session_service,
    )

    query = "Should i eat pizza on a regular basis?"

    events = runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    )

    async for event in events:
        if event.content and event.content.parts:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
