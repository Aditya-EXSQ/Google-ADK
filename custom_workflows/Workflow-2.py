import asyncio
from typing import Dict

import aiohttp

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
langfuse = get_client()
GoogleADKInstrumentor().instrument()
MODEL_ID = LiteLlm("openai/gpt-4o-mini")

# -----------------------------
# Tool 1: Call Agify (age)
# -----------------------------
async def fetch_age(name: str) -> Dict:
    """
    Predict age for a name using agify.io.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.agify.io",
            params={"name": name},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()

    return {
        "age": data.get("age"),
        "count": data.get("count"),
    }


# -----------------------------
# Tool 2: Call Genderize (gender)
# -----------------------------
async def fetch_gender(name: str) -> Dict:
    """
    Predict gender for a name using genderize.io.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.genderize.io",
            params={"name": name},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()

    return {
        "gender": data.get("gender"),
        "probability": data.get("probability"),
    }


# -------------------------------------------------
# Coordinator Tool (PARALLEL EXECUTION)
# -------------------------------------------------
async def fetch_demographics(name: str) -> Dict:
    """
    Fetch age and gender predictions in parallel.
    """
    age_task = fetch_age(name)
    gender_task = fetch_gender(name)

    age, gender = await asyncio.gather(
        age_task,
        gender_task,
    )

    return {
        "name": name,
        "age_prediction": age,
        "gender_prediction": gender,
    }


# -----------------------------
# Agent Definition
# -----------------------------
agent = Agent(
    name="demographics_agent",
    model=MODEL_ID,
    tools=[
        fetch_age,
        fetch_gender,
        fetch_demographics,  # coordinator tool
    ],
)


# -----------------------------
# Example invocation
# -----------------------------
async def main():
    # -----------------------------
    # Session setup
    # -----------------------------
    session_service = InMemorySessionService()

    app_name = "AgeGenderApp"
    user_id = "user_1"
    session_id = "session_1"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # -----------------------------
    # Runner
    # -----------------------------
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    print("\n--- Demographics Workflow Test ---")

    query = "For the name Aditya, predict age and gender."
    print(f"User Query: {query}")
    print("-" * 40)

    # -----------------------------
    # Run + stream events
    # -----------------------------
    events = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    )

    async for event in events:
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if hasattr(part, "text") and part.text:
                print(part.text)


if __name__ == "__main__":
    asyncio.run(main())
