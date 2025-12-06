from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from groq import Groq
import os
import logging
import json
from dotenv import load_dotenv

# --------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API"))
model = "openai/gpt-oss-120b"

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Data models (unchanged)
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    description: str
    is_calendar_event: bool
    confidence_score: float


class EventDetails(BaseModel):
    name: str
    date: str
    duration_minutes: int
    participants: List[str]


class EventConfirmation(BaseModel):
    confirmation_message: str
    calendar_link: Optional[str] = None


# --------------------------------------------------------------
# Helper function: Single Groq call with Pydantic validation
# --------------------------------------------------------------


def run_llm(prompt: str, schema: BaseModel):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = completion.choices[0].message.content

    try:
        return schema.model_validate_json(content)
    except ValidationError:
        logger.error("Schema validation failed.")
        logger.error(content)
        raise


# --------------------------------------------------------------
# First chain: Detect if event
# --------------------------------------------------------------


def extract_event_info(user_input: str) -> EventExtraction:
    today = datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""
Today is {today}.

Analyze this text and return JSON exactly in this format:

{{
  "description": "string",
  "is_calendar_event": true or false,
  "confidence_score": number between 0 and 1
}}

Text:
{user_input}
"""

    return run_llm(prompt, EventExtraction)


# --------------------------------------------------------------
# Second chain: Extract event details
# --------------------------------------------------------------


def parse_event_details(description: str) -> EventDetails:
    today = datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""
Today is {today}.

Extract event details in this JSON format:

{{
  "name": "string",
  "date": "ISO 8601 date-time",
  "duration_minutes": number,
  "participants": ["name1", "name2"]
}}

Text:
{description}
"""

    return run_llm(prompt, EventDetails)


# --------------------------------------------------------------
# Third chain: Generate confirmation message
# --------------------------------------------------------------


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    prompt = f"""
Generate confirmation message in this JSON format:

{{
  "confirmation_message": "string",
  "calendar_link": "string or null"
}}

Event:
{event_details.model_dump_json()}

Include name as "Susie".
"""

    return run_llm(prompt, EventConfirmation)


# --------------------------------------------------------------
# Main Prompt Chain Controller
# --------------------------------------------------------------


def process_calendar_request(user_input: str):
    logger.info("Step 1: Extracting event")

    extraction = extract_event_info(user_input)

    # Gate Check
    if not extraction.is_calendar_event or extraction.confidence_score < 0.7:
        logger.warning("Gate check failed")
        print("Not a calendar event.")
        return None   # EXIT IMMEDIATELY

    logger.info("Gate passed")

    # Step 2
    details = parse_event_details(extraction.description)

    # Step 3
    confirmation = generate_confirmation(details)

    return confirmation



# --------------------------------------------------------------
# Tests
# --------------------------------------------------------------

if __name__ == "__main__":
    user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the roadmap."
    result = process_calendar_request(user_input)

    if result:
        print("\nCONFIRMATION\n")
        print(result.confirmation_message)
    else:
        print("Not a calendar event.")

