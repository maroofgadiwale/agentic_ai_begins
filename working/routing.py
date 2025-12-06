from typing import Optional, Literal, List
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
# Data Models
# --------------------------------------------------------------


class CalendarRequestType(BaseModel):
    request_type: Literal["new_event", "modify_event", "other"]
    confidence_score: float
    description: str


class NewEventDetails(BaseModel):
    name: str
    date: str
    duration_minutes: int
    participants: List[str]


class Change(BaseModel):
    field: str
    new_value: str


class ModifyEventDetails(BaseModel):
    event_identifier: str
    changes: List[Change]
    participants_to_add: List[str]
    participants_to_remove: List[str]


class CalendarResponse(BaseModel):
    success: bool
    message: str
    calendar_link: Optional[str]


# --------------------------------------------------------------
# Helper: Run Groq + Pydantic Validation
# --------------------------------------------------------------


def run_llm(prompt: str, schema):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON for the schema. No extra text."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = completion.choices[0].message.content

    try:
        return schema.model_validate_json(content)
    except ValidationError:
        logger.error("Schema validation failed")
        logger.error(content)
        raise


# --------------------------------------------------------------
# Router
# --------------------------------------------------------------


def route_calendar_request(user_input: str) -> CalendarRequestType:
    logger.info("Routing calendar request")

    prompt = f"""
Classify the request and return JSON:

{{
  "request_type": "new_event" | "modify_event" | "other",
  "confidence_score": number between 0 and 1,
  "description": "cleaned user intent"
}}

Text:
{user_input}
"""
    return run_llm(prompt, CalendarRequestType)


# --------------------------------------------------------------
# New Event Handler
# --------------------------------------------------------------


def handle_new_event(description: str) -> CalendarResponse:
    logger.info("Processing new event")

    prompt = f"""
Extract new event details in JSON:

{{
  "name": "string",
  "date": "ISO 8601 format",
  "duration_minutes": number,
  "participants": ["name1", "name2"]
}}

Text:
{description}
"""
    details = run_llm(prompt, NewEventDetails)

    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


# --------------------------------------------------------------
# Modify Event Handler
# --------------------------------------------------------------


def handle_modify_event(description: str) -> CalendarResponse:
    logger.info("Processing modification")

    prompt = f"""
Extract modification details:

{{
  "event_identifier": "string",
  "changes": [
      {{ "field": "string", "new_value": "string" }}
  ],
  "participants_to_add": ["name"],
  "participants_to_remove": ["name"]
}}

Text:
{description}
"""
    details = run_llm(prompt, ModifyEventDetails)

    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}' successfully",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


# --------------------------------------------------------------
# Controller (AI Agent)
# --------------------------------------------------------------


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    logger.info("Processing calendar request")

    route = route_calendar_request(user_input)

    # Gate check
    if route.confidence_score < 0.7:
        logger.warning("Low confidence - request rejected")
        return None

    if route.request_type == "new_event":
        return handle_new_event(route.description)

    elif route.request_type == "modify_event":
        return handle_modify_event(route.description)

    else:
        logger.warning("Non-calendar request")
        return None


# --------------------------------------------------------------
# Tests
# --------------------------------------------------------------

if __name__ == "__main__":

    print("\n--- NEW EVENT ---")
    result = process_calendar_request(
        "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
    )
    print(result.message if result else "Not recognized")

    print("\n--- MODIFY EVENT ---")
    result = process_calendar_request(
        "Move the team meeting with Alice and Bob to Wednesday at 3pm"
    )
    print(result.message if result else "Not recognized")

    print("\n--- INVALID ---")
    result = process_calendar_request("What's the weather today?")
    print(result.message if result else "Not recognized as calendar request")
