from groq import Groq
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API"))


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "Extract event info in JSON only."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        {
            "role": "system",
            "content": """
Return JSON in this structure:
{
  "name": "...",
  "date": "...",
  "participants": ["..."]
}
Do not add explanations.
"""
        }
    ],
    temperature=0,
)

# ✅ Extract JSON text
raw_json = completion.choices[0].message.content

# ✅ Convert to dict
data = json.loads(raw_json)

# ✅ Validate using Pydantic
event = CalendarEvent(**data)

print(f"Event Name: {event.name}\nDate: {event.date}\nParticipants: {', '.join(event.participants)}")
