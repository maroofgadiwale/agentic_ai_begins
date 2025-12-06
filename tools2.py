import os
import json
import sys
import requests
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# -----------------------------
# Force UTF-8 Output (Fix Windows Emoji Errors)
# -----------------------------
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Initialize Groq Client
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API"))

# -----------------------------
# WeatherResponse Schema (STRUCTURED OUTPUT)
# -----------------------------
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

# -----------------------------
# Weather Tool Function
# -----------------------------
def get_weather(location):
    api_key = os.getenv("WEATHER_API")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

# -----------------------------
# Tool Schema
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information using city name",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name to get weather information for"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# -----------------------------
# Initial Prompt (Schema Enforced)
# -----------------------------
messages = [
    {
        "role": "system",
        "content": """You are a strict weather assistant.
You must return output ONLY in this JSON format:

{
  "temperature": number,
  "response": "natural language weather description"
}

Do NOT return markdown.
Do NOT return extra fields.
Do NOT wrap in code blocks.
"""
    },
    {
        "role": "user",
        "content": "What's the current weather in Delhi?"
    }
]

# -----------------------------
# First Groq Call (Tool Enabled)
# -----------------------------
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools,
    temperature=0
)

# -----------------------------
# Execute tool if requested
# -----------------------------
if completion.choices[0].message.tool_calls:

    for tool_call in completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"\nCalling tool: {name}, arguments: {args}")

        result = get_weather(**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": name,
            "content": json.dumps(result)
        })

    # -----------------------------
    # Final Response (Schema Enforced JSON)
    # -----------------------------
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )

    content = final_response.choices[0].message.content

    # -----------------------------
    # Validate Output using WeatherResponse Schema
    # -----------------------------
    try:
        weather = WeatherResponse.model_validate_json(content)


        print("\nFINAL VERIFIED WEATHER RESPONSE\n")
        print(f"Temperature: {weather.temperature} °C")
        print(f"Message: {weather.response}")

    except ValidationError as e:
        print("\nAI RESPONSE INVALID (Schema validation failed)")
        print("Raw output:")
        print(content)
        print("\nValidation Error:")
        print(e)

else:
    print("No tool was triggered by the model.")
