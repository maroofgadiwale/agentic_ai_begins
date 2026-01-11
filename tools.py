import os
import json
import sys
import requests
from groq import Groq
from dotenv import load_dotenv

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
    print(response.json())
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
                        "description": "Any random city name to get weather information for"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# -----------------------------
# Initial Prompt (JSON included)
# -----------------------------
messages = [
    {
        "role": "system",
        "content": "You are a helpful weather assistant. Provide weather information in natural human conversation (make it a simple 2 line sentence with no escape sequences) in json format and key should be message."
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
    # Final Response (No tools, JSON enforced)
    # -----------------------------
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tool_choice="none",
        response_format={"type": "json_object"}
    )

    content = final_response.choices[0].message.content

    print("\nFINAL WEATHER RESPONSE:\n")
    print(content)

else:
    print("No tool was triggered by the model.")


