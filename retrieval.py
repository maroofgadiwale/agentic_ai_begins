# Knowledge Base:
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
import json
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API"))

# --------------------------------------------------------------
# Mock Knowledge Base
# --------------------------------------------------------------

def search_kb(question: str):
    with open("kb.json", "r") as f:
        return json.load(f)

# --------------------------------------------------------------
# Tool definition
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    }
]

# --------------------------------------------------------------
# System Prompt
# --------------------------------------------------------------

system_prompt = """You are a helpful assistant that answers questions from the knowledge base.

You must ALWAYS reply ONLY in JSON in this format:

{
  "answer": "string",
  "source": number
}

If the answer is not found, reply ONLY in this JSON format:

{
  "answer": "I'm sorry, I don't have that information.",
  "source": -1
}
"""


# --------------------------------------------------------------
# Messages
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "How the payment is accepted?"}
]

# --------------------------------------------------------------
# First Call
# --------------------------------------------------------------

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools,
    temperature=0
)

# --------------------------------------------------------------
# Tool Execution (Safe)
# --------------------------------------------------------------

if completion.choices[0].message.tool_calls:

    messages.append(completion.choices[0].message)

    for tool_call in completion.choices[0].message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = search_kb(**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # --------------------------------------------------------------
    # Final AI Call (Strict JSON)
    # --------------------------------------------------------------

    completion2 = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )

    content = completion2.choices[0].message.content

    # --------------------------------------------------------------
    # Pydantic Validation
    # --------------------------------------------------------------

    class KBResponse(BaseModel):
        answer: str
        source: int

    try:
        parsed = KBResponse.model_validate_json(content)
        print("\nANSWER:", parsed.answer)
        print("SOURCE ID:", parsed.source)

    except ValidationError as e:
        print("\nINVALID RESPONSE FORMAT")
        print(content)
        print(e)

else:
    print("Tool was not triggered by the model")
