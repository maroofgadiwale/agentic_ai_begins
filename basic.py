import sys
from groq import Groq
from dotenv import load_dotenv  

# windows encoding:
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()  # take environment variables from .env file
#LLM Code:
client = Groq(api_key=os.getenv("GROQ_API"))

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "user",
        "content": "who is current weather in sangli?"
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
