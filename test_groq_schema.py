
import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

schema = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["yes", "no"]
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["verdict", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False
}

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert retrieval evaluation judge. "
                "Return verdicts as valid JSON."
            )
        },
        {
            "role": "user",
            "content": (
                "Question: What is online evaluation?\n"
                "Retrieved context: Online evaluation tests an LLM "
                "application using real user interactions.\n"
                "Determine whether the context supports the question."
            )
        }
    ],
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "verdicts",
            "strict": True,
            "schema": schema
        }
    }
)

content = response.choices[0].message.content

print("Groq response:")
print(content)

print("\nParsed JSON:")
print(json.loads(content))
