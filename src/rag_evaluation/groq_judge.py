
import os
import json
import re
import copy

from groq import Groq, AsyncGroq
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Check your .env file."
            )

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.8-27b"
        )

        self.client = Groq(
            api_key=self.api_key,
            max_retries=2,
            timeout=120.0
        )

        self.async_client = AsyncGroq(
            api_key=self.api_key,
            max_retries=2,
            timeout=120.0
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    # ==================================================
    # PREPARE DEEPEVAL PYDANTIC SCHEMA FOR GROQ
    # ==================================================

    def _prepare_schema(self, schema):

        original_schema = schema.model_json_schema()
        definitions = original_schema.get("$defs", {})

        def expand_references(node):

            if isinstance(node, dict):

                if "$ref" in node:

                    reference = node["$ref"]

                    if not reference.startswith("#/$defs/"):
                        raise ValueError(
                            f"Unsupported schema reference: {reference}"
                        )

                    definition_name = reference.split("/")[-1]

                    if definition_name not in definitions:
                        raise ValueError(
                            f"Missing schema definition: {definition_name}"
                        )

                    expanded = copy.deepcopy(
                        definitions[definition_name]
                    )

                    for key, value in node.items():
                        if key != "$ref":
                            expanded[key] = value

                    return expand_references(expanded)

                cleaned = {}

                for key, value in node.items():

                    if key in ("$defs", "title"):
                        continue

                    cleaned[key] = expand_references(value)

                if cleaned.get("type") == "object":

                    properties = cleaned.get("properties", {})

                    cleaned["required"] = list(properties.keys())
                    cleaned["additionalProperties"] = False

                return cleaned

            if isinstance(node, list):
                return [
                    expand_references(item)
                    for item in node
                ]

            return node

        return expand_references(original_schema)

    # ==================================================
    # BUILD GROQ STRUCTURED OUTPUT FORMAT
    # ==================================================

    def _response_format(self, schema):

        json_schema = self._prepare_schema(schema)

        schema_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            schema.__name__
        )[:64]

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema
            }
        }

    # ==================================================
    # PREPARE MESSAGES
    # ==================================================

    def _prepare_messages(self, prompt):

        return [
            {
                "role": "system",
                "content": (
                    "You are an expert LLM evaluation judge. "
                    "Evaluate the supplied content objectively. "
                    "Return a valid JSON response that follows "
                    "the provided JSON schema exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    # ==================================================
    # PARSE GROQ RESPONSE
    # ==================================================

    def _parse_response(self, content, schema=None):

        if not content:
            raise ValueError(
                "Groq returned an empty response."
            )

        if schema is None:
            return content

        try:

            data = json.loads(content)

            return schema.model_validate(data)

        except (json.JSONDecodeError, ValueError) as error:

            raise ValueError(
                f"Groq returned invalid structured output: {content}"
            ) from error

    # ==================================================
    # SYNCHRONOUS GENERATION
    # ==================================================

    def generate(self, prompt: str, schema: BaseModel = None):

        kwargs = {
            "model": self.model_name,
            "messages": self._prepare_messages(prompt),
            "temperature": 0,
            "reasoning_effort": "low",
            "max_completion_tokens": 512
        }

        if schema is not None:
            kwargs["response_format"] = self._response_format(schema)

        response = self.client.chat.completions.create(
            **kwargs
        )

        content = response.choices[0].message.content

        return self._parse_response(content, schema)

    # ==================================================
    # ASYNCHRONOUS GENERATION
    # ==================================================

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel = None
    ):

        kwargs = {
            "model": self.model_name,
            "messages": self._prepare_messages(prompt),
            "temperature": 0,
            "reasoning_effort": "none",
            "max_completion_tokens": 4096
        }

        if schema is not None:
            kwargs["response_format"] = self._response_format(schema)

        response = await self.async_client.chat.completions.create(
            **kwargs
        )

        content = response.choices[0].message.content

        return self._parse_response(content, schema)
