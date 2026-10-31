
from dotenv import load_dotenv
from pydantic import BaseModel
from src.rag_evaluation.groq_judge import GroqJudge

load_dotenv(override=True)


class JudgeTestResponse(BaseModel):
    verdict: str
    reason: str


def main():

    judge = GroqJudge()

    print("Model:", judge.get_model_name())
    print("Testing Groq structured output...")

    prompt = """
    Evaluate this simple statement.

    Statement: The Earth revolves around the Sun.

    Return verdict as PASS and give a short reason.
    """

    result = judge.generate(
        prompt=prompt,
        schema=JudgeTestResponse
    )

    print("\nSUCCESS: Groq API is working.")
    print("Verdict:", result.verdict)
    print("Reason:", result.reason)


if __name__ == "__main__":
    main()
