
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from rag_evaluation.retriever import build_retriever
from rag_evaluation.groq_judge import GroqJudge


# ==================================================
# 1. ENVIRONMENT AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=True)

GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retriever_goldens.json"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "contextual_metrics_results.json"

THRESHOLD = 0.7

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
TOP_K = 5

# Small pause between metric requests.
# This controls request frequency, not daily token consumption.
WAIT_BETWEEN_METRICS = 30


# ==================================================
# 2. JSON REPORT FUNCTIONS
# ==================================================

def save_json_report(report):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def create_metric(metric_name, judge):

    if metric_name == "Contextual Recall":
        return ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        )

    if metric_name == "Contextual Precision":
        return ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        )

    raise ValueError(f"Unknown metric: {metric_name}")


# ==================================================
# 3. EVALUATE ONE METRIC AT A TIME
# ==================================================

def evaluate_single_metric(metric_name, test_case, judge):

    metric = create_metric(metric_name, judge)

    try:

        metric.measure(test_case)

        score = metric.score
        reason = metric.reason

        success = metric.is_successful()

        return {
            "metric": metric_name,
            "score": float(score) if score is not None else None,
            "threshold": THRESHOLD,
            "status": "PASS" if success else "FAIL",
            "reason": reason,
            "error": None,
        }

    except Exception as error:

        return {
            "metric": metric_name,
            "score": None,
            "threshold": THRESHOLD,
            "status": "API_ERROR",
            "reason": None,
            "error": f"{type(error).__name__}: {error}",
        }


# ==================================================
# 4. AGGREGATE METRIC RESULTS
# ==================================================

def build_metric_summary(cases):

    summary = {}

    metric_names = [
        "Contextual Recall",
        "Contextual Precision",
    ]

    for metric_name in metric_names:

        scores = []
        passed = 0
        failed = 0
        api_errors = 0

        for case in cases:

            for metric in case["metrics"]:

                if metric["metric"] != metric_name:
                    continue

                score = metric["score"]

                if isinstance(score, (int, float)):
                    scores.append(score)

                if metric["status"] == "PASS":
                    passed += 1

                elif metric["status"] == "FAIL":
                    failed += 1

                elif metric["status"] == "API_ERROR":
                    api_errors += 1

        evaluated = passed + failed

        summary[metric_name] = {
            "average_score": (
                round(sum(scores) / len(scores), 4)
                if scores else None
            ),
            "pass_rate_percent": (
                round(passed / evaluated * 100, 2)
                if evaluated else None
            ),
            "passed": passed,
            "failed": failed,
            "api_errors": api_errors,
            "total_with_scores": len(scores),
        }

    return summary


# ==================================================
# 5. OVERALL SUMMARY
# ==================================================

def build_overall_summary(cases):

    passed = 0
    failed = 0
    api_errors = 0

    for case in cases:

        statuses = [
            metric["status"]
            for metric in case["metrics"]
        ]

        if "API_ERROR" in statuses:
            api_errors += 1

        elif len(statuses) == 2 and all(
            status == "PASS" for status in statuses
        ):
            passed += 1

        elif len(statuses) == 2:
            failed += 1

    total = len(cases)

    return {
        "passed": passed,
        "failed": failed,
        "api_errors": api_errors,
        "total_test_cases": total,
        "pass_rate_percent": (
            round(passed / total * 100, 2)
            if total else None
        ),
    }


# ==================================================
# 6. PRINT FINAL RESULTS TABLE
# ==================================================

def print_summary(report):

    print("\n")
    print("=" * 75)
    print("                 RAG RETRIEVER EVALUATION SUMMARY")
    print("=" * 75)

    print(
        f"{'Metric':<25}"
        f"{'Average Score':<18}"
        f"{'Pass Rate':<15}"
        f"{'Passed':<10}"
        f"{'Failed':<10}"
    )

    print("-" * 75)

    for metric_name, result in report["metric_summary"].items():

        average = result["average_score"]
        pass_rate = result["pass_rate_percent"]

        average_text = (
            f"{average:.4f}" if average is not None else "N/A"
        )

        pass_rate_text = (
            f"{pass_rate:.2f}%"
            if pass_rate is not None else "N/A"
        )

        print(
            f"{metric_name:<25}"
            f"{average_text:<18}"
            f"{pass_rate_text:<15}"
            f"{result['passed']:<10}"
            f"{result['failed']:<10}"
        )

    print("-" * 75)

    overall = report["overall_summary"]

    print("\nOVERALL RESULTS")
    print(f"Passed test cases : {overall['passed']}")
    print(f"Failed test cases : {overall['failed']}")
    print(f"API error cases   : {overall['api_errors']}")
    print(
        f"Overall pass rate : "
        f"{overall['pass_rate_percent']}%"
        if overall["pass_rate_percent"] is not None
        else "Overall pass rate : N/A"
    )

    print("=" * 75)


# ==================================================
# 7. LOAD AND VALIDATE GOLDEN DATASET
# ==================================================

if not GOLDEN_PATH.exists():
    raise FileNotFoundError(
        f"Golden dataset not found: {GOLDEN_PATH}"
    )

with open(GOLDEN_PATH, "r", encoding="utf-8") as file:
    goldens = json.load(file)

if not isinstance(goldens, list) or not goldens:
    raise ValueError(
        "Golden dataset must be a non-empty JSON list."
    )

for index, golden in enumerate(goldens, start=1):

    if "query" not in golden or "ideal_answer" not in golden:
        raise ValueError(
            f"Invalid golden test case at position {index}."
        )


# ==================================================
# 8. INITIALIZE REPORT
# ==================================================

report = {
    "project": "RAG Retrieval Evaluation",
    "evaluation_date": datetime.now(timezone.utc).isoformat(),

    "configuration": {
        "judge_model": None,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "execution_mode": "Sequential",
        "golden_dataset": str(GOLDEN_PATH),
    },

    "total_test_cases": len(goldens),
    "evaluation_status": "INITIALIZING",
    "api_error": None,
    "test_cases": [],
    "metric_summary": {},
    "overall_summary": {},
}

save_json_report(report)


# ==================================================
# 9. INITIALIZE JUDGE AND RETRIEVER
# ==================================================

print("\nInitializing Groq judge and retriever...")

try:

    judge = GroqJudge()

    report["configuration"]["judge_model"] = (
        judge.get_model_name()
    )

    retriever = build_retriever()

    print("Judge model:", judge.get_model_name())
    print("Retriever initialized successfully.")

except Exception as error:

    report["evaluation_status"] = "INITIALIZATION_ERROR"
    report["api_error"] = (
        f"{type(error).__name__}: {error}"
    )

    save_json_report(report)

    raise


# ==================================================
# 10. SEQUENTIAL RETRIEVAL AND EVALUATION
# ==================================================

metric_names = [
    "Contextual Recall",
    "Contextual Precision",
]

report["evaluation_status"] = "RUNNING"
save_json_report(report)

print("\n" + "=" * 75)
print("STARTING SEQUENTIAL DEEPEVAL METRIC EVALUATION")
print("=" * 75)

try:

    for index, golden in enumerate(goldens, start=1):

        query = golden["query"]
        ideal_answer = golden["ideal_answer"]

        print(
            f"\n\nTEST CASE {index}/{len(goldens)}"
        )

        print("Query:", query)

        # ------------------------------------------
        # Retrieve context
        # ------------------------------------------

        print("Retrieving context...")

        retrieved_docs = retriever.invoke(query)

        retrieval_context = [
            doc.page_content
            for doc in retrieved_docs
        ]

        test_case = LLMTestCase(
            input=query,
            expected_output=ideal_answer,
            retrieval_context=retrieval_context,
            actual_output="Generator not evaluated in this run.",
        )

        # Create and save the case before evaluation.
        case_result = {
            "case_number": index,
            "input": query,
            "expected_output": ideal_answer,
            "retrieval_context": retrieval_context,
            "success": None,
            "metrics": [],
        }

        report["test_cases"].append(case_result)

        save_json_report(report)

        print(
            f"Retrieved {len(retrieval_context)} context chunks."
        )

        # ------------------------------------------
        # Evaluate each metric sequentially
        # ------------------------------------------

        for metric_name in metric_names:

            print(f"\nEvaluating {metric_name}...")

            result = evaluate_single_metric(
                metric_name,
                test_case,
                judge,
            )

            case_result["metrics"].append(result)

            print("Score :", result["score"])
            print("Status:", result["status"])

            if result["reason"]:
                print("Reason:", result["reason"])

            if result["error"]:
                print("Error:", result["error"])

            # Save immediately after each metric.
            report["metric_summary"] = build_metric_summary(
                report["test_cases"]
            )

            report["overall_summary"] = build_overall_summary(
                report["test_cases"]
            )

            save_json_report(report)

            # Pause before the next API metric request.
            time.sleep(WAIT_BETWEEN_METRICS)

        # ------------------------------------------
        # Determine test case result
        # ------------------------------------------

        statuses = [
            metric["status"]
            for metric in case_result["metrics"]
        ]

        if "API_ERROR" in statuses:
            case_result["success"] = None

        else:
            case_result["success"] = all(
                status == "PASS" for status in statuses
            )

        report["metric_summary"] = build_metric_summary(
            report["test_cases"]
        )

        report["overall_summary"] = build_overall_summary(
            report["test_cases"]
        )

        save_json_report(report)

        print("\nTest case completed.")

        # ------------------------------------------
        # Stop after an API error
        # ------------------------------------------

        if "API_ERROR" in statuses:

            report["evaluation_status"] = (
                "INTERRUPTED_API_ERROR"
            )

            report["api_error"] = (
                "Evaluation stopped after an API error. "
                "Completed metric results have been saved."
            )

            save_json_report(report)

            print(
                "\nStopping evaluation because of an API error."
            )

            break

    else:

        report["evaluation_status"] = "COMPLETED"

except KeyboardInterrupt:

    report["evaluation_status"] = "INTERRUPTED_BY_USER"
    report["api_error"] = "Evaluation manually interrupted."

except Exception as error:

    report["evaluation_status"] = "ERROR"

    report["api_error"] = (
        f"{type(error).__name__}: {error}"
    )

finally:

    report["metric_summary"] = build_metric_summary(
        report["test_cases"]
    )

    report["overall_summary"] = build_overall_summary(
        report["test_cases"]
    )

    report["completed_test_cases"] = len(
        report["test_cases"]
    )

    report["report_updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    save_json_report(report)

    print("\nJSON report saved at:")
    print(REPORT_PATH)

    print_summary(report)


# ==================================================
# 11. FINAL STATUS
# ==================================================

print("\nEvaluation status:", report["evaluation_status"])

if report["evaluation_status"] == "COMPLETED":

    print("All test cases evaluated successfully.")

else:

    print(
        "Evaluation did not finish completely. "
        "Check the JSON report for saved results and errors."
    )
