import csv
import json
import time
from pathlib import Path

from orchestrator import evaluate_response


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "test_cases.csv"
OUTPUT_FILE = BASE_DIR / "batch_results.json"

# True  = Mock mode (no Gemini API calls)
# False = Real Gemini evaluation
USE_MOCK = True

# Delay between test cases (seconds)
DELAY = 2


# ==========================================
# MOCK EVALUATOR
# ==========================================

def mock_evaluate_response(
    question,
    ai_response,
    reference_context
):
    """
    Returns predefined results for testing.
    Does NOT call Gemini.
    """

    return {
        "accuracy": {
            "score": 5,
            "reasoning": "Mock accuracy evaluation"
        },
        "completeness": {
            "score": 5,
            "reasoning": "Mock completeness evaluation"
        },
        "relevance": {
            "score": 5,
            "reasoning": "Mock relevance evaluation"
        },
        "hallucination": {
            "score": 5,
            "risk level": "Low Risk",
            "status": "No Hallucination"
        },
        "verdict": {
            "overall_score": 5.0,
            "verdict": "Pass",
            "risk level": "Low Risk"
        }
    }


# ==========================================
# LOAD CSV DATASET
# ==========================================

def load_test_cases(csv_file):

    test_cases = []

    required_columns = [
        "question",
        "ai_response",
        "reference_context"
    ]

    with open(
        csv_file,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        # Validate CSV headers
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no headers.")

        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]

        missing_columns = [
            col for col in required_columns
            if col not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required CSV columns: {missing_columns}"
            )

        # Read each row
        for row_number, row in enumerate(reader, start=2):

            test_case = {
                key: (row.get(key) or "").strip()
                for key in required_columns
            }

            # Skip completely empty rows
            if not any(test_case.values()):
                continue

            # Validate required values
            missing_values = [
                key for key, value in test_case.items()
                if not value
            ]

            if missing_values:
                raise ValueError(
                    f"Row {row_number} has missing values: "
                    f"{missing_values}"
                )

            test_cases.append(test_case)

    if not test_cases:
        raise ValueError("No valid test cases found in CSV.")

    print(f"Loaded {len(test_cases)} test cases from CSV.")

    return test_cases


# ==========================================
# BATCH EVALUATION
# ==========================================

def run_batch_evaluation(
    test_cases,
    evaluator,
    delay=2
):

    results = []

    total_cases = len(test_cases)

    for i, test_case in enumerate(test_cases, start=1):

        print(
            f"\nEvaluating test case {i}/{total_cases}..."
        )

        try:

            evaluation = evaluator(
                test_case["question"],
                test_case["ai_response"],
                test_case["reference_context"]
            )

            results.append({
                "test_case": i,
                "question": test_case["question"],
                "ai_response": test_case["ai_response"],
                "reference_context": test_case["reference_context"],
                "evaluation": evaluation,
                "status": "Success"
            })

            print(f"Test case {i} completed successfully.")

        except Exception as e:

            results.append({
                "test_case": i,
                "question": test_case["question"],
                "ai_response": test_case["ai_response"],
                "reference_context": test_case["reference_context"],
                "evaluation": None,
                "status": "Failed",
                "error": str(e)
            })

            print(f"Test case {i} failed: {e}")

        # Pause between cases
        if i < total_cases and delay > 0:
            time.sleep(delay)

    return results


# ==========================================
# SAVE RESULTS TO JSON
# ==========================================

def save_results(results, output_file):

    with open(
        output_file,
        mode="w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(f"\nResults saved to: {output_file}")


# ==========================================
# PRINT BATCH SUMMARY
# ==========================================

def print_batch_summary(results):

    successful = sum(
        1 for result in results
        if result["status"] == "Success"
    )

    failed = len(results) - successful

    print("\n========== BATCH EVALUATION SUMMARY ==========")

    print(f"Total test cases : {len(results)}")
    print(f"Successful       : {successful}")
    print(f"Failed           : {failed}")

    for result in results:

        print(f"\nTest Case: {result['test_case']}")
        print(f"Question: {result['question']}")
        print(f"Status: {result['status']}")

        if result["evaluation"] is not None:
            print(
                "Evaluation:",
                json.dumps(
                    result["evaluation"],
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            )
        else:
            print("Error:", result.get("error", "Unknown error"))


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    try:

        # Step 1: Load CSV
        test_cases = load_test_cases(CSV_FILE)

        # Step 2: Select evaluator
        if USE_MOCK:
            selected_evaluator = mock_evaluate_response
            print("\nRunning in MOCK MODE (No Gemini API calls)")

        else:
            selected_evaluator = evaluate_response
            print("\nRunning in REAL GEMINI MODE")

        # Step 3: Run batch evaluation
        results = run_batch_evaluation(
            test_cases,
            evaluator=selected_evaluator,
            delay=DELAY
        )

        # Step 4: Print summary
        print_batch_summary(results)

        # Step 5: Save JSON results
        save_results(results, OUTPUT_FILE)

    except Exception as e:
        print(f"\nBatch evaluation could not start: {e}")