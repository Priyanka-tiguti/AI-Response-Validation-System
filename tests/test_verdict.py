import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from agents.verdict_agent import verdict_agent


def run_test(scores, risk_level, expected_verdict):
    accuracy = {"score": scores[0], "reasoning": "Accuracy tested"}
    completeness = {"score": scores[1], "reasoning": "Completeness tested"}
    relevance = {"score": scores[2], "reasoning": "Relevance tested"}
    hallucination = {
        "score": scores[3],
        "risk_level": risk_level,
        "reasoning": "Hallucination tested"
    }

    result = verdict_agent(
        question="What is Python?",
        ai_response="Python is a programming language.",
        accuracy_result=accuracy,
        completeness_result=completeness,
        relevance_result=relevance,
        hallucination_result=hallucination
    )

    print(f"\nRisk Level: {risk_level}")
    print(f"Overall Score: {result['overall_score']}")
    print(f"Expected Verdict: {expected_verdict}")
    print(f"Actual Verdict: {result['verdict']}")

    assert result["verdict"] == expected_verdict, "Test Failed!"

    print("Test Passed!")


# Test 1: Low Risk + High Score
run_test([5, 5, 5, 5], "Low Risk", "Pass")

# Test 2: Medium Risk + High Score
run_test([5, 5, 5, 5], "Medium Risk", "Needs Improvement")

# Test 3: High Risk + High Score
run_test([5, 5, 5, 5], "High Risk", "Fail")

# Test 4: Low Risk + Low Score
run_test([2, 2, 2, 2], "Low Risk", "Fail")

print("\nAll verdict tests completed successfully!")