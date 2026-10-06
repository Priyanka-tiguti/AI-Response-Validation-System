
from agents.accuracy_agent import accuracy_agent
from agents.completeness_agent import completeness_agent
from agents.relevance_agent import relevance_agent
from agents.hallucination_agent import hallucination_agent
from agents.verdict_agent import verdict_agent


def evaluate_response(
    question,
    ai_response,
    reference_context,
    reference_answer=None
):
    # Accuracy Agent
    accuracy_result = accuracy_agent(
        question,
        ai_response,
        reference_context,
        reference_answer
    )

    # Completeness Agent
    completeness_result = completeness_agent(
        question,
        ai_response,
        reference_context
    )

    # Relevance Agent
    relevance_result = relevance_agent(
        question,
        ai_response
    )

    # Hallucination Detection Agent
    hallucination_result = hallucination_agent(
        question,
        ai_response,
        reference_context,
        reference_answer
    )

    # Final Verdict Agent
    verdict_result = verdict_agent(
        question,
        ai_response,
        accuracy_result,
        completeness_result,
        relevance_result,
        hallucination_result
    )

    return {
        "accuracy": accuracy_result,
        "completeness": completeness_result,
        "relevance": relevance_result,
        "hallucination": hallucination_result,
        "verdict": verdict_result
    }


if __name__ == "__main__":
    question = "What is the capital of France?"

    ai_response = "The capital of France is Paris."

    reference_answer = "Paris is the capital of France."

    reference_context = """
    Paris is the capital and largest city of France.
    """

    result = evaluate_response(
        question,
        ai_response,
        reference_context,
        reference_answer
    )

    print(result)


