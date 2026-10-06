def verdict_agent(
    question,
    ai_response,
    accuracy_result,
    completeness_result,
    relevance_result,
    hallucination_result
):
    # ============================================================
    # EXTRACT SCORES
    # ============================================================

    accuracy_score = float(accuracy_result["score"])
    completeness_score = float(completeness_result["score"])
    relevance_score = float(relevance_result["score"])
    hallucination_score = float(hallucination_result["score"])

    risk_level = hallucination_result.get(
        "risk_level",
        "Low Risk"
    )

    valid_risk_levels = [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]

    if risk_level not in valid_risk_levels:
        raise ValueError(
            f"Invalid risk level: {risk_level}"
        )

    # ============================================================
    # VALIDATE SCORES
    # ============================================================

    scores = [
        accuracy_score,
        completeness_score,
        relevance_score,
        hallucination_score
    ]

    if any(score < 0 or score > 5 for score in scores):
        raise ValueError(
            "All agent scores must be between 0 and 5."
        )

    # ============================================================
    # WEIGHTED SCORE
    # ============================================================

    weighted_score = (
        (accuracy_score * 0.30) +
        (completeness_score * 0.15) +
        (relevance_score * 0.15) +
        (hallucination_score * 0.40)
    )

    overall_score = round(weighted_score, 2)

    # ============================================================
    # FINAL VERDICT
    # ============================================================

    # Critical factual or hallucination problems
    # should not be hidden by the weighted average.

    if risk_level == "High Risk":

        verdict_category = "Fail"

        verdict_reason = (
            "High hallucination risk was detected."
        )

    elif accuracy_score <= 1:

        verdict_category = "Fail"

        verdict_reason = (
            "The accuracy score is critically low, "
            "indicating major factual errors."
        )

    elif hallucination_score <= 2:

        verdict_category = "Fail"

        verdict_reason = (
            "The hallucination score is critically low, "
            "indicating significant unsupported or "
            "incorrect claims."
        )

    elif risk_level == "Medium Risk":

        verdict_category = "Needs Improvement"

        verdict_reason = (
            "Medium hallucination risk was detected "
            "and the response requires review."
        )

    elif overall_score >= 4.0:

        verdict_category = "Pass"

        verdict_reason = (
            "The response meets the passing score threshold "
            "with no critical accuracy or hallucination issue."
        )

    elif overall_score >= 2.5:

        verdict_category = "Needs Improvement"

        verdict_reason = (
            "The response needs improvement to meet "
            "the passing threshold."
        )

    else:

        verdict_category = "Fail"

        verdict_reason = (
            "The response scored below the minimum "
            "acceptable threshold."
        )

    # ============================================================
    # AGENT REASONING
    # ============================================================

    accuracy_reasoning = accuracy_result.get(
        "reasoning",
        ""
    )

    completeness_reasoning = completeness_result.get(
        "reasoning",
        ""
    )

    relevance_reasoning = relevance_result.get(
        "reasoning",
        ""
    )

    hallucination_reasoning = hallucination_result.get(
        "reasoning",
        ""
    )

    # ============================================================
    # CONSOLIDATED REASONING
    # ============================================================

    consolidated_reasoning = (
        f"Accuracy: {accuracy_reasoning}\n"
        f"Completeness: {completeness_reasoning}\n"
        f"Relevance: {relevance_reasoning}\n"
        f"Hallucination: {hallucination_reasoning}\n"
        f"Overall risk level: {risk_level}.\n"
        f"Final weighted score: {overall_score:.2f}/5.\n"
        f"Final Verdict: {verdict_category}\n"
        f"Verdict Explanation: {verdict_reason}"
    )

    # ============================================================
    # RETURN STRUCTURED VERDICT
    # ============================================================

    return {
        "overall_score": overall_score,
        "verdict": verdict_category,
        "verdict_reason": verdict_reason,
        "risk_level": risk_level,
        "reasoning": consolidated_reasoning,
        "score_breakdown": {
            "accuracy": accuracy_score,
            "completeness": completeness_score,
            "relevance": relevance_score,
            "hallucination": hallucination_score
        }
    }