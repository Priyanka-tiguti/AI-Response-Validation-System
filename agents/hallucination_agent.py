import json
from agents.base import call_llm


def hallucination_agent(
    question,
    ai_response,
    reference_context=None,
    reference_answer=None,
):

    # ============================================================
    # PREPARE EVIDENCE
    # ============================================================

    retrieved_evidence = (
        reference_context.strip()
        if reference_context and reference_context.strip()
        else "No RAG reference evidence was retrieved."
    )

    reference_answer_text = (
        reference_answer.strip()
        if reference_answer and reference_answer.strip()
        else "No reference answer was provided."
    )

    # ============================================================
    # PROMPT
    # ============================================================

    prompt = f"""
You are the Hallucination Detection Agent in an AI Response
Validation System.

Your ONLY responsibility is to identify factual claims in the
AI response that are contradicted, fabricated, or materially
unsupported.

==================================================
QUESTION
==================================================

{question}

==================================================
AI RESPONSE
==================================================

{ai_response}

==================================================
PRIMARY RAG EVIDENCE
==================================================

{retrieved_evidence}

==================================================
REFERENCE ANSWER
==================================================

{reference_answer_text}

==================================================
CORE RULE
==================================================

Evaluate ONLY the factual claims that are actually present
in the AI response.

Do NOT flag information that is simply missing.

Missing information is a Completeness issue.

==================================================
CRITICAL RULE FOR REFERENCE ANSWERS
==================================================

When a reference answer is provided:

If the AI response directly contradicts the reference answer,
the claim MUST be flagged as:

"contradicted"

Do NOT classify a directly contradicted claim as merely
"unsupported_world_knowledge".

Examples:

Reference:
The capital of Australia is Canberra.

AI Response:
The capital of Australia is Sydney.

Result:
Hallucination Detected.

Classification:
contradicted.

--------------------------------------------------

Reference:
RAM is volatile memory and does not retain data after power
is removed.

AI Response:
RAM is permanent storage that keeps data after the computer
is turned off.

Result:
Hallucination Detected.

Classification:
contradicted.

--------------------------------------------------

Reference:
Alexander Graham Bell was awarded a telephone patent in 1876.

AI Response:
Thomas Edison invented the telephone in 1876.

Result:
Hallucination Detected.

Classification:
contradicted.

==================================================
CLAIM CLASSIFICATION
==================================================

For every important factual claim, classify it as one of:

SUPPORTED
The claim agrees with the evidence.

CONTRADICTED
The claim directly conflicts with the reference or reliable
retrieved evidence.

FABRICATED
The claim is clearly invented, impossible, or demonstrably false.

UNSUPPORTED_WORLD_KNOWLEDGE
The claim cannot be verified using the available evidence,
but there is not enough evidence to say that it is false.

IMPORTANT:

Unsupported does NOT automatically mean hallucination.

However, a claim that directly contradicts the reference
MUST be classified as contradicted.

==================================================
DO NOT FLAG
==================================================

Do NOT flag a claim merely because:

- the wording is different
- it is a reasonable paraphrase
- the response is short
- the response is incomplete
- information is missing
- RAG does not contain every fact
- the response does not provide an explanation

==================================================
IMPORTANT EXAMPLE 1
==================================================

Question:
What are the three states of matter?

Reference:
The three common states of matter are solid, liquid, and gas.

Response:
Solid.

Result:
No Hallucination.

Reason:
"Solid" is a true claim. Liquid and gas are missing, which
is a Completeness issue.

==================================================
IMPORTANT EXAMPLE 2
==================================================

Question:
What is the capital of Australia?

Reference:
The capital of Australia is Canberra.

Response:
The capital of Australia is Sydney.

Result:
Hallucination Detected.

Flagged claim:
"The capital of Australia is Sydney."

Classification:
contradicted.

==================================================
IMPORTANT EXAMPLE 3
==================================================

Question:
What is RAM?

Reference:
RAM is volatile memory used to temporarily store data.

Response:
RAM is permanent storage that keeps data after the computer
is turned off.

Result:
Hallucination Detected.

Classification:
contradicted.

==================================================
IMPORTANT EXAMPLE 4
==================================================

Question:
What is an API?

Reference:
An API allows software components to communicate.

Response:
An API allows different software components to communicate.

Result:
No Hallucination.

The response is a reasonable paraphrase.

==================================================
IMPORTANT EXAMPLE 5
==================================================

Question:
What is photosynthesis?

Reference:
Photosynthesis is the process by which plants use sunlight,
carbon dioxide, and water to produce glucose and oxygen.

Response:
Plants use sunlight to produce food.

Result:
No Hallucination.

The response is incomplete but does not contain a false claim.

==================================================
MIXED CLAIMS
==================================================

If the response contains both correct and incorrect claims:

- Keep correct claims unflagged.
- Flag only the incorrect claims.
- Do not mark every claim as hallucinated.
- The final score must reflect the severity of the problematic
  claims.

Example:

Response:
"The Sun is a star and the Sun is the largest planet."

If evidence says the Sun is a star and not a planet:

Flag only:
"The Sun is the largest planet."

==================================================
SCORING
==================================================

5 =
No meaningful hallucination.

4 =
Only a minor questionable claim with little impact.

3 =
Moderate concern with weak or questionable claims.

2 =
At least one important factual claim is contradicted or fabricated.

1 =
Multiple important claims are contradicted or fabricated.

0 =
The main answer is predominantly false, fabricated, or contradicted.

IMPORTANT:

If the main factual answer directly contradicts the reference,
the score MUST NOT be 4 or 5.

A directly contradicted main answer should normally receive
0-2 depending on severity.

==================================================
RISK LEVEL
==================================================

5-4 = Low Risk

3 = Medium Risk

2-0 = High Risk

==================================================
STATUS
==================================================

Use:

"No Hallucination"

when there are no meaningful hallucinated claims.

Use:

"Hallucination Detected"

when at least one important claim is contradicted,
fabricated, or materially unsupported.

==================================================
FINAL DECISION
==================================================

Before returning the result:

1. Extract factual claims from the AI response.
2. Compare each claim against the available evidence.
3. Look specifically for direct contradictions.
4. Flag direct contradictions.
5. Ignore missing information.
6. Assign a score based on the problematic claims.
7. Set risk level according to the score.
8. Set status consistently with the result.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Use exactly:

{{
    "score": 5,
    "risk_level": "Low Risk",
    "status": "No Hallucination",
    "flagged_claims": [],
    "reasoning": "The response does not contain any meaningful hallucinated claims."
}}

For a contradicted claim:

{{
    "score": 1,
    "risk_level": "High Risk",
    "status": "Hallucination Detected",
    "flagged_claims": [
        {{
            "claim": "The capital of Australia is Sydney.",
            "classification": "contradicted",
            "evidence": "The reference states that Canberra is the capital of Australia.",
            "reason": "The response directly contradicts the provided reference answer."
        }}
    ],
    "reasoning": "The response contains a factual claim that directly contradicts the reference evidence."
}}

Allowed classifications ONLY:

"unsupported_world_knowledge"
"contradicted"
"fabricated"

Rules:

- score must be an integer from 0 to 5
- risk_level must match the score
- status must match the score
- flagged_claims must always be a list
- do not flag missing information
- do not flag incompleteness
- do not flag reasonable paraphrases
- direct contradictions MUST be flagged
- return only valid JSON
"""

    # ============================================================
    # CALL LLM
    # ============================================================

    result = call_llm(prompt)

    # ============================================================
    # CLEAN OUTPUT
    # ============================================================

    cleaned_result = (
        result
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # ============================================================
    # PARSE JSON
    # ============================================================

    try:
        data = json.loads(cleaned_result)

    except json.JSONDecodeError:

        return {
            "score": 3,
            "risk_level": "Medium Risk",
            "status": "Hallucination Detected",
            "flagged_claims": [],
            "reasoning": (
                "The hallucination detection model returned invalid JSON."
            ),
        }

    # ============================================================
    # VALIDATE OBJECT
    # ============================================================

    if not isinstance(data, dict):

        return {
            "score": 3,
            "risk_level": "Medium Risk",
            "status": "Hallucination Detected",
            "flagged_claims": [],
            "reasoning": (
                "The hallucination detection model returned an invalid object."
            ),
        }

    # ============================================================
    # SCORE
    # ============================================================

    try:
        score = int(data.get("score", 3))
    except (ValueError, TypeError):
        score = 3

    score = max(0, min(5, score))

    # ============================================================
    # FLAGGED CLAIMS
    # ============================================================

    flagged_claims = data.get("flagged_claims", [])

    if not isinstance(flagged_claims, list):
        flagged_claims = []

    allowed_classifications = {
        "unsupported_world_knowledge",
        "contradicted",
        "fabricated",
    }

    valid_claims = []

    for claim in flagged_claims:

        if not isinstance(claim, dict):
            continue

        classification = claim.get("classification")

        if classification not in allowed_classifications:
            continue

        valid_claims.append(
            {
                "claim": str(claim.get("claim", "")),
                "classification": classification,
                "evidence": str(claim.get("evidence", "")),
                "reason": str(claim.get("reason", "")),
            }
        )

    # ============================================================
    # IMPORTANT SAFETY CHECK
    # ============================================================

    # If the model found a contradiction/fabrication, do not allow
    # the final result to incorrectly remain Low Risk / No Hallucination.

    serious_claims = [
        claim
        for claim in valid_claims
        if claim["classification"] in {
            "contradicted",
            "fabricated",
        }
    ]

    if serious_claims:

        if score > 2:
            score = 2

    # ============================================================
    # FINAL SCORE
    # ============================================================

    if score >= 4:

        risk_level = "Low Risk"

        status = (
            "Hallucination Detected"
            if serious_claims
            else "No Hallucination"
        )

    elif score == 3:

        risk_level = "Medium Risk"

        status = (
            "Hallucination Detected"
            if valid_claims
            else "No Hallucination"
        )

    else:

        risk_level = "High Risk"
        status = "Hallucination Detected"

    # ============================================================
    # REASONING
    # ============================================================

    reasoning = data.get("reasoning")

    if not isinstance(reasoning, str) or not reasoning.strip():

        reasoning = "Hallucination evaluation completed."

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "score": score,
        "risk_level": risk_level,
        "status": status,
        "flagged_claims": valid_claims,
        "reasoning": reasoning,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    question = "What is RAM?"

    ai_response = (
        "RAM is permanent storage that keeps data after "
        "the computer is turned off."
    )

    reference_answer = (
        "RAM is volatile computer memory used to temporarily "
        "store data and programs being actively used."
    )

    result = hallucination_agent(
        question,
        ai_response,
        reference_answer=reference_answer,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )