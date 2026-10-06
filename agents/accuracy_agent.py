import json
from agents.base import call_llm


def accuracy_agent(
    question,
    ai_response,
    reference_context=None,
    reference_answer=None,
):

    # --------------------------------------------------
    # SELECT PRIMARY EVIDENCE
    # --------------------------------------------------

    if reference_answer and reference_answer.strip():

        evaluation_source = f"""
PRIMARY GROUND-TRUTH REFERENCE ANSWER:
{reference_answer}

IMPORTANT:
When the AI response makes a claim that directly contradicts
this reference answer, the claim MUST be classified as incorrect.
"""

    elif reference_context and reference_context.strip():

        evaluation_source = f"""
PRIMARY REFERENCE CONTEXT:
{reference_context}

Use this context as the primary evidence for factual evaluation.
"""

    else:

        evaluation_source = """
No reference answer or reference context is available.

Use reliable general knowledge to evaluate factual claims.
Do not invent evidence.
"""

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = f"""
You are the Accuracy Judge Agent in an AI Response Validation System.

Your ONLY task is to evaluate the factual accuracy of the claims
actually stated in the AI response.

==================================================
QUESTION
==================================================

{question}

==================================================
AI RESPONSE
==================================================

{ai_response}

==================================================
REFERENCE EVIDENCE
==================================================

{evaluation_source}

==================================================
MOST IMPORTANT RULE
==================================================

Evaluate the claims that are ACTUALLY PRESENT in the AI response.

Do NOT evaluate missing information as an accuracy error.

However:

IF THE REFERENCE ANSWER DIRECTLY CONTRADICTS A CLAIM IN THE
AI RESPONSE, THAT CLAIM IS FACTUALLY INCORRECT.

Do NOT label a directly contradicted claim as "unverified."

==================================================
CLAIM EVALUATION
==================================================

For every important factual claim in the AI response:

1. Extract the claim.
2. Compare it with the reference evidence.
3. Determine whether it is:
   - CORRECT
   - INCORRECT
   - UNVERIFIED

Use these rules:

CORRECT:
The claim agrees with the reference evidence.

INCORRECT:
The claim directly contradicts the reference evidence.

UNVERIFIED:
The reference evidence does not contain enough information
to determine whether the claim is true or false.

IMPORTANT:
If the reference explicitly states one fact and the response
states the opposite fact, use INCORRECT.

==================================================
EXAMPLES
==================================================

Example 1:

Question:
What is the capital of Australia?

Response:
The capital of Australia is Sydney.

Reference:
The capital of Australia is Canberra.

Result:
INCORRECT

Accuracy score:
0

--------------------------------------------------

Example 2:

Question:
What is RAM?

Response:
RAM is permanent storage that keeps data after the computer
is turned off.

Reference:
RAM is volatile memory used to temporarily store data and
does not retain data after power is removed.

Result:
INCORRECT

Accuracy score:
0

IMPORTANT:
"Permanent storage" directly contradicts "volatile memory."

Do NOT give Accuracy 5.

--------------------------------------------------

Example 3:

Question:
What is an API?

Response:
An API allows different software components to communicate.

Reference:
An API is an interface that allows software components to
communicate and exchange data or functionality.

Result:
CORRECT

Accuracy score:
5

--------------------------------------------------

Example 4:

Question:
What are the three states of matter?

Response:
Matter exists as solid.

Reference:
The three common states are solid, liquid, and gas.

Result:
CORRECT for the claim actually stated.

Accuracy score:
5

The missing liquid and gas are a Completeness issue,
not an Accuracy issue.

--------------------------------------------------

Example 5:

Question:
Who invented the telephone?

Response:
Thomas Edison invented the telephone in 1876.

Reference:
Alexander Graham Bell was awarded a telephone patent in 1876.

Result:
INCORRECT

Accuracy should be substantially reduced.

==================================================
SPECIAL RULE FOR REFERENCE ANSWERS
==================================================

When a reference answer is supplied, treat it as the PRIMARY
GROUND TRUTH for the evaluation.

If the AI response says:

"The capital of Australia is Sydney."

and the reference says:

"The capital of Australia is Canberra."

then the response is INCORRECT.

If the AI response says:

"RAM is permanent storage."

and the reference says:

"RAM is volatile memory."

then the response is INCORRECT.

Do not allow an incorrect response to receive a high accuracy
score merely because the response is fluent, relevant, or
well-written.

==================================================
ACCURACY SCORING
==================================================

5 =
All important factual claims are correct.

4 =
Almost all factual claims are correct with only a minor
factual imprecision.

3 =
The response contains both correct and incorrect claims,
with meaningful errors but some correct information.

2 =
Several important claims are incorrect.

1 =
Most important claims are incorrect.

0 =
The main factual claim is directly contradicted by the
reference evidence or is clearly false.

==================================================
IMPORTANT DISTINCTIONS
==================================================

Accuracy is NOT Completeness.

Do not lower Accuracy merely because information is missing.

Accuracy is NOT Relevance.

Do not evaluate whether the response answers the question
well from a relevance perspective.

Accuracy is NOT Hallucination Risk.

Only evaluate factual correctness.

==================================================
FINAL DECISION PROCESS
==================================================

Before assigning the score:

1. Extract the factual claims actually stated.
2. Compare them with the reference.
3. Identify direct contradictions.
4. Mark each claim CORRECT, INCORRECT, or UNVERIFIED.
5. Determine the importance of each incorrect claim.
6. Assign the final 0-5 Accuracy score.

If the main answer is directly wrong, do NOT give a score of 4 or 5.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "score": 0,
    "reasoning": "The response directly contradicts the reference evidence.",
    "supporting_evidence": [
        "The reference states the correct fact."
    ]
}}

Rules:

- score must be an integer from 0 to 5.
- reasoning must discuss factual accuracy only.
- supporting_evidence must be a JSON list.
- Do not discuss completeness.
- Do not discuss relevance.
- Do not discuss hallucination risk.
- Do not use Markdown.
- Do not use code fences.
- Return only valid JSON.
"""

    # --------------------------------------------------
    # CALL LLM
    # --------------------------------------------------

    result = call_llm(prompt)

    # --------------------------------------------------
    # CLEAN OUTPUT
    # --------------------------------------------------

    cleaned_result = (
        result
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # --------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------

    try:

        data = json.loads(cleaned_result)

    except json.JSONDecodeError:

        return {
            "score": 3,
            "reasoning": (
                "The accuracy evaluation returned invalid JSON."
            ),
            "supporting_evidence": [],
        }

    # --------------------------------------------------
    # VALIDATE OBJECT
    # --------------------------------------------------

    if not isinstance(data, dict):

        return {
            "score": 3,
            "reasoning": (
                "The accuracy evaluation returned an invalid object."
            ),
            "supporting_evidence": [],
        }

    # --------------------------------------------------
    # SCORE
    # --------------------------------------------------

    try:

        score = int(data.get("score", 3))

    except (ValueError, TypeError):

        score = 3

    score = max(
        0,
        min(
            5,
            score,
        ),
    )

    data["score"] = score

    # --------------------------------------------------
    # REASONING
    # --------------------------------------------------

    if not isinstance(
        data.get("reasoning"),
        str,
    ) or not data["reasoning"].strip():

        data["reasoning"] = (
            "Accuracy was evaluated using the factual claims "
            "stated in the AI response."
        )

    # --------------------------------------------------
    # SUPPORTING EVIDENCE
    # --------------------------------------------------

    if not isinstance(
        data.get("supporting_evidence"),
        list,
    ):

        data["supporting_evidence"] = []

    return data