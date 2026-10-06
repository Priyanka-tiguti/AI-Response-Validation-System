import json
from agents.base import call_llm


def relevance_agent(question, ai_response):

    prompt = f"""
You are a Relevance Judge Agent in an AI Response Validation System.

Your ONLY task is to judge whether the AI response addresses
the user's question.

You must evaluate RELEVANCE ONLY.

==================================================
QUESTION
==================================================

{question}

==================================================
AI RESPONSE
==================================================

{ai_response}

==================================================
CRITICAL SEPARATION RULE
==================================================

Relevance is NOT factual correctness.

You MUST NOT use factual correctness when assigning the
relevance score.

A response can be completely relevant even when its answer
is factually wrong.

For example:

Question:
What is the capital of Australia?

Response:
The capital of Australia is Sydney.

Correct evaluation:

Relevance = 5

Why:
The response directly answers the question by identifying
a city as the capital of Australia.

The fact that Sydney is incorrect is NOT a relevance issue.
It must be evaluated by the Accuracy and Hallucination agents.

==================================================
WHAT RELEVANCE MEANS
==================================================

Ask ONLY this question:

"Does the response attempt to answer or address what the
user asked?"

If YES, the response should generally receive a high
relevance score even if the answer is wrong, incomplete,
or unsupported.

==================================================
DO NOT PENALIZE RELEVANCE FOR
==================================================

Do NOT reduce relevance because:

1. The answer is factually incorrect.
2. The answer contradicts known facts.
3. The answer contradicts a reference answer.
4. The answer contains hallucinated information.
5. The answer is incomplete.
6. The answer is too short.
7. The answer is missing some required details.

These belong to other evaluation dimensions.

==================================================
SCORING
==================================================

5 = Directly answers the question.

4 = Directly answers the question but includes minor
    unnecessary information.

3 = Addresses the question but only covers part of it,
    especially for a multi-part question.

2 = Related to the topic but does not actually answer
    what was asked.

1 = Very weak connection to the question.

0 = Completely unrelated or off-topic.

==================================================
IMPORTANT EXAMPLES
==================================================

Example 1 — CORRECT AND RELEVANT

Question:
What is the capital of India?

Response:
The capital of India is New Delhi.

Score: 5

Reason:
The response directly answers the question.

--------------------------------------------------

Example 2 — WRONG BUT RELEVANT

Question:
What is the capital of Australia?

Response:
The capital of Australia is Sydney.

Score: 5

Reason:
The response directly attempts to answer the question.
The fact that Sydney is incorrect must NOT reduce relevance.

--------------------------------------------------

Example 3 — WRONG BUT RELEVANT

Question:
What is 2 + 2?

Response:
The answer is 5.

Score: 5

Reason:
The response directly answers the mathematical question.
Whether the answer is correct is an Accuracy issue.

--------------------------------------------------

Example 4 — INCOMPLETE BUT RELEVANT

Question:
What are the three states of matter?

Response:
Matter exists as solid.

Score: 5

Reason:
The response directly addresses the question by identifying
one of the requested states. Missing states are a Completeness
issue, not a Relevance issue.

--------------------------------------------------

Example 5 — IRRELEVANT

Question:
What is the capital of France?

Response:
France is a country in Western Europe known for its cuisine.

Score: 2

Reason:
The response discusses France but does not answer the
question about its capital.

--------------------------------------------------

Example 6 — COMPLETELY IRRELEVANT

Question:
What is the capital of France?

Response:
The human heart pumps blood through the body.

Score: 0

Reason:
The response does not address the question.

--------------------------------------------------

Example 7 — MULTI-PART QUESTION

Question:
What is the capital of India and who is the Prime Minister?

Response:
The capital of India is New Delhi.

Score: 3

Reason:
The response addresses one part of the question but does
not address the second part.

==================================================
DECISION PROCESS
==================================================

Before assigning a score, follow these steps:

Step 1:
Identify exactly what the question is asking.

Step 2:
Identify what the response is attempting to answer.

Step 3:
Determine whether the response addresses the requested
topic or task.

Step 4:
IGNORE whether the response is factually correct.

Step 5:
IGNORE whether information is missing.

Step 6:
Assign the relevance score.

IMPORTANT:
Never reason like this:

"The answer is wrong, therefore relevance is low."

That reasoning is INVALID.

Instead reason:

"The response directly attempts to answer the question,
therefore relevance is high. Accuracy is evaluated separately."

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "score": 5,
    "reasoning": "The response directly addresses the question."
}}

Rules:

- score must be an integer from 0 to 5.
- reasoning must explain ONLY relevance.
- Do not mention factual correctness in the reasoning unless
  explicitly explaining that correctness does not affect
  relevance.
- Do not evaluate completeness.
- Do not evaluate hallucination.
- Do not include Markdown.
- Do not include ```json.
- Do not include any text outside the JSON object.
"""

    result = call_llm(prompt)

    # --------------------------------------------------
    # CLEAN MODEL OUTPUT
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
                "The relevance evaluation model returned "
                "an invalid JSON response."
            )
        }

    # --------------------------------------------------
    # VALIDATE SCORE
    # --------------------------------------------------

    try:

        score = int(
            data.get(
                "score",
                3
            )
        )

    except (ValueError, TypeError):

        score = 3

    score = max(
        0,
        min(
            5,
            score
        )
    )

    data["score"] = score

    # --------------------------------------------------
    # REASONING FALLBACK
    # --------------------------------------------------

    if not data.get("reasoning"):

        data["reasoning"] = (
            "The response was evaluated based on how directly "
            "it addresses the question."
        )

    return data


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = "What is the capital of Australia?"

    ai_response = (
        "The capital of Australia is Sydney."
    )

    result = relevance_agent(
        question,
        ai_response
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )