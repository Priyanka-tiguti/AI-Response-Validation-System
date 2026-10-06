import json
from agents.base import call_llm


def completeness_agent(question, ai_response, reference_context=None):

    prompt = f"""
You are the COMPLETENESS JUDGE in an AI Response Validation System.

Your ONLY responsibility is to determine whether the AI response
provides all information requested by the user's question.

You are NOT an accuracy judge.
You are NOT a hallucination judge.
You are NOT a relevance judge.

==================================================
USER QUESTION
==================================================

{question}

==================================================
AI RESPONSE
==================================================

{ai_response}

==================================================
CORE RULE
==================================================

Completeness means:

"Did the response address all meaningful requirements
contained in the question?"

Evaluate the QUESTION first.

Then determine which requirements the AI response addressed.

==================================================
IMPORTANT RULES
==================================================

1. Do NOT judge whether the information is factually correct.

2. Do NOT lower completeness because a stated fact is wrong.

3. Do NOT treat a factual error as a missing component.

4. Do NOT require unnecessary information.

5. Do NOT require every sentence or detail from a reference answer.

6. A short answer can still be complete when the question asks
   for only one simple fact.

7. If the question asks for multiple items, concepts, steps,
   reasons, benefits, examples, comparisons, or parts, determine
   whether the response addresses ALL of them.

8. If the response addresses only some requested items,
   completeness must be reduced.

==================================================
IDENTIFYING MULTI-PART QUESTIONS
==================================================

Pay special attention to questions containing phrases such as:

- What are the steps...?
- What are the main steps...?
- What are the types...?
- What are the causes...?
- What are the benefits...?
- What are the advantages and disadvantages...?
- What are the components...?
- What are the characteristics...?
- Explain the process...
- Describe the workflow...
- What factors...?
- What methods...?
- Compare X and Y...
- List...
- Give examples...
- Explain the stages...
- What are the key features...?

These questions usually require MULTIPLE meaningful components.

Do NOT assume that mentioning one component makes the answer complete.

==================================================
CRITICAL EXAMPLE 1 — MULTIPLE ITEMS
==================================================

Question:
What are the three states of matter?

Response:
Matter exists as solid.

The question explicitly asks for THREE items.

Addressed:
- solid

Missing:
- liquid
- gas

Completeness:
2

==================================================
CRITICAL EXAMPLE 2 — ALL ITEMS
==================================================

Question:
What are the three states of matter?

Response:
Solid, liquid, and gas.

All requested items are addressed.

Completeness:
5

==================================================
CRITICAL EXAMPLE 3 — ONE FACT
==================================================

Question:
What is the capital of Australia?

Response:
The capital of Australia is Sydney.

The question asks for ONE fact.

The response provides an answer to that fact.

Completeness:
5

Whether Sydney is correct is an Accuracy issue.

==================================================
CRITICAL EXAMPLE 4 — TWO REQUIREMENTS
==================================================

Question:
What is the capital of France and what is its population?

Response:
The capital of France is Paris.

Two requirements exist:

1. Capital of France
2. Population

Only the first is addressed.

Completeness:
3

==================================================
CRITICAL EXAMPLE 5 — MACHINE LEARNING WORKFLOW
==================================================

Question:
What are the main steps in a machine learning workflow?

Response:
The main steps are collecting data and training the model.

This is NOT automatically complete.

The question asks for the MAIN STEPS of a workflow.

A typical response should address the meaningful stages of the
workflow, such as:

- problem definition
- data collection
- data preprocessing
- exploratory analysis
- feature engineering
- model selection
- model training
- evaluation
- tuning
- deployment or monitoring

The exact number of stages can vary.

However, mentioning only:

- collecting data
- training the model

does NOT address the workflow sufficiently.

Completeness:
LOW / PARTIAL

A suitable score is approximately 2 or 3 depending on how much
of the requested workflow is actually covered.

==================================================
CRITICAL EXAMPLE 6 — BENEFITS
==================================================

Question:
What are the benefits of exercise?

Response:
Exercise improves cardiovascular health.

The response provides one benefit, but the question asks for
benefits in general.

If multiple important benefits are expected, the answer is
partially complete.

Do not require an arbitrary number of benefits.

Judge whether the response provides a meaningful coverage of
the requested topic.

==================================================
CRITICAL EXAMPLE 7 — ADVANTAGES AND DISADVANTAGES
==================================================

Question:
What are the advantages and disadvantages of cloud computing?

Response:
Cloud computing provides scalability and reduces infrastructure
costs.

The response addresses advantages.

It does not address disadvantages.

Therefore the answer is incomplete.

==================================================
CRITICAL EXAMPLE 8 — EXPLANATION
==================================================

Question:
Explain photosynthesis.

Response:
Plants use sunlight to make food.

This addresses the basic concept but provides only a minimal
explanation.

The answer is partially complete because "explain" requests
more than simply naming the concept.

Completeness:
approximately 3

==================================================
CRITICAL EXAMPLE 9 — SIMPLE DEFINITION
==================================================

Question:
What is an API?

Response:
An API allows different software components to communicate
with each other.

The question asks for a definition.

The response provides the requested definition.

Completeness:
5

==================================================
CRITICAL EXAMPLE 10 — LIST REQUEST
==================================================

Question:
List the primary colors of light.

Response:
Red, green, and blue.

All requested items are provided.

Completeness:
5

==================================================
CRITICAL EXAMPLE 11 — COMPARISON
==================================================

Question:
What is the difference between RAM and ROM?

Response:
RAM is temporary memory.

The response addresses RAM but does not explain ROM or the
difference between them.

Completeness:
2 or 3

==================================================
CRITICAL EXAMPLE 12 — WRONG BUT COMPLETE
==================================================

Question:
What is the capital of Australia?

Response:
The capital of Australia is Sydney.

The answer is factually wrong.

However, the question asks for ONE fact and the response
provides ONE answer.

Completeness:
5

Accuracy is responsible for determining whether Sydney is correct.

==================================================
CRITICAL EXAMPLE 13 — INCOMPLETE BUT CORRECT
==================================================

Question:
What are the main steps in a machine learning workflow?

Response:
The first step is collecting data.

The statement may be factually correct.

However, most of the requested workflow is missing.

Completeness:
1 or 2

==================================================
HOW TO EVALUATE
==================================================

STEP 1:
Read the question carefully.

STEP 2:
Identify every meaningful requirement in the question.

STEP 3:
Determine whether the question expects:
- one fact
- multiple facts
- a list
- multiple steps
- an explanation
- a comparison
- advantages/disadvantages
- causes/effects
- components
- examples
- stages
- another multi-part response

STEP 4:
Read the AI response.

STEP 5:
Identify which requested requirements are addressed.

STEP 6:
Identify which meaningful requirements remain missing.

STEP 7:
Assign the completeness score.

IMPORTANT:

Do NOT compare the response against the reference answer
as a checklist.

The reference may help provide context, but completeness
must primarily come from what the QUESTION requests.

==================================================
SCORING
==================================================

5 = Fully Complete

All meaningful requirements of the question are addressed.

4 = Mostly Complete

Almost all important requirements are addressed.
Only minor information is missing.

3 = Partially Complete

The main request is addressed, but one or more important
requirements are missing.

2 = Incomplete

Only part of the requested information is provided.

1 = Very Incomplete

Very little of the requested information is provided.

0 = Completely Unresponsive

The response does not meaningfully address the question.

==================================================
IMPORTANT SCORING GUIDANCE
==================================================

For a question asking for ONE simple fact:

One direct answer → normally 5.

For a question asking for MULTIPLE explicit items:

Only some items → reduce the score.

For a question asking for STEPS / WORKFLOW / PROCESS:

Do not treat two generic steps as automatically complete.

For a question asking for ADVANTAGES AND DISADVANTAGES:

Both sides must be addressed.

For a COMPARISON:

Both subjects and the requested comparison must be addressed.

For an EXPLANATION:

A bare keyword or extremely short statement may be incomplete.

==================================================
REFERENCE CONTEXT
==================================================

Reference context may be available:

{reference_context if reference_context else "No reference context available."}

IMPORTANT:

Do NOT use the reference context as a checklist.

Do NOT mark something missing merely because it appears
in the reference.

The question determines what information is required.

==================================================
FINAL CHECK
==================================================

Before returning the result, verify:

- Did I evaluate ONLY completeness?
- Did I identify what the question actually requests?
- Did I identify all meaningful requirements?
- Did I check which requirements the response addresses?
- Did I identify genuinely missing requirements?
- Did I avoid judging factual correctness?
- Did I avoid judging hallucination?
- Did I avoid requiring unnecessary details?
- If the question asks for one fact and one answer is provided,
  did I normally give 5?
- If the question asks for multiple steps/items and only some
  are provided, did I reduce the score?

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Use exactly:

{{
    "score": 0,
    "addressed_components": [],
    "missing_components": [],
    "reasoning": "Brief explanation focused only on completeness."
}}

Rules:

- score must be an integer from 0 to 5.
- addressed_components must be a JSON list.
- missing_components must be a JSON list.
- reasoning must be a string.
- Do NOT mention factual correctness.
- Do NOT mention hallucination.
- Do NOT evaluate relevance.
- Do NOT list factual errors as missing components.
- Return only raw JSON.
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
            "addressed_components": [],
            "missing_components": [],
            "reasoning": (
                "The completeness evaluation returned invalid JSON."
            ),
        }

    # --------------------------------------------------
    # VALIDATE OBJECT
    # --------------------------------------------------

    if not isinstance(data, dict):

        return {
            "score": 3,
            "addressed_components": [],
            "missing_components": [],
            "reasoning": (
                "The completeness evaluation returned an invalid object."
            ),
        }

    # --------------------------------------------------
    # SCORE
    # --------------------------------------------------

    try:
        score = int(data.get("score", 3))
    except (ValueError, TypeError):
        score = 3

    score = max(0, min(5, score))
    data["score"] = score

    # --------------------------------------------------
    # ADDRESSED COMPONENTS
    # --------------------------------------------------

    if not isinstance(
        data.get("addressed_components"),
        list,
    ):
        data["addressed_components"] = []

    # --------------------------------------------------
    # MISSING COMPONENTS
    # --------------------------------------------------

    if not isinstance(
        data.get("missing_components"),
        list,
    ):
        data["missing_components"] = []

    # --------------------------------------------------
    # REASONING
    # --------------------------------------------------

    if not isinstance(
        data.get("reasoning"),
        str,
    ) or not data["reasoning"].strip():

        data["reasoning"] = (
            "Completeness was evaluated based on the "
            "requirements of the question."
        )

    return data