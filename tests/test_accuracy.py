from agents.accuracy_agent import evaluate_accuracy


question = "What is the capital of France?"

ai_response = "London is the capital of France."

reference_context = "Paris is the capital of France."


result = evaluate_accuracy(
    question,
    ai_response,
    reference_context
)

print("Accuracy Agent Result:")
print(result)