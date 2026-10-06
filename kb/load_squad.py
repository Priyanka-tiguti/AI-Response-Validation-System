import json


def load_squad_file(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for article in data["data"]:

        title = article["title"]

        for paragraph in article["paragraphs"]:

            context = paragraph["context"]

            for qa in paragraph["qas"]:

                question = qa["question"]

                answers = qa["answers"]

                answer = answers[0]["text"] if answers else ""

                documents.append({
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "source": "squad",
                    "title": title
                })

    return documents


train_documents = load_squad_file(
    "data/squad_train.json"
)

validation_documents = load_squad_file(
    "data/squad_validation.json"
)

documents = train_documents + validation_documents

#print("Total documents:", len(documents))

#print("\nFirst document:")
#print(documents[0])
