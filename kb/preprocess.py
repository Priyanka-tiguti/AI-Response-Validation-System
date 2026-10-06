from load_squad import documents

def clean_data(docs):
    cleaned_docs = []
    # Remove whitespace from docs
    for doc in docs:
        doc["question"] = doc["question"].strip()
        doc["answer"] = doc["answer"].strip()
        doc["context"] = doc["context"].strip()
        doc["title"] = doc["title"].strip()
        doc["source"] = doc["source"].strip()

        # Remove docs with empty fields
        if doc["question"] and doc["answer"] and doc["context"] and doc["title"] and doc["source"]:
            cleaned_docs.append(doc)
            
    # Remove duplicates from cleaned_docs
    seen = set()
    unique_docs = []
    for doc in cleaned_docs:
        key = (doc["question"], doc["answer"], doc["context"], doc["title"], doc["source"])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

cleaned_docs = clean_data(documents)

#print("Original documents:", len(documents))
#print("Cleaned documents:", len(cleaned_docs))

#print("\nFirst cleaned document:")
#print(cleaned_docs[0])

def clean_truthfulqa_data(df):
    # Remove rows where Question or Best Answer is missing
    df = df.dropna(subset=["Question", "Best Answer"])

    # Remove extra spaces
    df["Question"] = df["Question"].str.strip()
    df["Best Answer"] = df["Best Answer"].str.strip()
    df["Category"] = df["Category"].str.strip()
    df["Type"] = df["Type"].str.strip()
    df["Source"] = df["Source"].str.strip()

    # Remove empty questions and answers
    df = df[
        (df["Question"] != "") &
        (df["Best Answer"] != "")
    ]

    # Remove duplicate questions
    df = df.drop_duplicates(subset=["Question"])

    return df

if __name__ == "__main__":

    from load_truthfulqa import load_truthfulqa_file

    path = r"D:\AI-Response-Validation-System\kb\data\TruthfulQA.csv"

    df = load_truthfulqa_file(path)

    cleaned_df = clean_truthfulqa_data(df)

    #print("\nOriginal rows:", len(df))
    #print("Cleaned rows:", len(cleaned_df))