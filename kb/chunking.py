from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from preprocess import cleaned_docs


def split_documents(docs, chunk_size=500, chunk_overlap=50):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = []

    for doc in docs:

        question = doc["question"]
        context = doc["context"]
        answer = doc["answer"]

        # Split only the context
        context_chunks = text_splitter.split_text(context)

        for chunk in context_chunks:

            # Add question and answer to every context chunk
            text = (
                f"Question: {question}\n"
                f"Context: {chunk}\n"
                f"Answer: {answer}"
            )

            new_doc = Document(
                page_content=text,
                metadata={
                    "question": question,
                    "answer": answer,
                    "source": doc["source"],
                    "title": doc["title"]
                }
            )

            split_docs.append(new_doc)

    return split_docs


# Create SQuAD chunks
chunks = split_documents(cleaned_docs)


def chunk_truthfulqa(df):

    documents = []

    for _, row in df.iterrows():

        text = (
            f"Question: {row['Question']}\n"
            f"Answer: {row['Best Answer']}"
        )

        document = Document(
            page_content=text,
            metadata={
                "source": "truthfulqa",
                "category": row["Category"],
                "type": row["Type"],
                "question": row["Question"],
                "answer": row["Best Answer"]
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    from load_truthfulqa import load_truthfulqa_file
    from preprocess import clean_truthfulqa_data

    path = r"D:\AI-Response-Validation-System\kb\data\TruthfulQA.csv"

    df = load_truthfulqa_file(path)

    cleaned_df = clean_truthfulqa_data(df)

    truthful_chunks = chunk_truthfulqa(cleaned_df)

    print("SQuAD chunks:", len(chunks))
    print("TruthfulQA rows:", len(cleaned_df))
    print("TruthfulQA chunks:", len(truthful_chunks))

    print("\nFirst SQuAD chunk:")
    print(chunks[0])

    print("\nFirst TruthfulQA chunk:")
    print(truthful_chunks[0])