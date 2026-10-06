import pandas as pd


def load_truthfulqa_file(path):
    df = pd.read_csv(path)

    #print("Number of rows:", len(df))
    #print("\nColumns:")
    #print(df.columns.tolist())

    #print("\nFirst 5 rows:")
    #print(df.head())

    return df


if __name__ == "__main__":
    path = r"D:\AI-Response-Validation-System\kb\data\TruthfulQA.csv"

    data = load_truthfulqa_file(path)