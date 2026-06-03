import pandas as pd
from src.utils.text_preprocessing import clean_text


def prepare_dataset(input_path, output_path):
    df = pd.read_csv(input_path, sep=";")

    df = df.dropna(subset=["description"])

    df["text"] = (
        (df["title"].fillna("") + " ") * 3 +
        (df["key_skills"].fillna("") + " ") * 2 +
        df["description"].fillna("")
    )

    df["text"] = df["text"].apply(clean_text)
    df = df[["id", "title", "text"]]
    df.to_csv(output_path, index=False)

    print("Очищенный датасет сохранён:", df.shape)


if __name__ == "__main__":
    prepare_dataset(
        "data/raw/Jobs_Cleaned_Full.csv",
        "data/processed/jobs_cleaned_all.csv"
    )