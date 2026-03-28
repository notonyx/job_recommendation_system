import pandas as pd
from src.utils.text_preprocessing import clean_text


def prepare_dataset(input_path, output_path):

    # df = pd.read_csv(
    #     input_path,
    #     sep=",",
    #     engine="python",
    #     on_bad_lines="skip"
    # )

    df = pd.read_csv(input_path, sep=";")

    # удаляем строки без описания
    df = df.dropna(subset=["description"])

    # объединяем текст
    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["key_skills"].fillna("")
    )

    # очищаем текст
    df["text"] = df["text"].apply(clean_text)

    # df = df[["id", "text"]]  # до постпроцессинга результатов одинаковых

    df = df[["id", "title", "text"]]

    df.to_csv(output_path, index=False)

    print("Очищенный датасет сохранён:", df.shape)


if __name__ == "__main__":

    # prepare_dataset(
    #     "data/raw/First_15000_Jobs_Cleaned.csv",
    #     "data/processed/jobs_cleaned.csv"
    # )

    prepare_dataset(
        "data/raw/Jobs_Cleaned_Full.csv",
        "data/processed/jobs_cleaned_all.csv"
    )