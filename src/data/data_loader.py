import pandas as pd


def load_jobs(path):
    """
    Загружает датасет вакансий
    """

    df = pd.read_csv(path)

    print("Размер датасета:", df.shape)

    return df


if __name__ == "__main__":

    # df = load_jobs("../../data/raw/First_15000_Jobs_Cleaned.csv")
    df = load_jobs("../../data/raw/Jobs_Cleaned_Full.csv")

    print(df.head())