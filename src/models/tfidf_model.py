import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommenderTFIDF:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None
        )

        self.job_vectors = None
        self.df = None


    def load_data(self, path):
        """
        Загружает очищенный датасет
        """

        self.df = pd.read_csv(path)

        print("Данные загружены:", self.df.shape)


    def train(self):
        """
        Обучение TF-IDF модели
        """

        texts = self.df["text"]

        self.job_vectors = self.vectorizer.fit_transform(texts)

        print("TF-IDF модель обучена")


    def recommend(self, resume_text, top_k=5):
        """
        Возвращает наиболее похожие вакансии
        """

        resume_vector = self.vectorizer.transform([resume_text])

        similarities = cosine_similarity(resume_vector, self.job_vectors)

        top_indexes = similarities[0].argsort()[-top_k:][::-1]

        results = self.df.iloc[top_indexes]

        return results