import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class JobRecommenderBERT:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.job_embeddings = None
        self.df = None

    def load_data(self, path):
        self.df = pd.read_csv(path)
        print("Данные загружены:", self.df.shape)

    def encode_jobs(self):
        texts = self.df["text"].tolist()
        print("Векторизация вакансий BERT...")
        self.job_embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Векторизация завершена.")

    def recommend(self, resume_text, top_k=5):
        resume_vector = self.model.encode([resume_text])
        similarities = cosine_similarity(resume_vector, self.job_embeddings)
        top_indexes = similarities[0].argsort()[-top_k:][::-1]
        results = self.df.iloc[top_indexes]
        return results