import pandas as pd
import numpy as np
import os

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from src.utils.text_preprocessing import clean_text


class JobRecommenderHybrid:

    def __init__(self, top_n_bm25=100):

        self.df = None

        # BM25
        self.corpus = None
        self.bm25 = None
        self.top_n_bm25 = top_n_bm25

        # BERT
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None

    def load_data(self, path):
        self.df = pd.read_csv(path)
        print("Данные загружены:", self.df.shape)

    def prepare_bm25(self):
        print("Подготовка BM25...")

        self.corpus = [text.split() for text in self.df["text"]]
        self.bm25 = BM25Okapi(self.corpus)

        print("BM25 готов")

    def encode_jobs(self, embeddings_path="data/processed/job_embeddings.npy"):
        # ---------- ПРОБУЕМ ЗАГРУЗИТЬ ----------
        if os.path.exists(embeddings_path):
            print("Загрузка эмбеддингов из файла...")

            try:
                self.embeddings = np.load(embeddings_path)

                if len(self.embeddings) == len(self.df):
                    print("Эмбеддинги загружены, размер:", len(self.embeddings))
                    return
                else:
                    print("Размер не совпадает, пересчитываем...")

            except Exception as e:
                print("Ошибка загрузки:", e)
                print("Пересчитываем...")

        print("Кодирование вакансий BERT...")

        texts = self.df["text"].tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        np.save(embeddings_path, self.embeddings)
        print("Эмбеддинги сохранены:", embeddings_path)

    def recommend(self, resume_text, top_k=10):

        resume_text = clean_text(resume_text)
        query_tokens = resume_text.split()

        # ---------- ШАГ 1: BM25 ----------
        bm25_scores = self.bm25.get_scores(query_tokens)

        top_bm25_idx = np.argsort(bm25_scores)[-self.top_n_bm25:][::-1]

        # ---------- ШАГ 2: BERT ----------
        candidate_embeddings = self.embeddings[top_bm25_idx]

        resume_embedding = self.model.encode([resume_text])

        index = faiss.IndexFlatL2(candidate_embeddings.shape[1])
        index.add(candidate_embeddings)

        distances, indices = index.search(resume_embedding, top_k)

        final_indices = top_bm25_idx[indices[0]]

        results = self.df.iloc[final_indices].copy()
        results["similarity"] = 1 - distances[0]

        return results