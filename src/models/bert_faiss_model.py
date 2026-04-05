import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.utils.text_preprocessing import clean_text
from src.utils.recommendation_postprocessing import unique_by_title, diversity_filter, rerank
import os
from tqdm import tqdm

class JobRecommenderBERTFAISS:

    def __init__(self, model_name="all-MiniLM-L6-v2", embeddings_path="data/processed/job_embeddings.npy", batch_size=128):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings_path = embeddings_path
        self.job_embeddings = None
        self.index = None
        self.batch_size = batch_size

    # def load_data(self, csv_path): # для main.py
    #     """Загрузка вакансий"""
    #     self.df = pd.read_csv(csv_path)
    #     print("Данные загружены:", self.df.shape)

    def load_data(self, df): # для web части
        """Теперь принимаем DataFrame, НЕ CSV"""
        self.df = df
        print("Данные загружены:", self.df.shape)

    def encode_jobs(self):
        """Создаём или загружаем эмбеддинги вакансий"""
        if os.path.exists(self.embeddings_path):
            print("Загрузка эмбеддингов из файла...")
            self.job_embeddings = np.load(self.embeddings_path)
            print("Размер эмбенддингов: ", len(self.job_embeddings))
        else:
            print("Векторизация вакансий BERT по батчам...")
            texts = self.df["text"].tolist()
            embeddings_list = []

            for i in tqdm(range(0, len(texts), self.batch_size), desc="Batches"):
                batch_texts = texts[i:i+self.batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                embeddings_list.append(batch_embeddings)

            self.job_embeddings = np.vstack(embeddings_list)
            np.save(self.embeddings_path, self.job_embeddings)
            print("Эмбеддинги сохранены:", self.embeddings_path)

        # FAISS индекс
        dim = self.job_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity через inner product
        faiss.normalize_L2(self.job_embeddings)
        self.index.add(self.job_embeddings)
        print("FAISS индекс построен. Вакансии готовы к поиску.")

    # def recommend(self, resume_text, top_k=10):
    #     """Рекомендации вакансий по резюме"""
    #     resume_text = clean_text(resume_text)
    #     resume_vec = self.model.encode([resume_text], convert_to_numpy=True)
    #     faiss.normalize_L2(resume_vec)
    #     D, I = self.index.search(resume_vec, top_k)
    #     results = self.df.iloc[I[0]].copy()
    #     results["similarity"] = D[0]
    #     return results

    def recommend(self, resume_text, top_k=10):
        resume_text = clean_text(resume_text)
        resume_vec = self.model.encode([resume_text], convert_to_numpy=True)
        faiss.normalize_L2(resume_vec)
        # берём больше кандидатов (важно для diversity)
        D, I = self.index.search(resume_vec, top_k * 5)
        results = self.df.iloc[I[0]].copy()
        results["similarity"] = D[0]
        # 👉 embeddings для этих кандидатов
        candidate_embeddings = self.job_embeddings[I[0]]
        # ---------- 1. убираем дубли ----------
        results = unique_by_title(results)
        # после удаления дублей нужно синхронизировать embeddings
        candidate_embeddings = candidate_embeddings[:len(results)]
        # # ---------- 2. diversity ----------
        # results = diversity_filter(results, candidate_embeddings, threshold=0.85)
        # # ---------- 3. rerank ----------
        # results = rerank(results, resume_vec, self.model)
        return results.head(top_k)