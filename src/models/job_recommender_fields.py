import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommenderFields:

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        self.model = SentenceTransformer(model_name)

        self.df = None

        self.title_embeddings = None
        self.skills_embeddings = None
        self.desc_embeddings = None

        self.index_desc = None


    def load_data(self, path):

        self.df = pd.read_csv(path, sep=";")

        # удаляем пустые значения
        self.df = self.df.dropna(subset=["title", "description"])

        # если навыков нет — ставим пустую строку
        self.df["key_skills"] = self.df["key_skills"].fillna("")

        print("Dataset loaded:", self.df.shape)


    def encode_jobs(self):

        print("Encoding titles...")
        titles = self.df["title"].tolist()
        self.title_embeddings = self.model.encode(titles, convert_to_numpy=True)

        print("Encoding skills...")
        skills = self.df["key_skills"].tolist()
        self.skills_embeddings = self.model.encode(skills, convert_to_numpy=True)

        print("Encoding descriptions...")
        desc = self.df["description"].tolist()
        self.desc_embeddings = self.model.encode(desc, convert_to_numpy=True)

        # нормализация
        faiss.normalize_L2(self.desc_embeddings)

        dim = self.desc_embeddings.shape[1]

        self.index_desc = faiss.IndexFlatIP(dim)

        self.index_desc.add(self.desc_embeddings)

        print("FAISS index built.")


    def recommend(self, resume_text, top_k=10):

        resume_vec = self.model.encode([resume_text], convert_to_numpy=True)
        faiss.normalize_L2(resume_vec)

        # ищем top 50 кандидатов по описанию
        D, I = self.index_desc.search(resume_vec, 50)

        candidates = I[0]

        title_sims = cosine_similarity(
            resume_vec, self.title_embeddings[candidates]
        )[0]

        skills_sims = cosine_similarity(
            resume_vec, self.skills_embeddings[candidates]
        )[0]

        desc_sims = cosine_similarity(
            resume_vec, self.desc_embeddings[candidates]
        )[0]

        scores = (
            0.5 * skills_sims +
            0.3 * title_sims +
            0.2 * desc_sims
        )

        sorted_idx = np.argsort(scores)[::-1][:top_k]

        final_ids = candidates[sorted_idx]

        results = self.df.iloc[final_ids].copy()

        results["score"] = scores[sorted_idx]

        return results