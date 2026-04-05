import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def unique_by_title(results):
    seen = set()
    unique_rows = []

    for _, row in results.iterrows():
        title = row.get("title", "")

        if title not in seen:
            seen.add(title)
            unique_rows.append(row)

    return pd.DataFrame(unique_rows).reset_index(drop=True)

def diversity_filter(df, embeddings, threshold=0.85):
    selected = []
    selected_embeddings = []

    for i in range(len(df)):
        emb = embeddings[i]

        if len(selected_embeddings) == 0:
            selected.append(i)
            selected_embeddings.append(emb)
            continue

        sims = cosine_similarity([emb], selected_embeddings)[0]

        # если не слишком похож на уже выбранные
        if max(sims) < threshold:
            selected.append(i)
            selected_embeddings.append(emb)

    return df.iloc[selected].reset_index(drop=True)

def rerank(results, resume_vec, model):
    titles = results["title"].fillna("").tolist()
    descriptions = results["text"].fillna("").tolist()

    # кодируем отдельно
    title_embeddings = model.encode(titles, convert_to_numpy=True)
    desc_embeddings = model.encode(descriptions, convert_to_numpy=True)

    # считаем similarity
    title_sim = cosine_similarity(resume_vec, title_embeddings)[0]
    desc_sim = cosine_similarity(resume_vec, desc_embeddings)[0]

    # комбинируем (веса можно менять)
    results["rerank_score"] = (
        results["similarity"] * 0.6 +
        title_sim * 0.3 +
        desc_sim * 0.1
    )

    return results.sort_values(by="rerank_score", ascending=False).drop(columns=["rerank_score"])