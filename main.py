from src.models.tfidf_model import JobRecommenderTFIDF
from src.utils.text_preprocessing import clean_text
from src.models.bert_model import JobRecommenderBERT
from src.models.bert_faiss_model import JobRecommenderBERTFAISS
from src.models.job_recommender_fields import JobRecommenderFields
from src.utils.resume_reader import read_resume
from src.utils.recommendation_postprocessing import unique_by_title
from src.models.hybrid_model import JobRecommenderHybrid


import time
import psutil
import os

def main():

    start_time = time.time()
    process = psutil.Process(os.getpid())

    # --------------- TF IDF -------------------

    recommender = JobRecommenderTFIDF()

    recommender.load_data("data/processed/jobs_cleaned.csv")
    # recommender.load_data("data/processed/jobs_cleaned_all.csv")

    recommend_start = time.time()
    recommender.train()


    # resume_path = "data/resumes/ramir_resume.pdf"
    # resume = read_resume(resume_path)
    # resume = clean_text(resume)

    # resume = """
    # Python developer
    # Django
    # REST API
    # PostgreSQL
    # Docker
    # """

    resume = """
    Я опытный Python-разработчик с более чем 3 годами работы в веб-разработке и бэкенд-проектах. 
    Имею глубокие знания Django и Flask, умею создавать REST API и интегрировать сторонние сервисы. 
    Работал с PostgreSQL и MySQL, умею проектировать базы данных и оптимизировать запросы. 
    Опыт работы с Docker и Docker Compose для контейнеризации приложений, CI/CD, настройка Git для командной разработки. 
    Знаком с Linux-серверами, умею настраивать виртуальные окружения, мониторинг и логирование. 
    Имею опыт работы с Redis, Celery, RabbitMQ для асинхронной обработки задач. 
    Участвовал в создании масштабируемых веб-приложений и интеграции сторонних API, тестировал код через pytest. 
    Готов работать как на стартап-проектах, так и на крупных продуктивных системах.
    """

    # resume = """
    # Я опытный разработчик с более чем 3 годами работы. 
    # Аналитик
    # Excel
    # ML
    # Тестировщик
    # Python
    # Java
    # C++
    # CI/CD
    # Docker
    # """

    resume = clean_text(resume)

    results = recommender.recommend(resume, top_k=10)
    recommend_end = time.time()

    end_time = time.time()
    memory_mb = process.memory_info().rss / 1024 / 1024

    print("\nРекомендуемые вакансии:\n")

    for i, row in results.iterrows():

        print("ID:", row["id"])
        print("Text:", row["text"][:200])
        print()

    print("\n========= PERFORMANCE =========")

    print(f"Общее время: {end_time - start_time:.2f} сек")

    print(f"Время поиска: {recommend_end - recommend_start:.4f} сек")

    print(f"Использование памяти: {memory_mb:.2f} MB")

    # --------------- BERT -------------------
    
    # recommender = JobRecommenderBERT()
    # recommender.load_data("data/processed/jobs_cleaned.csv")

    # encode_start = time.time()
    # recommender.encode_jobs()
    # encode_end = time.time()

    # resume = clean_text(resume)

    # recommend_start = time.time()
    # results = recommender.recommend(resume, top_k=10)
    # recommend_end = time.time()

    # end_time = time.time()
    # memory_mb = process.memory_info().rss / 1024 / 1024

    # print("\nРекомендуемые вакансии (BERT):\n")
    # for i, row in results.iterrows():
    #     print("ID:", row["id"])
    #     print("Text:", row["text"][:200])
    #     print()

    # print("\n========= PERFORMANCE =========")

    # print(f"Общее время: {end_time - start_time:.2f} сек")

    # print(f"Время векторизации: {encode_end - encode_start:.2f} сек")

    # print(f"Время поиска: {recommend_end - recommend_start:.4f} сек")

    # print(f"Использование памяти: {memory_mb:.2f} MB")

    # --------------- BERT + FAISS -------------------

    # resume = """
    # Python developer
    # Django
    # REST API
    # PostgreSQL
    # Docker
    # """

    # resume = """
    # Я опытный Python-разработчик с более чем 3 годами работы в веб-разработке и бэкенд-проектах. 
    # Имею глубокие знания Django и Flask, умею создавать REST API и интегрировать сторонние сервисы. 
    # Работал с PostgreSQL и MySQL, умею проектировать базы данных и оптимизировать запросы. 
    # Опыт работы с Docker и Docker Compose для контейнеризации приложений, CI/CD, настройка Git для командной разработки. 
    # Знаком с Linux-серверами, умею настраивать виртуальные окружения, мониторинг и логирование. 
    # Имею опыт работы с Redis, Celery, RabbitMQ для асинхронной обработки задач. 
    # Участвовал в создании масштабируемых веб-приложений и интеграции сторонних API, тестировал код через pytest. 
    # Готов работать как на стартап-проектах, так и на крупных продуктивных системах.
    # """

    # resume_path = "data/resumes/resume_pdf.pdf"

    # print(resume)

    # resume = """
    # Я опытный маркетолог с 5-летним стажем работы в digital-рекламе и продвижении брендов. 
    # Занимался созданием контент-планов, запуском рекламных кампаний в социальных сетях, 
    # анализировал эффективность рекламы через Google Analytics и Yandex.Metrica. 
    # Имею опыт работы с SEO, SMM, email-маркетингом и созданием лендингов. 
    # Люблю работать с командами, ставить KPI и улучшать показатели продаж. 
    # Ищу позицию, где могу развивать бренд и повышать узнаваемость компании.
    # """

    # results = recommender.recommend(resume, top_k=10)

# ----------------------------------------------------------------------------------------------

    # recommender = JobRecommenderBERTFAISS(batch_size=256)
    # recommender.load_data("data/processed/jobs_cleaned_all.csv")

    # encode_start = time.time()
    # recommender.encode_jobs()  # создаёт эмбеддинги и FAISS индекс
    # encode_end = time.time()


    # recommend_start = time.time()
    # results = recommender.recommend(resume, top_k=30)
    # recommend_end = time.time()

    # results = unique_by_title(results)
    # results = results[:10]

    # end_time = time.time()
    # memory_mb = process.memory_info().rss / 1024 / 1024

    # print("\nРекомендуемые вакансии (BERT + FAISS):\n")
    # for i, row in results.iterrows():
    #     print("ID:", row["id"])
    #     print("Text:", row["text"][:200])
    #     print("Similarity:", row["similarity"])
    #     print()

    # print("\n========= PERFORMANCE =========")

    # print(f"Общее время: {end_time - start_time:.2f} сек")

    # print(f"Время векторизации: {encode_end - encode_start:.2f} сек")

    # print(f"Время поиска: {recommend_end - recommend_start:.4f} сек")

    # print(f"Использование памяти: {memory_mb:.2f} MB")


# -------  Hybrid система (не очень)  ---------------------------------------------------------------------------------------

    # recommender = JobRecommenderHybrid(top_n_bm25=100)

    # recommender.load_data("data/processed/jobs_cleaned_all.csv")

    # bm25_start = time.time()
    # recommender.prepare_bm25()
    # bm25_end = time.time()

    # encode_start = time.time()
    # recommender.encode_jobs()
    # encode_end = time.time()

    # recommend_start = time.time()
    # results = recommender.recommend(resume, top_k=10)
    # recommend_end = time.time()

    # end_time = time.time()
    # memory_mb = process.memory_info().rss / 1024 / 1024

    # print("\nРекомендуемые вакансии (hybrid model):\n")
    # for i, row in results.iterrows():
    #     print("ID:", row["id"])
    #     print("Text:", row["text"][:200])
    #     print("Similarity:", row["similarity"])
    #     print()

    # print("\n========= PERFORMANCE =========")

    # print(f"Общее время: {end_time - start_time:.2f} сек")

    # print(f"Время prepare_bm25: {bm25_end - bm25_start:.2f} сек")

    # print(f"Время векторизации: {encode_end - encode_start:.2f} сек")

    # print(f"Время поиска: {recommend_end - recommend_start:.4f} сек")

    # print(f"Использование памяти: {memory_mb:.2f} MB")

# ----------------------------------------------------------------------------------------------

    # recommender = JobRecommenderFields()

    # recommender.load_data(
    #     "data/raw/First_15000_Jobs_Cleaned.csv"
    # )

    # recommender.encode_jobs()

    # resume = """
    # Я опытный Python-разработчик с более чем 3 годами работы в веб-разработке и бэкенд-проектах. 
    # Имею глубокие знания Django и Flask, умею создавать REST API и интегрировать сторонние сервисы. 
    # Работал с PostgreSQL и MySQL, умею проектировать базы данных и оптимизировать запросы. 
    # Опыт работы с Docker и Docker Compose для контейнеризации приложений, CI/CD, настройка Git для командной разработки. 
    # Знаком с Linux-серверами, умею настраивать виртуальные окружения, мониторинг и логирование. 
    # Имею опыт работы с Redis, Celery, RabbitMQ для асинхронной обработки задач. 
    # Участвовал в создании масштабируемых веб-приложений и интеграции сторонних API, тестировал код через pytest. 
    # Готов работать как на стартап-проектах, так и на крупных продуктивных системах.
    # """

    # results = recommender.recommend(resume, top_k=10)

    # print("\nRecommended jobs:\n")

    # for _, row in results.iterrows():

    #     print("ID:", row["id"])
    #     print("Title:", row["title"])
    #     print("Skills:", row["key_skills"])
    #     print("Score:", row["score"])
    #     print()

if __name__ == "__main__":
    main()