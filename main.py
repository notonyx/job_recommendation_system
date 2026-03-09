from src.models.tfidf_model import JobRecommenderTFIDF
from src.utils.text_preprocessing import clean_text
from src.models.bert_model import JobRecommenderBERT
from src.models.bert_faiss_model import JobRecommenderBERTFAISS


def main():

    # --------------- TF IDF -------------------

    # recommender = JobRecommenderTFIDF()

    # recommender.load_data("data/processed/jobs_cleaned.csv")

    # recommender.train()

    # resume = """
    # Python developer
    # Django
    # REST API
    # PostgreSQL
    # Docker
    # """

    # resume = clean_text(resume)

    # results = recommender.recommend(resume, top_k=10)

    # print("\nРекомендуемые вакансии:\n")

    # for i, row in results.iterrows():

    #     print("ID:", row["id"])
    #     print("Text:", row["text"][:200])
    #     print()

    # --------------- BERT -------------------
    
    # recommender = JobRecommenderBERT()
    # recommender.load_data("data/processed/jobs_cleaned.csv")
    # recommender.encode_jobs()

    # resume = """
    # Python developer
    # Django
    # REST API
    # PostgreSQL
    # Docker
    # """
    # resume = clean_text(resume)

    # results = recommender.recommend(resume, top_k=10)

    # print("\nРекомендуемые вакансии (BERT):\n")
    # for i, row in results.iterrows():
    #     print("ID:", row["id"])
    #     print("Text:", row["text"][:200])
    #     print()

    # --------------- BERT + FAISS -------------------
    
    recommender = JobRecommenderBERTFAISS(batch_size=256)
    recommender.load_data("data/processed/jobs_cleaned_all.csv")
    recommender.encode_jobs()  # создаёт эмбеддинги и FAISS индекс

    resume = """
    Python developer
    Django
    REST API
    PostgreSQL
    Docker
    """

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

    # resume = """
    # Я опытный маркетолог с 5-летним стажем работы в digital-рекламе и продвижении брендов. 
    # Занимался созданием контент-планов, запуском рекламных кампаний в социальных сетях, 
    # анализировал эффективность рекламы через Google Analytics и Yandex.Metrica. 
    # Имею опыт работы с SEO, SMM, email-маркетингом и созданием лендингов. 
    # Люблю работать с командами, ставить KPI и улучшать показатели продаж. 
    # Ищу позицию, где могу развивать бренд и повышать узнаваемость компании.
    # """

    results = recommender.recommend(resume, top_k=10)

    print("\nРекомендуемые вакансии (BERT + FAISS):\n")
    for i, row in results.iterrows():
        print("ID:", row["id"])
        print("Text:", row["text"][:200])
        print("Similarity:", row["similarity"])
        print()

if __name__ == "__main__":
    main()