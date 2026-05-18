from django.shortcuts import render, redirect
import pandas as pd
import os
from src.models.bert_faiss_model import JobRecommenderBERTFAISS
from src.utils.resume_reader import read_resume, normalize_resume, clean_resume, read_hh_resume
from src.utils.text_preprocessing import clean_text
from .forms import ResumeUploadForm

import time
import psutil

jobs_full = pd.read_csv("data/raw/Jobs_Cleaned_Full.csv", sep=";")

recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        df = pd.read_csv("data/processed/jobs_cleaned_all.csv")

        recommender = JobRecommenderBERTFAISS()
        recommender.load_data(df)
        recommender.encode_jobs()

    return recommender


def index(request):
    return render(request, 'web/index.html')

def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)

        if form.is_valid():
            file = request.FILES.get('file')
            text_input = form.cleaned_data.get("text")

            if text_input:
                text = text_input

            elif file:
                path = f"temp_{file.name}"
                with open(path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                text = read_resume(path)
                os.remove(path)
            else:
                return render(request, 'web/upload.html', {'form': form})

            request.session['resume_text'] = text
            return redirect('results')
    else:
        form = ResumeUploadForm()

    return render(request, 'web/upload.html', {'form': form})

def format_commas(text):
    if isinstance(text, str):
        return ", ".join([t.strip() for t in text.split(",")])
    return text

def format_salary(s):
    if pd.isna(s) or s == "Не указано":
        return "Не указано"
    return s

import re

def format_description(text):
    if not isinstance(text, str):
        return text

    sections = [
        "Обязанности:",
        "Требования:",
        "Условия:",
        "Ключевые навыки:",
        "Навыки:"
    ]

    for section in sections:
        # если раздел НЕ в начале текста
        text = re.sub(
            rf"\s+{re.escape(section)}",
            f"\n\n{section}\n",
            text
        )

        # если раздел в начале текста
        if text.startswith(section):
            text = text.replace(section, f"{section}\n", 1)

    return text.strip()

def is_hh_resume(text):
    keywords = [
        "желаемая должность и зарплата",
        "предпочитаемый способ связи",
        "проживает",
        "гражданство",
        "специализации",
        "тип занятости",
        "занятость",
        "график работы",
        "формат работы",
        "желательное время в пути до работы",
        "образование",
        "повышение квалификации",
        "курсы"
    ]

    for k in keywords:
        if k.lower() in text.lower():
            print(k)

    score = sum(1 for k in keywords if k.lower() in text.lower())

    print("score: ", score)
    return score >= 7


def results(request):
    start_time = time.time()
    process = psutil.Process(os.getpid())

    resume_text = request.session.get("resume_text", "")
    print("resume_text:\n\n",resume_text, "\n\n")
    resume_text = normalize_resume(resume_text)

    if is_hh_resume(resume_text):
        resume_text = read_hh_resume(resume_text)
    else:
        resume_text = clean_resume(resume_text)
    
    resume_text = clean_text(resume_text)

    recommend_start = time.time()
    model = get_recommender()
    jobs = model.recommend(resume_text)
    recommend_end = time.time()

    print(jobs.head())
    print(jobs.columns)

    jobs["similarity_percent"] = jobs["similarity"] * 100
    jobs_list = jobs.to_dict(orient="records")

    for job in jobs_list:
        job_id = int(job.get("id"))
        full_row = jobs_full[jobs_full["id"] == job_id]

        if not full_row.empty:
            full_row = full_row.iloc[0]
            job["description"] = full_row.get("description")
            job["salary"] = full_row.get("salary")
            job["experience"] = full_row.get("experience")
            job["job_type"] = full_row.get("job_type")

            job["salary"] = format_salary(job["salary"])
            # job["description"] = format_description(job["description"])
            job["job_type"] = format_commas(job["job_type"])
            job["experience"] = format_commas(job["experience"])

    end_time = time.time()

    memory_mb = process.memory_info().rss / 1024 / 1024

    print("\n========= PERFORMANCE =========")

    print(f"Общее время обработки: {end_time - start_time:.2f} сек")

    print(f"Время поиска рекомендаций: {recommend_end - recommend_start:.4f} сек")

    print(f"Использование памяти: {memory_mb:.2f} MB")

    return render(request, "web/results.html", {"jobs": jobs_list})


def job_detail(request, job_id):
    jobs_full["id"] = jobs_full["id"].astype(int)
    job = jobs_full[jobs_full["id"] == int(job_id)]

    if job.empty:
        return render(request, "web/job_detail.html", {"job": None})

    job = job.iloc[0]

    job["description"] = format_description(job["description"])
    
    context = {
        "job": job
    }

    return render(request, "web/job_detail.html", context)