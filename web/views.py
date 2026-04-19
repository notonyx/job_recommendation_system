from django.shortcuts import render, redirect
import pandas as pd
import os
from src.models.bert_faiss_model import JobRecommenderBERTFAISS
from src.utils.resume_reader import read_resume, normalize_resume, clean_resume
from src.utils.text_preprocessing import clean_text
from .forms import ResumeUploadForm

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


# def upload_resume(request):
#     if request.method == 'POST':
#         form = ResumeUploadForm(request.POST, request.FILES)

#         if form.is_valid():
#             file = request.FILES['file']

#             path = f"temp_{file.name}"
#             with open(path, 'wb+') as destination:
#                 for chunk in file.chunks():
#                     destination.write(chunk)

#             text = read_resume(path)
#             os.remove(path)
#             request.session['resume_text'] = text

#             return redirect('results')

#     else:
#         form = ResumeUploadForm()

#     return render(request, 'web/upload.html', {'form': form})

def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)

        if form.is_valid():
            file = request.FILES.get('file')   # ✅ было []
            text_input = form.cleaned_data.get("text")  # ✅ новое

            # 👉 если пользователь вставил текст — используем его
            if text_input:
                text = text_input

            # 👉 иначе работаем как раньше с файлом
            elif file:
                path = f"temp_{file.name}"
                with open(path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                text = read_resume(path)
                os.remove(path)

            else:
                # защита (хотя форма уже валидирует)
                return render(request, 'web/upload.html', {'form': form})

            request.session['resume_text'] = text
            return redirect('results')

    else:
        form = ResumeUploadForm()

    return render(request, 'web/upload.html', {'form': form})


def results(request):
    resume_text = request.session.get("resume_text", "")

    resume_text = normalize_resume(resume_text)
    resume_text = clean_resume(resume_text)
    resume_text = clean_text(resume_text)

    model = get_recommender()
    jobs = model.recommend(resume_text)

    print(jobs.head())
    print(jobs.columns)

    jobs["similarity_percent"] = jobs["similarity"] * 100
    jobs_list = jobs.to_dict(orient="records")

    # ----------------------------
    for job in jobs_list:
        job_id = int(job.get("id"))

        full_row = jobs_full[jobs_full["id"] == job_id]

        if not full_row.empty:
            full_row = full_row.iloc[0]

            job["description"] = full_row.get("description")
            job["salary"] = full_row.get("salary")
            job["experience"] = full_row.get("experience")
            job["job_type"] = full_row.get("job_type")

    return render(request, "web/results.html", {"jobs": jobs_list})

def job_detail(request, job_id):
    jobs_full["id"] = jobs_full["id"].astype(int)
    job = jobs_full[jobs_full["id"] == int(job_id)]

    if job.empty:
        return render(request, "web/job_detail.html", {"job": None})

    job = job.iloc[0]

    context = {
        "job": job
    }

    return render(request, "web/job_detail.html", context)