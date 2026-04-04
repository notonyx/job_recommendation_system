from django.shortcuts import render, redirect
import pandas as pd

from src.models.bert_faiss_model import JobRecommenderBERTFAISS
from src.utils.resume_reader import read_resume
from .forms import ResumeUploadForm

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
            file = request.FILES['file']

            path = f"temp_{file.name}"
            with open(path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            text = read_resume(path)
            request.session['resume_text'] = text

            return redirect('results')

    else:
        form = ResumeUploadForm()

    return render(request, 'web/upload.html', {'form': form})


def results(request):
    resume_text = request.session.get("resume_text", "")

    model = get_recommender()
    jobs = model.recommend(resume_text)

    print(jobs.head())
    print(jobs.columns)

    jobs_list = jobs.to_dict(orient="records")

    return render(request, "web/results.html", {"jobs": jobs_list})