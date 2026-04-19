# from django import forms

# class ResumeUploadForm(forms.Form):
#     file = forms.FileField()

from django import forms

class ResumeUploadForm(forms.Form):
    file = forms.FileField(required=False)
    text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            "placeholder": "Или вставьте текст резюме сюда...",
            "rows": 8,
            "class": "resume-textarea"
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get("file")
        text = cleaned_data.get("text")

        if not file and not text:
            raise forms.ValidationError("Загрузите файл или вставьте текст резюме")

        return cleaned_data