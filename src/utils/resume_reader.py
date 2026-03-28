import pdfplumber
from docx import Document


def read_pdf(file_path):
    """
    Чтение текста из PDF
    """
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


def read_docx(file_path):
    """
    Чтение текста из Word (.docx)
    """
    doc = Document(file_path)
    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return "\n".join(text)


def read_resume(file_path):
    """
    Универсальная функция
    """
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)

    elif file_path.endswith(".docx"):
        return read_docx(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Неподдерживаемый формат файла")