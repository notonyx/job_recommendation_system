import os
import docx
import PyPDF2
import easyocr
import fitz

reader = easyocr.Reader(['ru', 'en'])


def read_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- TXT ----------
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # ---------- DOCX ----------
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # ---------- PDF ----------
    elif ext == ".pdf":

        text = ""

        # 1) Пытаемся вытащить текст (быстро)
        try:
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

        except Exception as e:
            print("PyPDF2 error:", e)

        # 2) Если текст нормальный — возвращаем
        if len(text.strip()) > 50:
            return text

        print("PDF без текста → используем OCR")

        # 3) OCR fallback (без pdf2image!)
        try:
            doc = fitz.open(file_path)

            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = pix.tobytes("png")

                result = reader.readtext(img, detail=0)
                text += " ".join(result) + " "

        except Exception as e:
            print("OCR error:", e)

        return text.strip()

    else:
        raise ValueError("Unsupported file format")

def normalize_resume(text):
    text = text.replace("•", "")
    text = text.replace("\n", " ")
    text = text.replace(":", ". ")
    return text


def clean_resume(text):
    stop_words = [
        "обязанности",
        "опыт работы",
        "образование",
        "цель",
        "ключевые навыки"
    ]

    text = text.lower()

    for word in stop_words:
        text = text.replace(word, "")

    return text