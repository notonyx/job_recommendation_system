# import os
# import docx
# import PyPDF2
# import easyocr
# from pdf2image import convert_from_path

# reader = easyocr.Reader(['ru', 'en'])

# def read_pdf(file_path):
#     """
#     Чтение текста из PDF
#     """
#     text = ""

#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ""

#     return text


# def read_docx(file_path):
#     """
#     Чтение текста из Word (.docx)
#     """
#     doc = Document(file_path)
#     text = []

#     for para in doc.paragraphs:
#         text.append(para.text)

#     return "\n".join(text)


# def read_resume(file_path):
#     """
#     Универсальная функция
#     """
#     if file_path.endswith(".pdf"):
#         return read_pdf(file_path)

#     elif file_path.endswith(".docx"):
#         return read_docx(file_path)

#     elif file_path.endswith(".txt"):
#         with open(file_path, "r", encoding="utf-8") as f:
#             return f.read()

#     else:
#         raise ValueError("Неподдерживаемый формат файла")

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