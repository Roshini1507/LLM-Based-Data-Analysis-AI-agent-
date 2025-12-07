import pandas as pd
import io
import pdfplumber
from typing import List, Union
from pdfminer.high_level import extract_text as pdf_extract_text

def parse_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def parse_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

def parse_json(file) -> pd.DataFrame:
    import json
    data = json.load(file)
    # try normalize
    try:
        return pd.json_normalize(data)
    except Exception:
        return pd.DataFrame(data)

def parse_pdf_to_text_chunks(file, max_chunk_chars=2000) -> List[str]:
    text = ""
    try:
        # pdfplumber gives good text extraction for many PDFs
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
    except Exception:
        try:
            # fallback to pdfminer
            text = pdf_extract_text(file)
        except Exception:
            text = ""
    # split into chunks roughly max_chunk_chars
    chunks = []
    cur = ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 <= max_chunk_chars:
            cur += line + "\n"
        else:
            chunks.append(cur.strip())
            cur = line + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def parse_file_to_df_or_docs(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return parse_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return parse_excel(uploaded_file)
    if name.endswith(".json"):
        return parse_json(uploaded_file)
    if name.endswith(".pdf"):
        return parse_pdf_to_text_chunks(uploaded_file)
    # default try to read as csv
    try:
        return parse_csv(uploaded_file)
    except Exception:
        return None
