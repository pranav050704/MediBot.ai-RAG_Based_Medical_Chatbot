import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DATA_DIR = Path("data")
OUT_DIR = Path("vectorstore")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def read_pdfs(data_dir: Path) -> List[Dict]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}")
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}. Put PDFs there and re-run.")
    docs = []
    for p in pdfs:
        try:
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    docs.append({"source": p.name, "page": i + 1, "text": text})
        except Exception as e:
            print(f"[!] Warning: couldn't read {p.name}: {e}")
    print(f"[+] Loaded {len(docs)} PDF pages.")
    return docs

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

def docs_to_chunks(docs: List[Dict]) -> List[Dict]:
    out = []
    for d in docs:
        text = d["text"]
        chunks = chunk_text(text)
        for c in chunks:
            out.append({"source": d["source"], "page": d["page"], "text": c})
    print(f"[+] Created {len(out)} chunks from {len(docs)} pages.")
    return out

def embed_texts(texts: List[str], model_name=EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_vectorstore(chunks: List[Dict], embeddings: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "docs.jsonl"
    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[+] Saved embeddings to {emb_path}")
    print(f"[+] Saved docs metadata to {meta_path}")

def main():
    docs = read_pdfs(DATA_DIR)
    chunks = docs_to_chunks(docs)
    if not chunks:
        print("[!] No chunks produced — exiting.")
        return
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    save_vectorstore(chunks, embeddings, OUT_DIR)
    print("[+] Vectorstore build complete.")

if __name__ == "__main__":
    main()
