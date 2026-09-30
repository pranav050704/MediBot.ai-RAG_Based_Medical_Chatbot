import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR = Path("vectorstore")
EMBED_PATH = VECTOR_DIR / "embeddings.npy"
DOCS_PATH = VECTOR_DIR / "docs.jsonl"

class BM25Index:
    def __init__(self, docs: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_texts = [d.get("text", "") for d in docs]
        self.N = len(self.doc_texts)
        self.doc_tokens = [self._tokenize(t) for t in self.doc_texts]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / (self.N or 1)
        self.df = defaultdict(int)
        self.tfs = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            self.tfs.append(tf)
            for term in tf.keys():
                self.df[term] += 1
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = np.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _tokenize(self, text: str):
        return [t.lower() for t in text.split() if t.strip()]

    def score(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        if not q_tokens:
            return scores
        for i, tf in enumerate(self.tfs):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avg_dl or 1.0))
            s = 0.0
            for term in q_tokens:
                if term not in tf or term not in self.idf:
                    continue
                freq = tf[term]
                idf = self.idf[term]
                s += idf * (freq * (self.k1 + 1)) / (freq + denom_norm)
            scores[i] = s
        return scores

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    arr = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return arr / norms

def embed_query(text: str) -> np.ndarray:
    model = get_embedder()
    v = model.encode([text], convert_to_numpy=True)[0]
    return v / (np.linalg.norm(v) + 1e-9)

def load_vectorstore() -> Tuple[np.ndarray, List[Dict]]:
    """
    Load embeddings.npy and docs.jsonl and return (embeddings_normalized, docs_list).
    Raises FileNotFoundError if missing.
    """
    if not EMBED_PATH.exists() or not DOCS_PATH.exists():
        raise FileNotFoundError("Vectorstore not found. Run create_memory_for_llm.py first.")

    embeddings = np.load(str(EMBED_PATH))
    docs = []
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms
    return embeddings, docs

def load_vectorstore_and_indexes():
    """
    Convenience loader that returns (embeddings, docs, bm25_index) or (None, None, None) if files missing.
    Useful for Streamlit caching.
    """
    if not EMBED_PATH.exists() or not DOCS_PATH.exists():
        return None, None, None
    embeddings, docs = load_vectorstore()
    bm25 = BM25Index(docs)
    return embeddings, docs, bm25

def retrieve_documents(
    query: str,
    embeddings: np.ndarray,
    docs: List[Dict],
    bm25: BM25Index = None,
    k: int = 4,
    alpha: float = 0.6,
):
    """
    Hybrid retrieval combining BM25 (if available) and semantic similarity.
    Returns list of dicts: {score, doc, cosine, bm25}
    """
    if embeddings is None or docs is None:
        return []

    qv = embed_query(query)
    cos_sims = embeddings.dot(qv)

    bm25_scores = bm25.score(query) if bm25 is not None else np.zeros_like(cos_sims)

    def norm(arr):
        arr = arr.astype(np.float32)
        minv, maxv = float(arr.min()), float(arr.max())
        if maxv - minv < 1e-9:
            return np.zeros_like(arr)
        return (arr - minv) / (maxv - minv)

    combined = alpha * norm(bm25_scores) + (1 - alpha) * norm(cos_sims)
    idx = np.argsort(-combined)[:k]
    results = []
    for i in idx:
        results.append(
            {
                "score": float(combined[i]),
                "doc": docs[i],
                "cosine": float(cos_sims[i]),
                "bm25": float(bm25_scores[i]) if bm25 is not None else 0.0,
            }
        )
    return results

def build_context_from_results(results):
    context_parts = []
    sources = []
    for r in results:
        d = r["doc"]
        source = d.get("source", "unknown")
        page = d.get("page", "?")
        text = d.get("text", "")
        snippet = text[:600].replace("\n", " ").strip()
        context_parts.append(f"[{source} — page {page}]\n{snippet}")
        sources.append(f"{source} (page {page}) — score={r['score']:.3f}")
    context = "\n\n".join(context_parts)
    return context, sources

def extract_text_from_pdf(file_obj) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(file_obj)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(t)
    return "\n\n".join(texts)

def extract_text_from_image(file_obj) -> str:
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError("OCR (pytesseract + pillow) not available. Install them to use image analysis.") from e
    image = Image.open(file_obj)
    text = pytesseract.image_to_string(image)
    return text

def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 600) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment or .env file.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful medical assistant. Use only the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print("connect_memory_with_llm.py module. Import functions from this file in medibot.py")
