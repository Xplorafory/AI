# src/rag_query.py
import os, sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv, find_dotenv
from typing import List
import numpy as np

# Load env from .env and infra\infra.env
found = find_dotenv(filename=".env", usecwd=True)
if found:
    load_dotenv(found)
repo_root = Path(__file__).resolve().parents[1]
infra_env = repo_root / "infra" / "infra.env"
if infra_env.exists():
    load_dotenv(infra_env.as_posix(), override=True)

PG_DSN    = os.getenv("PG_DSN", "postgresql://xploraforys:xploraforys@localhost:5432/xploraforys")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "local-MiniLM-384":       384,
}

def dim_for_model(m: str, have_key: bool) -> int:
    if have_key: return MODEL_DIMS.get(m, 1536)
    return 384

def table_for_dim(dim: int) -> str:
    return f"kb_chunks_{dim}"

def embed_local(text: str) -> List[float]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = model.encode([text])[0].tolist()
    return v

def vector_literal(vec: List[float]) -> str:
    # Format as pgvector literal: [v1,v2,...]
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def retrieve(query: str, k: int = 5) -> List[dict]:
    have_key = bool(OPENAI_API_KEY)
    dim = dim_for_model(EMB_MODEL, have_key)
    table = table_for_dim(dim)

    if have_key:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        emb = client.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    else:
        print("ℹ️ Falling back to local embeddings: No OPENAI_API_KEY set")
        emb = embed_local(query)

    lit = vector_literal(emb)

    conn = psycopg2.connect(PG_DSN)
    with conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, content, source, title,
                   1 - (embedding <=> %s::vector) AS score
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (lit, lit, k)
        )
        rows = cur.fetchall()
    return [{"id": r[0], "content": r[1], "source": r[2], "title": r[3], "score": float(r[4])} for r in rows]

def answer(q: str, k: int = 5) -> str:
    ctx = retrieve(q, k=k)
    print("\n--- ANSWER ---")
    print("(Local mode) Top-5 context prepared." if not OPENAI_API_KEY else "Top-5 context prepared.")
    print("\nCONTEXT (top-5):")
    for i, r in enumerate(ctx, 1):
        snippet = (r["content"] or "")[:700]
        print(f"[{i}] ({r['title']})\n\n{snippet}\n")
    return "Done."

if __name__ == "__main__":
    print(f"✅ Using {'OpenAI' if OPENAI_API_KEY else 'local'} embeddings ({dim_for_model(EMB_MODEL, bool(OPENAI_API_KEY))}). Table: {table_for_dim(dim_for_model(EMB_MODEL, bool(OPENAI_API_KEY)))}")
    print(f"DB: {PG_DSN}")
    q = "Test prompt: what’s inside my docs?"
    print(answer(q, k=5))
