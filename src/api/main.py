# src/api/main.py
import os
import sys
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Body, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import psycopg2
from psycopg2 import sql
# ---------------------------
# Env loading (.env + infra/)
# ---------------------------
found = find_dotenv(filename=".env", usecwd=True)
if found:
    load_dotenv(found)
infra_env = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "infra",
    "infra.env",
)
if os.path.exists(infra_env):
    load_dotenv(infra_env, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")  # optional
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")  # 1536-dim
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en-US")
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://xploraforys:xploraforys@localhost:5432/xploraforys",
)
TABLE_NAME = os.getenv("KB_TABLE")  # e.g., kb_chunks_384 / kb_chunks_1536 / kb_chunks
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
# Shared-secret auth for protected endpoints
RAG_SHARED_SECRET = os.getenv("RAG_SHARED_SECRET", "Cg9QK2J!9DgVXeTyjg&H")
# -------------
# FastAPI app
# -------------
app = FastAPI(title="Xploraforys RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------
# Auth dependency
# -------------------
def check_auth(authorization: str = Header(default="")):
    """
    Simple bearer check using RAG_SHARED_SECRET.
    Keep /healthz public so Azure probes can succeed without a token.
    """
    if not RAG_SHARED_SECRET:
        return  # auth disabled
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer")
    token = authorization.split(" ", 1)[1].strip()
    if token != RAG_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Bad token")
# -----------------------------------------
# OpenAI or local sentence-transformer init
# -----------------------------------------
client = None
local_embedder = None
use_local = False
def _init_openai():
    global client, use_local
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID or None)
            use_local = False
        except Exception as e:
            print(
                f":warning: OpenAI init failed → falling back to local embeddings: {e}",
                file=sys.stderr,
            )
            use_local = True
    else:
        use_local = True
def _init_local_embedder():
    global local_embedder
    if local_embedder is None:
        from sentence_transformers import SentenceTransformer
        local_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed(text: str) -> List[float]:
    """
    Returns a numeric vector (list[float]) for the query.
    """
    if not use_local and client:
        vec = client.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding
        return vec
    _init_local_embedder()
    v = local_embedder.encode([text], normalize_embeddings=True)[0]
    return v.astype(float).tolist()
# -----------
# DB helpers
# -----------
def to_vector_literal(vec: List[float]) -> str:
    # pgvector accepts string literal '[v1,v2,...]'
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
def get_conn():
    return psycopg2.connect(PG_DSN)
def guess_table(conn) -> str:
    """
    Choose best existing kb table based on model dims or availability.
    """
    desired = TABLE_NAME
    dim = None
    if use_local:
        desired = desired or "kb_chunks_384"
    else:
        # text-embedding-3-small → 1536, -3-large → 3072
        if "3-large" in EMB_MODEL:
            dim = 3072
            desired = desired or "kb_chunks"
        elif "3-small" in EMB_MODEL:
            dim = 1536
            desired = desired or "kb_chunks_1536"
        else:
            dim = 1536
            desired = desired or "kb_chunks_1536"
    candidates = [c for c in [desired, "kb_chunks_384", "kb_chunks_1536", "kb_chunks"] if c]
    with conn.cursor() as cur:
        for t in candidates:
            cur.execute("SELECT to_regclass(%s)", (t,))
            if cur.fetchone()[0] is not None:
                return t
    # fallback if nothing exists
    return desired or "kb_chunks_1536"
def search(conn, table: str, qvec: List[float], k: int = 6) -> List[Dict[str, Any]]:
    """
    Proper vector cast fix:
    Use %s::vector with a string literal '[...]' to avoid 'vector <=> numeric[]' errors.
    """
    vec_str = to_vector_literal(qvec)
    query = sql.SQL(
        """
        SELECT id, content, source, title,
               1 - (embedding <=> {vec}::vector) AS score
        FROM {table}
        ORDER BY embedding <=> {vec}::vector
        LIMIT %s
        """
    ).format(table=sql.Identifier(table), vec=sql.Literal(vec_str))
    with conn.cursor() as cur:
        cur.execute(query, (k,))
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append(
            {"id": r[0], "content": r[1], "source": r[2], "title": r[3], "score": float(r[4])}
        )
    return out
# -------------
# API models
# -------------
class AskIn(BaseModel):
    query: str
    k: int = 6
    lang: Optional[str] = None  # "en-US" or "sv-SE"
class AskOut(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    table: str
    used_embeddings: str
    used_chat_model: Optional[str] = None
    elapsed_ms: int
# -------------
# Endpoints
# -------------
@app.get("/healthz")
def healthz():
    """
    Public health check (no auth) so Azure probes pass.
    """
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            _ = cur.fetchone()
        conn.close()
        return {
            "ok": True,
            "pg": "up",
            "pg_dsn_present": bool(PG_DSN),
            "embeddings": ("local" if use_local else EMB_MODEL),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.post("/ask", response_model=AskOut, dependencies=[Depends(check_auth)])
def ask(payload: AskIn = Body(...)):
    t0 = time.time()
    _init_openai()
    q = payload.query.strip()
    lang = (payload.lang or DEFAULT_LANG or "en-US").lower()
    conn = get_conn()
    try:
        table = guess_table(conn)
        qvec = embed(q)
        ctx = search(conn, table, qvec, k=max(1, min(payload.k, 12)))
        # Build context string (trim to keep tokens smaller)
        def clean(s: str) -> str:
            return " ".join(s.split())
        context_blocks = []
        for i, c in enumerate(ctx, 1):
            snippet = clean(c["content"])[:1200]
            context_blocks.append(f"[{i}] ({c['title']})\n{snippet}\n")
        context = "\n\n".join(context_blocks)
        # Answer strategy
        answer_text = ""
        used_chat = None
        if not use_local and client:
            used_chat = CHAT_MODEL
            sysmsg = (
                "You are Xploraforys AI. Use the provided context. "
                "Default to American English unless the question is clearly in Swedish; "
                "be crisp, practical, and cite 1–2 source IDs like [#] at the end."
            )
            usr = (
                f"CONTEXT (top-{len(ctx)}):\n{context}\n\n"
                f"QUESTION: {q}\n\n"
                f"Answer in {'American English' if lang.startswith('en') else 'Swedish'}."
            )
            from openai import OpenAI  # safe if client already set
            # client is initialized in _init_openai
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": usr}],
                temperature=0.2,
            )
            answer_text = resp.choices[0].message.content
        else:
            # Local fallback summarizer (no LLM): returns concise bullets + sources.
            bullets = [
                "I prepared the top-K context from your knowledge base.",
                "This is a local (no-OpenAI) fallback answer.",
                "For richer summaries, set OPENAI_API_KEY and re-try.",
            ]
            answer_text = ("• " + "\n• ".join(bullets)) + (f"\n\nSources: [1]" if ctx else "\n\n(No sources)")
        elapsed = int((time.time() - t0) * 1000)
        return AskOut(
            answer=answer_text,
            sources=[{k: c[k] for k in ("id", "title", "source", "score")} for c in ctx],
            table=table,
            used_embeddings=("local:MiniLM-384" if use_local else EMB_MODEL),
            used_chat_model=used_chat,
            elapsed_ms=elapsed,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass