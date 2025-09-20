# src/api/main.py
import os, time, sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import psycopg2
from psycopg2 import sql
# ---------- basic app first (so decorators see it) ----------
app = FastAPI(title="Xploraforys RAG API", version="1.0.0")
# ---------- CORS ----------
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- env (.env + infra/infra.env) ----------
found = find_dotenv(filename=".env", usecwd=True)
if found:
    load_dotenv(found)
infra_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "infra", "infra.env")
if os.path.exists(infra_env):
    load_dotenv(infra_env, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID  = os.getenv("OPENAI_ORG_ID")
EMB_MODEL      = os.getenv("EMB_MODEL", "text-embedding-3-small")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_LANG   = os.getenv("DEFAULT_LANG", "en-US")
PG_DSN         = os.getenv("PG_DSN", "postgresql://xploraforys:xploraforys@localhost:5432/xploraforys")
TABLE_NAME     = os.getenv("KB_TABLE")
RAG_SHARED_SECRET = os.getenv("RAG_SHARED_SECRET", "")
# ---------- auth dep (NOT applied to /healthz) ----------
def check_auth(authorization: str = Header(default="")):
    if not RAG_SHARED_SECRET:
        return
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer")
    token = authorization.split(" ", 1)[1].strip()
    if token != RAG_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Bad token")
# ---------- embeddings (OpenAI only; no local fallback to keep image light) ----------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID or None) if OPENAI_API_KEY else None
def embed(text: str) -> List[float]:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return client.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding
# ---------- DB helpers ----------
def get_conn():
    return psycopg2.connect(PG_DSN)
def to_vector_literal(vec: List[float]) -> str:
    return "[" + ","join(f"{float(x):.6f}" for x in vec) + "]"
def guess_table(conn) -> str:
    desired = TABLE_NAME or "kb_chunks_1536"
    candidates = [desired, "kb_chunks_1536", "kb_chunks_384", "kb_chunks"]
    with conn.cursor() as cur:
        for t in candidates:
            cur.execute("SELECT to_regclass(%s)", (t,))
            if cur.fetchone()[0] is not None:
                return t
    return desired
def search(conn, table: str, qvec: List[float], k: int = 6) -> List[Dict[str, Any]]:
    vec_str = to_vector_literal(qvec)
    query = sql.SQL("""
        SELECT id, content, source, title, 1 - (embedding <=> {v}::vector) AS score
        FROM {t}
        ORDER BY embedding <=> {v}::vector
        LIMIT %s
    """).format(t=sql.Identifier(table), v=sql.Literal(vec_str))
    with conn.cursor() as cur:
        cur.execute(query, (k,))
        rows = cur.fetchall()
    return [{"id":r[0],"content":r[1],"source":r[2],"title":r[3],"score":float(r[4])} for r in rows]
# ---------- models ----------
class AskIn(BaseModel):
    query: str
    k: int = 6
    lang: Optional[str] = None
class AskOut(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    table: str
    used_embeddings: str
    used_chat_model: Optional[str]
    elapsed_ms: int
# ---------- endpoints ----------
@app.get("/healthz")
def healthz():
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()
        return {"ok": True, "pg": "up", "pg_dsn_present": bool(PG_DSN)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.post("/ask", response_model=AskOut, dependencies=[Depends(check_auth)])
def ask(payload: AskIn = Body(...)):
    if not client:
        raise HTTPException(500, "OPENAI_API_KEY not set on server.")
    t0 = time.time()
    q = payload.query.strip()
    conn = get_conn()
    try:
        table = guess_table(conn)
        qvec = embed(q)
        ctx = search(conn, table, qvec, k=max(1, min(payload.k, 12)))
        # small summary with chat model
        sysmsg = "Answer using the provided context. Cite 1–2 source IDs like [#]. Be concise."
        blocks = []
        for i,c in enumerate(ctx,1):
            snippet = " ".join(c["content"].split())[:1200]
            blocks.append(f"[{i}] ({c['title']})\n{snippet}\n")
        usr = f"CONTEXT:\n{'\n'.join(blocks)}\n\nQUESTION: {q}"
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        ms = int((time.time()-t0)*1000)
        return AskOut(
            answer=answer,
            sources=[{k:c[k] for k in ("id","title","source","score")} for c in ctx],
            table=table,
            used_embeddings=EMB_MODEL,
            used_chat_model=CHAT_MODEL,
            elapsed_ms=ms,
        )
    finally:
        try: conn.close()
        except: pass