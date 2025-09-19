import os, sys
from pathlib import Path
import psycopg2

# load .env robust
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
# 1) Försök hitta .env i cwd (om du skulle byta namn senare)
found = find_dotenv(filename=".env", usecwd=True)
if found:
    load_dotenv(found)
# 2) Ladda explicit infra\infra.env
repo_root = Path(__file__).resolve().parents[1]
infra_env = repo_root / "infra" / "infra.env"
if infra_env.exists():
    load_dotenv(infra_env.as_posix(), override=True)

PG_DSN = os.getenv("PG_DSN", "postgresql://xploraforys:xploraforys@localhost:5432/xploraforys")

def check_db():
    print(f"🔌 PG_DSN = {PG_DSN}")
    try:
        conn = psycopg2.connect(PG_DSN)
        print("✅ Connected to Postgres.")
        return conn
    except Exception as e:
        print(f"❌ Cannot connect to Postgres: {e}")
        print("→ Är Docker igång? Matchar PG_DSN porten du mappar i docker-compose (t.ex. 5432 eller 5433)?")
        sys.exit(1)

def has_extension(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname=%s", (name,))
        return cur.fetchone() is not None

def table_exists(conn, t: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT to_regclass(%s)
        """, (t,))
        return cur.fetchone()[0] is not None

def count_rows(conn, t: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        return cur.fetchone()[0]

if __name__ == "__main__":
    conn = check_db()

    # vector extension
    if has_extension(conn, "vector"):
        print("✅ pgvector extension installed.")
    else:
        print("❌ pgvector missing.")
        print("→ Kör:  docker compose -f infra\\docker-compose.pgvector.yml run --rm migrate")
        sys.exit(1)

    # tables
    problems = False
    for t in ("kb_chunks", "kb_chunks_384"):
        if table_exists(conn, t):
            n = count_rows(conn, t)
            print(f"✅ {t} exists. Rows: {n}")
        else:
            print(f"❌ {t} is missing.")
            problems = True

    if problems:
        print("→ Kör migrate: docker compose -f infra\\docker-compose.pgvector.yml run --rm migrate")
        print("→ Eller kör ingestion igen: python .\\src\\ingest.py")
    else:
        print("🎯 Health check OK.")
