import os
from psycopg2.extras import register_uuid

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL_ID = "meta-llama/llama-3.2-3b-instruct:free"

PG_DSN = "postgresql://postgres:Choco%4004@localhost:5432/loopdb"

register_uuid()
