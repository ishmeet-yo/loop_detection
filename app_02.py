import os
import json
import numpy as np
import torch
import requests
import uuid
import psycopg2
from psycopg2.extras import register_uuid
from sentence_transformers import SentenceTransformer

tokenizer = None
slm_model = None
embedder = None
chroma_client= None


EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
SUMMARIES_PATH = "summaries.json"


OPENROUTER_API_KEY =os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_URL= "https://openrouter.ai/api/v1/chat/completions"
LFM_MODEL_ID = "liquid/lfm-2.5-1.2b-thinking:free"

PG_DSN= "postgresql://postgres:Choco%4004@localhost:5432/loopdb"
register_uuid()


def load_models():
    global embedder

    embedder= SentenceTransformer(EMBED_MODEL)
    print('[INIT] Embedder loaded')

def safety_rule(summary: str):

    text = summary.lower()

    high_keywords = [
        "kill myself",
        "end my life",
        "end it all",
        "suicide",
        "die by suicide",
        "self-harm",
        "self harm",
        "hurt myself",
        "hurt myself on purpose",
        "i don't want to live",
        "i want to die",
    ]

    for kw in high_keywords:
        if kw in text:
            return "HIGH", False

    return "LOW", True



def input_layer(summary: str):
    

    global tokenizer, slm_model, embedder

    if tokenizer is None or slm_model is None:
        load_models()

    embedding = embedder.encode([summary])
    print(f"[INPUT] Embedding shape: {embedding.shape}")

    safety_label, is_safe = safety_rule(summary)
    print(f"[SAFETY] label={safety_label} | is_safe={is_safe}")

    result = {
        "summary": summary,
        "embedding": embedding[0],
        "safe": is_safe,
        "safety_label": safety_label,
    }
    return result

def get_pg_conn():
    return psycopg2.connect(
        host= 'localhost',
        port= 5432,
        dbname= 'loopdb',
        user='postgres',
        password='Choco@04'
    )


def add_and_retrieve(embedding: np.ndarray, summary: str, k: int = 5):
    

    if embedding.ndim > 1:
        emb = embedding[0]
    else:
        emb = embedding
    emb = emb.astype("float32")


    emb_list= emb.tolist()
    emb_str= "["+ ",".join(str(x)for x in emb_list) + "]"

    doc_id= uuid.uuid4()
    conn= get_pg_conn()
    cur= conn.cursor()

    cur.execute(
        """
    INSERT INTO daily_summaries(id,summary, embedding)
    VALUES (%s, %s, %s)
    """,
    (doc_id, summary, emb_str)

    )

    cur.execute(
    """
    SELECT id, summary, 1- (embedding <-> %s)AS similarity
    FROM daily_summaries
    ORDER BY embedding <-> %s
    LIMIT %s
    """,
    (emb_str,emb_str,k)
    )
    rows= cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()

    neighbors=[
        {
            'id': str(r[0]),
            'summary': r[1],
            'score': float(r[2])
        }
        for r in rows
    ]
    return {
        'current_id': str(doc_id),
        'neighbors':neighbors
    }

import re

def classify_usual_or_loop(summary: str, neighbors: list):
    if not neighbors:
        loop_decision = "usual"
        loop_strength = 0.0
    else:
        scores = [n["score"] for n in neighbors]
        max_score = max(scores)
        high_sim_neighbors = [s for s in scores if s >= 0.5]
        loop_decision = "loop" if len(high_sim_neighbors) >= 2 else "usual"
        loop_strength = float(max_score)

    ctx_lines = [f"- ({n['score']:.2f}) {n['summary']}" for n in neighbors[:2]]
    context_block = "\n".join(ctx_lines) if ctx_lines else "None"

    prompt = f"""You are an assistant that returns ONLY JSON.

Given the user's day summary and similar past days, extract:
- triggers: list of short situation/topic words (e.g. "work", "relationships")
- core_belief: one sentence capturing the underlying negative belief (or "" if none)
- intensity: integer 1-10 for how strong the distress seems today

Respond ONLY with a JSON object in this exact format, with no extra text:

{{
  "triggers": ["work", "relationships"],
  "core_belief": "I am not good enough.",
  "intensity": 7
}}

User summary:
\"\"\"{summary}\"\"\"

Similar past days:
\"\"\"{context_block}\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LFM_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON-only extractor. Never output anything except valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        full_text = data["choices"][0]["message"]["content"].strip()
        print("[STEP3 RAW OUTPUT]", repr(full_text))
    except Exception as e:
        print("[STEP3] OpenRouter call failed:", e)
        full_text = ""

    cleaned = full_text.strip()
    parsed = None

    if cleaned:
        try:
            parsed = json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                candidate = match.group(0)
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None

    if parsed is None:
        print("[STEP3] JSON parse failed, using defaults.")
        triggers = []
        core_belief = ""
        intensity = 5
    else:
        triggers = parsed.get("triggers", [])
        core_belief = parsed.get("core_belief", "")
        try:
            intensity = int(parsed.get("intensity", 5))
        except Exception:
            intensity = 5

    return {
        "decision": loop_decision,
        "loop_strength": loop_strength,
        "features": {
            "triggers": triggers,
            "core_belief": core_belief,
            "intensity": intensity,
        },
    }



if __name__ == "__main__":
    text = input("What's up: ")
    out1 = input_layer(text)

    print("\n=== INPUT LAYER OUTPUT ===")
    print(f"Summary: {out1['summary']}")
    print(f"Safe: {out1['safe']} (label={out1['safety_label']})")
    print(f"Embedding dim: {out1['embedding'].shape}")

    if not out1["safe"]:
        print("High Risk, crises handling ...")
    else:
        out2 = add_and_retrieve(out1["embedding"], out1["summary"], k=5)

        print("\n=== STEP 2: HISTORY & NEIGHBORS ===")
        print(f"Current ID: {out2['current_id']}")
        if not out2["neighbors"]:
            print("No past days yet (first entry or no neighbors).")
        else:
            print("Similar past days:")
            for n in out2["neighbors"]:
                print(f"- id={n['id']} | score={n['score']:.3f} | summary={n['summary']}")

        out3= classify_usual_or_loop(out1['summary'], out2["neighbors"])
        print('\n=== STEP3: CLASSIFICATION ===')
        print(f'Decision: {out3['decision']} (loop_strenth={out3['loop_strength']:.3f})')
        print(f'Features: {out3['features']}')


