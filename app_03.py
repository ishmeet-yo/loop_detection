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
    

    global  embedder

    if embedder is None:
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
    VALUES (%s, %s, %s::vector)
    """,
    (doc_id, summary, emb_str)

    )

    cur.execute(
    """
    SELECT id, summary, 1- (embedding <-> %s::vector)AS similarity
    FROM daily_summaries
    WHERE id <> %s
    ORDER BY embedding <-> %s::vector
    LIMIT %s
    """,
    (emb_str, doc_id,emb_str,k)
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

NEGATIVE_STEMS = [
    "sad", "depress", "anxious", "worry", "afraid", "scared",
    "overwhelm", "stressed", "angry", "frustrat", "hopeless",
    "worthless", "hate", "tired", "exhausted", "lonely",
]

POSITIVE_STEMS = [
    "happy", "glad", "joy", "excited", "proud",
    "grateful", "gratef", "satisfied", "content", "peaceful",
    "chill", "relaxed", "good day", "great day",
]

def lexicon_valence(summary: str) -> str:
    t= summary.lower()
    neg_hits= sum(1 for w in NEGATIVE_STEMS if w in t)
    pos_hits= sum(1 for w in POSITIVE_STEMS if w in t)
    if neg_hits> pos_hits and neg_hits >=2:
        return 'negative'
    if pos_hits> neg_hits and pos_hits>=2:
        return 'positive'
    return 'neutral'

def combined_valence(llm_valence: str, summary: str) ->str:
    lex= lexicon_valence(summary)
    if lex == 'negative' and llm_valence == "positive":
        return "negative"
    if lex == "positive" and llm_valence == "negative":
        return "positive"
    return llm_valence if llm_valence in ("positive", "negative") else lex

def update_core_belief_stats(core_belief:str, valence: str):
    if not core_belief:
        return None
    if valence not in ('positive', 'negative'):
        return None
    polarity= valence
    conn= get_pg_conn()
    cur= conn.cursor()
    cur.execute(
        """
    SELECT id,occurrences
    FROM  core_beliefs
    WHERE label= %s AND polarity= %s
""",
(core_belief, polarity)
    )
    row= cur.fetchone()

    if row:
        belief_id, occ= row
        cur.execute(
            """
    UPDATE core_beliefs
    SET occurrences= occurrences +1,
    last_seen= now()
    WHERE id= %s
""",
    (belief_id,),
        )
        new_occ= occ +1
    else:
        belief_id= uuid.uuid4()
        cur.execute(
            """
    INSERT INTO core_beliefs (id, label, polarity, occurrences)
    VALUES (%s, %s, %s,1)    
""",
    (belief_id, core_belief, polarity)
        )
        new_occ=1
    conn.commit()
    cur.close()
    conn.close()
    return {'id': str(belief_id), 'occurrences': new_occ, 'polarity': polarity}

import re

def classify_usual_or_loop(summary: str, neighbors: list):
    if not neighbors:
        loop_decision = "usual"
        loop_strength = 0.0
    else:
        scores = [n["score"] for n in neighbors]
        max_score = max(scores)
        high_sim_neighbors = [s for s in scores if s >= 0.65]
        loop_decision = "loop" if len(high_sim_neighbors) >= 2 else "usual"
        loop_strength = float(max_score)

    ctx_lines = [f"- ({n['score']:.2f}) {n['summary']}" for n in neighbors[:2]]
    context_block = "\n".join(ctx_lines) if ctx_lines else "None"

    prompt = f"""You are an assistant that returns ONLY JSON.

You analyze a single day's summary and a few similar past days.

Rules:
- Determine if today's summary an overall valence as positive, neutral, or negative.
- Only infer a core_belief if today's text and the neighbors clearly have a strong positive or negative emotion that could be converted to a singluar sentence or phrase.
- triggers are short topic words like "heavy work", "relationships", "health", "friends", but focus on the core reason, between all the texts.

Respond ONLY with a JSON object in this exact format, with no extra text:

{{
  "triggers": ["heavy work", "relationships"],
  "core_belief": "I'm incompetent",
  "intensity": 7,
  "valence": "negative",        // one of "positive", "neutral", "negative"
  "emotion": "sadness"          // e.g. "sadness", "anger", "anxiety", "joy", "calm"
}}
or
{{
  "triggers": ["good food", "workout"],
  "core_belief": "I'm healthy",
  "intensity": 7,
  "valence": "positive",        // one of "positive", "neutral", "negative"
  "emotion": "joy"          // e.g. "sadness", "anger", "anxiety", "joy", "calm"
}}
stuff like that, but only focus on the current and past similar days.
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
        valence_llm= 'neutral'
        emotion= ''
    else:
        triggers = parsed.get("triggers", [])
        core_belief = parsed.get("core_belief", "")
        try:
            intensity = int(parsed.get("intensity", 5))
        except Exception:
            intensity = 5
        valence_llm= parsed.get('valence', 'neutral')
        emotion= parsed.get('emotion', '')
    
    valence= combined_valence(valence_llm , summary)

    return {
        "decision": loop_decision,
        "loop_strength": loop_strength,
        "features": {
            "triggers": triggers,
            "core_belief": core_belief,
            "intensity": intensity,
            "valence": valence,
            "emotion": emotion
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

        out3 = classify_usual_or_loop(out1["summary"], out2["neighbors"])

        print("\n=== STEP3: CLASSIFICATION ===")
        print(f"Decision: {out3['decision']} (loop_strength={out3['loop_strength']:.3f})")
        print(f"Features: {out3['features']}")

        belief_info = update_core_belief_stats(
            out3["features"]["core_belief"],
            out3["features"]["valence"],
        )
        if belief_info and belief_info["occurrences"] >= 5:
            print(
                f"\n>>> STABLE {belief_info['polarity'].upper()} CORE BELIEF DETECTED: "
                f"'{out3['features']['core_belief']}' "
                f"(seen {belief_info['occurrences']} times)"
            ) 