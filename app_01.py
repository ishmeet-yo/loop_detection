import os
import json
import numpy as np
import torch
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

tokenizer = None
slm_model = None
embedder = None
faiss_index = None
summary_store = []

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
FAISS_PATH = "faiss_index.bin"
SUMMARIES_PATH = "summaries.json"


def load_models():
    global tokenizer, slm_model, embedder, faiss_index, summary_store

    print("[INIT] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[INIT] Tokenizer loaded")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[INIT] Loading SLM on {device}...")
    slm_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    print("[INIT] SLM loaded")

    print("[INIT] Loading sentence embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[INIT] Embedder loaded")

    print("[INIT] Setting up FAISS index...")
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        print(f"[INIT] Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        faiss_index = faiss.IndexFlatIP(EMBED_DIM)
        print("[INIT] Created new FAISS index")

    print("[INIT] Loading summary_store...")
    if os.path.exists(SUMMARIES_PATH):
        with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
            summary_store = json.load(f)
        print(f"[INIT] Loaded {len(summary_store)} summaries")
    else:
        summary_store = []
        print("[INIT] No existing summaries, starting fresh")

    print("[INIT] Models and indexes ready.")

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
    

    global tokenizer, slm_model, embedder, faiss_index, summary_store

    if tokenizer is None or slm_model is None or embedder is None or faiss_index is None:
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



def add_and_retrieve(embedding: np.ndarray, summary: str, k: int = 5):
    
    global faiss_index, summary_store

    if embedding.ndim == 1:
        emb = embedding.reshape(1, -1)
    else:
        emb = embedding
    emb = emb.astype("float32")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms

    if faiss_index.ntotal > 0:
        k_eff = min(k, faiss_index.ntotal)
        scores, idxs = faiss_index.search(emb_norm, k_eff)
        scores = scores[0]
        idxs = idxs[0]

        neighbors = []
        for s, i in zip(scores, idxs):
            i_int = int(i)
            neighbors.append({
                "id": i_int,
                "score": float(s),
                "summary": summary_store[i_int]["summary"] if 0 <= i_int < len(summary_store) else "<no stored summary>",
            })
    else:
        neighbors = []

    faiss_index.add(emb_norm)
    current_id = faiss_index.ntotal - 1

    summary_store.append({"summary": summary})
    faiss.write_index(faiss_index, FAISS_PATH)
    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_store, f, ensure_ascii=False, indent=2)

    return {
        "current_id": int(current_id),
        "neighbors": neighbors,
    }

import json
import re

def classify_usual_or_loop(summary: str, neighbors: list):
   
    if not neighbors:
        loop_decision = "usual"
        loop_strength = 0.0
    else:
        scores = [n["score"] for n in neighbors]
        max_score = max(scores)
        high_sim_neighbors = [s for s in scores if s >= 0.6]
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
  "triggers": ["..."],
  "core_belief": "...",
  "intensity": 5
}}

User summary:
\"\"\"{summary}\"\"\"

Similar past days:
\"\"\"{context_block}\"\"\"
"""

    device = slm_model.device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = slm_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print("[STEP3 RAW OUTPUT]", repr(full_text))

    cleaned= full_text.strip()
    if cleaned.startswith("'''"):
        cleaned= cleaned.split("'''",2)
        cleaned= cleaned.replace('json','',1).strip()
    parsed = None
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
        print(f"Current FAISS ID: {out2['current_id']}")
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


