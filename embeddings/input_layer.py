import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL

embedder = None

def load_models():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
        print("[INIT] Embedder loaded")

def safety_rule(summary: str):
    text = summary.lower()
    high_keywords = [
        "kill myself", "end my life", "end it all", "suicide",
        "die by suicide", "self-harm", "self harm", "hurt myself",
        "hurt myself on purpose", "i don't want to live", "i want to die",
    ]
    for kw in high_keywords:
        if kw in text:
            return "HIGH", False
    return "LOW", True

def input_layer(summary: str):
    global embedder
    if embedder is None:
        load_models()
    embedding = embedder.encode([summary])
    safety_label, is_safe = safety_rule(summary)
    return {
        "summary": summary,
        "embedding": embedding[0],
        "safe": is_safe,
        "safety_label": safety_label,
    }
