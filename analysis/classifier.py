import json
import re
import requests

from config import OPENROUTER_API_KEY, OPENROUTER_URL, GEN_MODEL_ID
from analysis.valence import combined_valence

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

...same prompt text...
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
        "model": GEN_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON-only extractor. Never output anything except valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
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
        triggers = []
        core_belief = ""
        intensity = 5
        valence_llm = "neutral"
        emotion = ""
    else:
        triggers = parsed.get("triggers", [])
        core_belief = parsed.get("core_belief", "")
        try:
            intensity = int(parsed.get("intensity", 5))
        except Exception:
            intensity = 5
        valence_llm = parsed.get("valence", "neutral")
        emotion = parsed.get("emotion", "")

    valence = combined_valence(valence_llm, summary)

    return {
        "decision": loop_decision,
        "loop_strength": loop_strength,
        "features": {
            "triggers": triggers,
            "core_belief": core_belief,
            "intensity": intensity,
            "valence": valence,
            "emotion": emotion,
        },
    }
