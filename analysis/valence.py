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
    t = summary.lower()
    neg_hits = sum(1 for w in NEGATIVE_STEMS if w in t)
    pos_hits = sum(1 for w in POSITIVE_STEMS if w in t)
    if neg_hits > pos_hits and neg_hits >= 2:
        return "negative"
    if pos_hits > neg_hits and pos_hits >= 2:
        return "positive"
    return "neutral"

def combined_valence(llm_valence: str, summary: str) -> str:
    lex = lexicon_valence(summary)
    if lex == "negative" and llm_valence == "positive":
        return "negative"
    if lex == "positive" and llm_valence == "negative":
        return "positive"
    return llm_valence if llm_valence in ("positive", "negative") else lex
