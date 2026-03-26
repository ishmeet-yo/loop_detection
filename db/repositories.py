import uuid
import numpy as np
from .connection import get_pg_conn

def add_and_retrieve(embedding: np.ndarray, summary: str, k: int = 5):
    if embedding.ndim > 1:
        emb = embedding[0]
    else:
        emb = embedding
    emb = emb.astype("float32")

    emb_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"

    doc_id = uuid.uuid4()
    conn = get_pg_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO daily_summaries (id, summary, embedding)
        VALUES (%s, %s, %s::vector)
        """,
        (doc_id, summary, emb_str),
    )

    cur.execute(
        """
        SELECT id, summary, 1 - (embedding <-> %s::vector) AS similarity
        FROM daily_summaries
        WHERE id <> %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (emb_str, doc_id, emb_str, k),
    )
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()

    neighbors = [
        {"id": str(r[0]), "summary": r[1], "score": float(r[2])} for r in rows
    ]
    return {"current_id": str(doc_id), "neighbors": neighbors}

def update_core_belief_stats(core_belief: str, valence: str):
    if not core_belief or valence not in ("positive", "negative"):
        return None

    polarity = valence
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, occurrences
        FROM core_beliefs
        WHERE label = %s AND polarity = %s
        """,
        (core_belief, polarity),
    )
    row = cur.fetchone()

    if row:
        belief_id, occ = row
        cur.execute(
            """
            UPDATE core_beliefs
            SET occurrences = occurrences + 1,
                last_seen = now()
            WHERE id = %s
            """,
            (belief_id,),   # fixed 1‑tuple
        )
        new_occ = occ + 1
    else:
        belief_id = uuid.uuid4()
        cur.execute(
            """
            INSERT INTO core_beliefs (id, label, polarity, occurrences)
            VALUES (%s, %s, %s, 1)
            """,
            (belief_id, core_belief, polarity),
        )
        new_occ = 1

    conn.commit()
    cur.close()
    conn.close()
    return {"id": str(belief_id), "occurrences": new_occ, "polarity": polarity}
