from embeddings.input_layer import input_layer
from db.repositories import add_and_retrieve, update_core_belief_stats
from analysis.classifier import classify_usual_or_loop

if __name__ == "__main__":
    text = input("What's up: ")
    out1 = input_layer(text)

    print("\n INPUT LAYER OUTPUT ")
    print(f"Summary: {out1['summary']}")
    print(f"Safe: {out1['safe']} (label={out1['safety_label']})")
    print(f"Embedding dim: {out1['embedding'].shape}")

    if not out1["safe"]:
        print("High Risk, crises handling ...")
    else:
        out2 = add_and_retrieve(out1["embedding"], out1["summary"], k=5)

        print("\n STEP 2: HISTORY & NEIGHBORS ")
        print(f"Current ID: {out2['current_id']}")
        if not out2["neighbors"]:
            print("No past days yet (first entry or no neighbors).")
        else:
            print("Similar past days:")
            for n in out2["neighbors"]:
                print(f"- id={n['id']} | score={n['score']:.3f} | summary={n['summary']}")

        out3 = classify_usual_or_loop(out1["summary"], out2["neighbors"])

        print("\n STEP3: CLASSIFICATION ")
        print(f"Decision: {out3['decision']} (loop_strength={out3['loop_strength']:.3f})")
        print(f"Features: {out3['features']}")

        belief_info = update_core_belief_stats(
            out3["features"]["core_belief"],
            out3["features"]["valence"],
        )
        if belief_info and belief_info["occurrences"] >= 5:
            print(
                f"\nSTABLE {belief_info['polarity'].upper()} CORE BELIEF DETECTED: "
                f"'{out3['features']['core_belief']}' "
                f"(seen {belief_info['occurrences']} times)"
            )
