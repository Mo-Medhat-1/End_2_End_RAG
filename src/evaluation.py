import pandas as pd


def recall_at_k(retrieved_docs, expected_page):
    pages = [int(d.metadata.get("page_number", -1)) for d in retrieved_docs]
    return int(int(expected_page) in pages)


def evaluate_retrieval(retriever, golden_csv: str, k: int = 5) -> pd.DataFrame:
    df = pd.read_csv(golden_csv)
    rows = []

    for _, row in df.iterrows():
        docs = retriever.invoke(row["question"])
        top_docs = docs[:k]
        hit = recall_at_k(top_docs, row["expected_page"])

        rows.append({
            "question": row["question"],
            "expected_page": row["expected_page"],
            "hit_at_k": hit,
            "retrieved_pages": [d.metadata.get("page_number") for d in top_docs],
            "retrieved_sources": [d.metadata.get("source") for d in top_docs],
        })

    return pd.DataFrame(rows)


def summarize_eval(eval_df: pd.DataFrame):
    return {
        "num_questions": len(eval_df),
        "recall_at_k": float(eval_df["hit_at_k"].mean()) if len(eval_df) else 0.0,
    }
