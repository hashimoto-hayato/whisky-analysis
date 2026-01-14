import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def cosine_similarity_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """cos(vec, mat[i]) for all i"""
    vec = vec.astype(float)
    mat = mat.astype(float)

    vnorm = np.linalg.norm(vec)
    mnorm = np.linalg.norm(mat, axis=1)

    denom = (vnorm * mnorm)
    denom[denom == 0] = np.nan

    sims = (mat @ vec) / denom
    sims = np.nan_to_num(sims, nan=-np.inf)
    return sims


def main():
    parser = argparse.ArgumentParser(description="Single-input whisky recommender (PCA3 + cosine)")
    parser.add_argument("--features", type=str, default="result/whisky_pca_features_pca3.csv",
                        help="CSV path containing name column and PC1,PC2,PC3")
    parser.add_argument("--name-col", type=str, default="Distillery",
                        help="Column name for whisky label (e.g., Distillery)")
    parser.add_argument("--query", type=str, required=True,
                        help='Query name (e.g., "Balmenach")')
    parser.add_argument("--topn", type=int, default=5, help="Top-N recommendations (default 5)")
    parser.add_argument("--out", type=str, default="result/recommend_single.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {feat_path}")

    df = pd.read_csv(feat_path)

    # Check required columns
    for c in [args.name_col, "PC1", "PC2", "PC3"]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in {feat_path}. "
                             f"Found columns: {list(df.columns)}")

    query = args.query.strip()
    if not query:
        raise ValueError("Empty --query")

    # Exact match
    qdf = df[df[args.name_col] == query].copy()
    if qdf.empty:
        # Helpful candidates
        candidates = df[df[args.name_col].str.contains(query, case=False, na=False)][args.name_col].head(10).tolist()
        raise ValueError(f"Query not found: '{query}'. Example candidates: {candidates}")

    qvec = qdf[["PC1", "PC2", "PC3"]].to_numpy()[0]

    mat = df[["PC1", "PC2", "PC3"]].to_numpy()
    sims = cosine_similarity_matrix(qvec, mat)

    out = df[[args.name_col]].copy()
    out["similarity"] = sims

    # Exclude the query itself
    out = out[out[args.name_col] != query]

    out = out.sort_values("similarity", ascending=False).head(args.topn).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("=== Single-input recommendation (PCA3 + cosine) ===")
    print("Query:", query)
    print(out.to_string(index=False))
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
