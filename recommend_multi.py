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

    # Avoid division by zero
    denom = (vnorm * mnorm)
    denom[denom == 0] = np.nan

    sims = (mat @ vec) / denom
    # nan -> -inf so they go to bottom
    sims = np.nan_to_num(sims, nan=-np.inf)
    return sims


def main():
    parser = argparse.ArgumentParser(description="Multi-input whisky recommender (mean vector, PCA3 + cosine)")
    parser.add_argument("--features", type=str, default="result/whisky_pca_features_pca3.csv",
                        help="CSV path containing name column and PC1,PC2,PC3")
    parser.add_argument("--name-col", type=str, default="Distillery",
                        help="Column name for whisky label (e.g., Distillery)")
    parser.add_argument("--queries", type=str, required=True,
                        help='Comma-separated query names (e.g., "Aberfeldy,Balmenach")')
    parser.add_argument("--topn", type=int, default=5, help="Top-N recommendations (default 5)")
    parser.add_argument("--out", type=str, default="result/recommend_multi.csv",
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

    # Parse query list
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    if len(queries) == 0:
        raise ValueError("No queries provided.")

    # Match queries (exact match)
    qdf = df[df[args.name_col].isin(queries)].copy()
    missing = sorted(set(queries) - set(qdf[args.name_col].tolist()))
    if missing:
        # Provide helpful candidates
        candidates = df[df[args.name_col].str.contains(missing[0], case=False, na=False)][args.name_col].head(10).tolist()
        raise ValueError(f"Query not found: {missing}. Example candidates for '{missing[0]}': {candidates}")

    # Mean preference vector
    pref = qdf[["PC1", "PC2", "PC3"]].to_numpy().mean(axis=0)

    # Similarities to all
    mat = df[["PC1", "PC2", "PC3"]].to_numpy()
    sims = cosine_similarity_matrix(pref, mat)

    out = df[[args.name_col]].copy()
    out["similarity"] = sims

    # Exclude query items
    out = out[~out[args.name_col].isin(queries)]

    # Sort and take topn
    out = out.sort_values("similarity", ascending=False).head(args.topn).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("=== Multi-input recommendation (mean vector, PCA3 + cosine) ===")
    print("Queries:", ", ".join(queries))
    print(out.to_string(index=False))
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
