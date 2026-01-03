import argparse
import pathlib
import numpy as np
import pandas as pd

DEFAULT_FEATURES_CSV = "result/whisky_pca_features_pca3.csv"
DEFAULT_NAME_COL = "Distillery"
DEFAULT_PCA_COLS = ["PC1", "PC2", "PC3"]
DEFAULT_TOPN = 5


def _resolve_cols_case_insensitive(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    lower_map = {c.lower(): c for c in df.columns}
    if col.lower() in lower_map:
        return lower_map[col.lower()]
    raise KeyError(f"Required column '{col}' not found. Found: {list(df.columns)}")


def load_features(path: str, name_col: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {p}\n"
            f"Generate PCA3 features first (PC1-3)."
        )

    df = pd.read_csv(p)

    name_col = _resolve_cols_case_insensitive(df, name_col)
    pc1 = _resolve_cols_case_insensitive(df, "PC1")
    pc2 = _resolve_cols_case_insensitive(df, "PC2")
    pc3 = _resolve_cols_case_insensitive(df, "PC3")
    pca_cols = [pc1, pc2, pc3]

    out = df[[name_col] + pca_cols].copy()
    out = out.dropna(subset=[name_col])

    for c in pca_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=pca_cols)

    out["_name_norm"] = out[name_col].astype(str).str.strip()
    out["_name_norm_lc"] = out["_name_norm"].str.lower()

    return out, name_col, pca_cols


def cosine_recommend_single(df: pd.DataFrame, query_name: str, name_col: str, pca_cols, topn: int):
    q = query_name.strip().lower()
    hit = df[df["_name_norm_lc"] == q]
    if len(hit) == 0:
        examples = df["_name_norm"].dropna().unique().tolist()[:10]
        raise ValueError(
            f"Query '{query_name}' not found in {name_col}.\n"
            f"Example names: {examples}"
        )

    qvec = hit.iloc[0][pca_cols].to_numpy(dtype=float)
    qnorm = np.linalg.norm(qvec)
    if qnorm == 0:
        raise ValueError("Query vector norm is zero; cannot compute cosine similarity.")
    qvec = qvec / qnorm

    X = df[pca_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms[:, None]

    sims = Xn @ qvec

    out = df.copy()
    out["similarity"] = sims

    # exclude the query itself
    out = out[out["_name_norm_lc"] != q]

    out = out.sort_values("similarity", ascending=False).head(topn).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Single-input whisky recommendation (PCA3 + cosine)."
    )
    parser.add_argument("--features", default=DEFAULT_FEATURES_CSV,
                        help="CSV with Distillery and PC1-3 (precomputed).")
    parser.add_argument("--name-col", default=DEFAULT_NAME_COL,
                        help="Name column (default: Distillery).")
    parser.add_argument("--query", required=True,
                        help='Single Distillery name. e.g. "Aberfeldy"')
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN,
                        help="Top-N recommendations (default: 5).")
    parser.add_argument("--out", default="result/recommend_single.csv",
                        help="Output CSV path (includes similarity; for evaluation).")
    parser.add_argument("--show-score", action="store_true",
                        help="Print similarity scores to console (default: hide).")

    args = parser.parse_args()

    df, name_col, pca_cols = load_features(args.features, args.name_col)
    rec = cosine_recommend_single(df, args.query, name_col, pca_cols, args.topn)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec_out = rec[["rank", name_col, "similarity"]].rename(columns={name_col: "Distillery"})
    rec_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== Recommendation (single input, PCA3 + cosine) ===")
    print("Query:", args.query)
    if args.show_score:
        print(rec_out.to_string(index=False))
    else:
        print(rec_out[["rank", "Distillery"]].to_string(index=False))
    print(f"✅ Saved (for evaluation): {out_path}")


if __name__ == "__main__":
    main()
