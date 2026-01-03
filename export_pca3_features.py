import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


DEFAULT_FEATURES = [
    "Body", "Sweetness", "Smoky", "Medicinal", "Tobacco", "Honey",
    "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"
]

def main():
    parser = argparse.ArgumentParser(description="Export PCA feature CSV (PC1..PCk) for recommendation.")
    parser.add_argument("--in", dest="in_path", type=str, default="whisky_clean.csv",
                        help="Input CSV path (default: whisky_clean.csv)")
    parser.add_argument("--out", dest="out_path", type=str, default="result/whisky_pca_features_pca3.csv",
                        help="Output CSV path (default: result/whisky_pca_features_pca3.csv)")
    parser.add_argument("--name-col", type=str, default="Distillery",
                        help="Name column (default: Distillery). If not found, fallback to 'Name' or first column.")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES),
                        help="Comma-separated feature columns (default: 12 flavor columns)")
    parser.add_argument("--components", type=int, default=3, help="Number of PCA components (default: 3)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # name column fallback
    name_col = args.name_col
    if name_col not in df.columns:
        if "Name" in df.columns:
            name_col = "Name"
        else:
            name_col = df.columns[0]  # 最悪これで動かす

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature columns not found: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"→ --features で列名を合わせて再実行してください"
        )

    # drop rows with NaN in features
    work = df[[name_col] + feature_cols].dropna().copy()

    X = work[feature_cols].to_numpy(dtype=float)

    # Min-Max normalize
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=args.components, random_state=0)
    Z = pca.fit_transform(Xs)

    # output
    out = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(args.components)])
    out.insert(0, name_col, work[name_col].values)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # print summary
    evr = pca.explained_variance_ratio_
    print("Explained variance ratio:")
    for i, r in enumerate(evr, 1):
        print(f"PC{i}: {r:.3f}")
    print(f"Cumulative (PC1-{args.components}): {evr.sum():.3f}")
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
