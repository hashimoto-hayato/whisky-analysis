import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def pairwise_euclid(A: np.ndarray) -> np.ndarray:
    G = (A * A).sum(axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (A @ A.T)
    D2[D2 < 0] = 0
    return np.sqrt(D2)

def spearman_from_dist(Da: np.ndarray, Db: np.ndarray) -> float:
    iu = np.triu_indices_from(Da, k=1)
    a = Da[iu]
    b = Db[iu]
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = (ra - ra.mean()) / ra.std(ddof=0)
    rb = (rb - rb.mean()) / rb.std(ddof=0)
    return float((ra * rb).mean())

def knn_indices(D: np.ndarray, k: int):
    idx = np.argsort(D, axis=1)
    out = []
    for i in range(D.shape[0]):
        neigh = idx[i][idx[i] != i][:k]  # 自分自身除外
        out.append(neigh)
    return out

def cosine_topn(A: np.ndarray, topn: int):
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    norm[norm == 0] = 1
    An = A / norm
    S = An @ An.T
    top = []
    for i in range(S.shape[0]):
        order = np.argsort(-S[i])
        order = order[order != i][:topn]
        top.append(order)
    return top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="例: whisky_clean.csv")
    ap.add_argument("--key", default="Distillery", help="銘柄名列")
    ap.add_argument("--k", type=int, default=10, help="近傍保存率のk")
    ap.add_argument("--topn", type=int, default=5, help="Jaccard@topn の topn")
    ap.add_argument("--out", default="result/eval_metrics_ch5.json", help="結果JSON出力先")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.key not in df.columns:
        raise ValueError(f"key列 '{args.key}' がCSVにありません。--key を見直して。")

    feat_cols = [c for c in df.columns if c != args.key]
    X = df[feat_cols].to_numpy(dtype=float)

    # 前処理：Min--Max 正規化
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    # PCA（3次元をfitして、2次元は先頭2軸）
    pca = PCA(n_components=3)
    Z3 = pca.fit_transform(Xs)
    Z2 = Z3[:, :2]

    # 累積寄与率
    evr = pca.explained_variance_ratio_
    cum1 = float(evr[:1].sum())
    cum2 = float(evr[:2].sum())
    cum3 = float(evr[:3].sum())

    # Spearman：元空間距離 vs 2D距離
    D_orig = pairwise_euclid(Xs)
    D_map2 = pairwise_euclid(Z2)
    rho = spearman_from_dist(D_orig, D_map2)

    # 近傍保存率（k）
    knn_o = knn_indices(D_orig, args.k)
    knn_m = knn_indices(D_map2, args.k)
    Pk = [len(set(knn_o[i]) & set(knn_m[i])) / args.k for i in range(len(Xs))]
    Pk_mean = float(np.mean(Pk))
    Pk_std = float(np.std(Pk, ddof=0))

    # Jaccard@topn（元空間cosine TopN vs PCA3 cosine TopN）
    top_o = cosine_topn(Xs, args.topn)
    top_3 = cosine_topn(Z3, args.topn)
    J = [len(set(top_o[i]) & set(top_3[i])) / len(set(top_o[i]) | set(top_3[i]))
         for i in range(len(Xs))]
    J_mean = float(np.mean(J))
    J_std = float(np.std(J, ddof=0))

    # 出力
    result = {
        "n": int(X.shape[0]),
        "feature_cols": feat_cols,
        "explained_variance_ratio": [float(x) for x in evr],
        "cumulative_explained_variance": {"m=1": cum1, "m=2": cum2, "m=3": cum3},
        "spearman_rho_orig_vs_map2": rho,
        "neighbor_preservation": {"k": args.k, "mean": Pk_mean, "std": Pk_std},
        "jaccard": {"topn": args.topn, "mean": J_mean, "std": J_std},
    }

    print("=== Chapter 5 Metrics ===")
    print(f"n={result['n']}")
    print(f"cumEVR m=1: {cum1:.3f}, m=2: {cum2:.3f}, m=3: {cum3:.3f}")
    print(f"Spearman rho (orig euclid vs map2 euclid): {rho:.3f}")
    print(f"Neighbor preservation k={args.k}: mean={Pk_mean:.3f}, std={Pk_std:.3f}")
    print(f"Jaccard@{args.topn} (orig cosine vs PCA3 cosine): mean={J_mean:.3f}, std={J_std:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
