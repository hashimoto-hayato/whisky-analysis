import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    "Body", "Sweetness", "Smoky", "Medicinal", "Tobacco",
    "Honey", "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"
]

def euclidean_dist_matrix(X):
    # (n,d) -> (n,n)
    # ||x-y||^2 = x^2 + y^2 - 2xy
    sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * (X @ X.T)
    D2 = np.maximum(D2, 0)
    return np.sqrt(D2)

def build_embeddings(df, dims):
    X = df[FEATURES].to_numpy()
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    if dims == 12:
        return Xs
    pca = PCA(n_components=dims, random_state=0)
    Z = pca.fit_transform(Xs)
    return Z

def recommend_topN(df, embed, query_idx, metric="cosine", topN=5):
    if metric == "cosine":
        sims = cosine_similarity(embed[query_idx:query_idx+1], embed).ravel()
        order = np.argsort(-sims)
        # 自分自身除外
        order = order[order != query_idx][:topN]
        return df.iloc[order].copy(), sims[order]
    elif metric == "euclid":
        D = euclidean_dist_matrix(embed)
        dist = D[query_idx]
        order = np.argsort(dist)
        order = order[order != query_idx][:topN]
        return df.iloc[order].copy(), dist[order]
    else:
        raise ValueError("metric must be 'cosine' or 'euclid'")

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def main():
    df = pd.read_csv("whisky_clean.csv")

    # 銘柄名列はあなたのCSVに合わせて調整
    # 例: "Distillery" や "Name" など
    NAME_COL = "Distillery" if "Distillery" in df.columns else df.columns[0]

    # 比較したいクエリ（例：3銘柄）
    # 自分がよく知っている銘柄 or 代表的な特徴の銘柄を選ぶのが論文的に強い
    query_names = [
        df[NAME_COL].iloc[0],
        df[NAME_COL].iloc[10],
        df[NAME_COL].iloc[20],
    ]

    configs = [
        ("PCA2", 2, "cosine"),
        ("PCA2", 2, "euclid"),
        ("PCA3", 3, "cosine"),
        ("PCA3", 3, "euclid"),
        # 余裕があればベースライン：
        # ("RAW12", 12, "cosine"),
        # ("RAW12", 12, "euclid"),
    ]

    rows = []
    overlap_rows = []

    for qname in query_names:
        qidx = df.index[df[NAME_COL] == qname][0]

        results = {}
        for tag, dims, metric in configs:
            emb = build_embeddings(df, dims)
            rec_df, scores = recommend_topN(df, emb, qidx, metric=metric, topN=5)

            # 表用に整形
            rec_names = rec_df[NAME_COL].tolist()
            results[(tag, metric)] = rec_names

            for rank, (nm, sc) in enumerate(zip(rec_names, scores), start=1):
                rows.append({
                    "query": qname,
                    "space": tag,
                    "metric": metric,
                    "rank": rank,
                    "recommendation": nm,
                    "score_or_dist": float(sc),
                })

        # 2D cosine vs 2D euclid の一致度（Top5の重なり）
        jc = jaccard(results[("PCA2","cosine")], results[("PCA2","euclid")])
        overlap_rows.append({"query": qname, "compare": "PCA2 cosine vs euclid", "jaccard@5": jc})

        # 2D cosine vs 3D cosine
        jc2 = jaccard(results[("PCA2","cosine")], results[("PCA3","cosine")])
        overlap_rows.append({"query": qname, "compare": "cosine PCA2 vs PCA3", "jaccard@5": jc2})

    out = pd.DataFrame(rows)
    out.to_csv("result/recommend_compare_detail.csv", index=False)

    out2 = pd.DataFrame(overlap_rows)
    out2.to_csv("result/recommend_compare_overlap.csv", index=False)

    print("✅ Saved: result/recommend_compare_detail.csv")
    print("✅ Saved: result/recommend_compare_overlap.csv")
    print(out2)

if __name__ == "__main__":
    main()
