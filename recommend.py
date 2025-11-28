import numpy as np
import pandas as pd
import pathlib

FEATURES = [
    "Body", "Sweetness", "Smoky", "Medicinal", "Tobacco",
    "Honey", "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"
]

def load_whisky(csv_path="result/whisky_pca_clusters.csv"):#推薦の元データを読み込む
    csv_path = pathlib.Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path}")
    except FileNotFoundError:
        # フォールバック: 元データ（前処理済み）から読み込む
        fallback = pathlib.Path("whisky_clean.csv")
        print(f"Warning: {csv_path} not found. Falling back to {fallback}.")
        df = pd.read_csv(fallback)

    # RowID がなければ作成（1始まり）
    if "RowID" not in df.columns:
        df = df.copy()
        df["RowID"] = np.arange(1, len(df) + 1)

    # Cluster 列が無ければダミー値を設定
    if "Cluster" not in df.columns:
        df["Cluster"] = -1

    # 必要な FEATURES が存在するか確認し、型を整える
    for f in FEATURES:
        if f not in df.columns:
            raise KeyError(f"Required feature column missing: {f}")
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    return df

def compute_user_vector(favorite_ids, whisky_df):#お気に入りウイスキーのベクトル平均を計算
    if len(favorite_ids) == 0:
        return None

    fav_vectors = whisky_df.loc[
        whisky_df["RowID"].isin(favorite_ids), FEATURES
    ].values

    user_vec = fav_vectors.mean(axis=0)
    user_vec = user_vec / np.linalg.norm(user_vec)
    return user_vec

def recommend_whisky(user_vec, whisky_df, top_n=10, exclude_ids=None):#コサイン類似度で推薦
    if user_vec is None:
        return pd.DataFrame()

    X = whisky_df[FEATURES].values
    norms = np.linalg.norm(X, axis=1).reshape(-1, 1)
    X_normed = X / norms

    sims = X_normed @ user_vec  # コサイン類似度を一括計算
    df = whisky_df.copy()
    df["similarity"] = sims #類似度をsimilarity列に追加

    # すでにお気に入りのIDは除外
    if exclude_ids is not None:
        df = df[~df["RowID"].isin(exclude_ids)]

    rec = df.sort_values("similarity", ascending=False).head(top_n) 
    return rec[["RowID", "Distillery", "Cluster", "similarity"]]    #類似度順にソートして上位top_n件を返す

if __name__ == "__main__":
    whisky_df = load_whisky()

    # テスト用お気に入り（RowIDで指定）
    favorite_ids = [3, 10, 25]

    user_vec = compute_user_vector(favorite_ids, whisky_df)
    rec = recommend_whisky(user_vec, whisky_df, top_n=10, exclude_ids=favorite_ids)

    print("=== おすすめウイスキー ===")
    print(rec)
