import pathlib
import pandas as pd#データ操作
import numpy as np#ベクトル・行列計算
from sklearn.preprocessing import StandardScaler#標準化
from sklearn.decomposition import PCA#主成分分析
from sklearn.cluster import KMeans#KMeansクラスタリング
import plotly.express as px
import plotly.io as pio

FEATURES = [
    "Body","Sweetness","Smoky","Medicinal","Tobacco","Honey",
    "Spicy","Winey","Nutty","Malty","Fruity","Floral"
]

def main():
    df = pd.read_csv("whisky_clean.csv")#前処理済みデータを読み込み
    X = df[FEATURES].values

    scaler = StandardScaler()#標準化
    Xs = scaler.fit_transform(X)#12軸を標準化

    pca = PCA(n_components=2, random_state=0)
    Xp = pca.fit_transform(Xs)
    df["PC1"], df["PC2"] = Xp[:,0], Xp[:,1]#出力PC1, PC2列追加（新しい座標）
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    # KMeans（とりあえずk=5）
    k = 5
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    df["Cluster"] = km.fit_predict(Xs)

    outdir = pathlib.Path("result"); outdir.mkdir(exist_ok=True)
    # インタラクティブ図（HTML）
    fig = px.scatter(df, x="PC1", y="PC2", color="Cluster",
                     hover_data=["Distillery"] + FEATURES,
                     title="Whisky Flavour Map (PCA + KMeans)")
    pio.write_html(fig, file=str(outdir / "whisky_pca_clusters.html"), auto_open=False)
    # CSV 出力: recommend.py が利用するファイル
    out_csv = outdir / "whisky_pca_clusters.csv"
    df.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)
    # 画像保存（必要なら）→ 初回は kaleido が必要: pip install -U kaleido
    # pio.write_image(fig, str(outdir / "whisky_pca_clusters.png"), scale=2)

    # クラスタごとの平均をCSV
    profile = df.groupby("Cluster")[FEATURES].mean().round(2)
    profile.to_csv(outdir / "cluster_profile.csv")
    print("✅ Saved:", outdir / "whisky_pca_clusters.html")
    print("✅ Saved:", outdir / "cluster_profile.csv")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1","PC2"],
        index=FEATURES
    )
    print(loadings)


if __name__ == "__main__":
    main()
