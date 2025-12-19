import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# データ読み込み（既存の前処理済CSVを使う）
df = pd.read_csv("whisky_clean.csv")

# 香味特徴量（12次元）
features = [
    "Body", "Sweetness", "Smoky", "Medicinal", "Tobacco",
    "Honey", "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"
]

X = df[features]

# 正規化（既存方針と一致させる）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 次元ごとのPCA
for n_components in [1, 2, 3]:
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = explained.sum()

    print(f"{n_components}次元 PCA")
    for i, v in enumerate(explained, start=1):
        print(f"  PC{i}: {v:.3f}")
    print(f"  累積寄与率: {cumulative:.3f}")
    print("-" * 30)
