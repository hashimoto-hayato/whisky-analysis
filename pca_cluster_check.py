import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# データ読み込み（本番と同じCSVを使う）
df = pd.read_csv("whisky_clean.csv")

# 香味特徴（12項目）
features = [
    "Body", "Sweetness", "Smoky", "Medicinal", "Tobacco",
    "Honey", "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"
]

X = df[features].values

# 正規化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# PCA（全成分）
pca = PCA()
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cum = explained.cumsum()

print("Explained variance ratio:")
for i in range(3):
    print(f"PC{i+1}: {explained[i]:.3f}")

print(f"Cumulative variance (PC1-3): {cum[2]:.3f}")
