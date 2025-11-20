import pandas as pd #データ整形ライブラリ
from kagglehub import dataset_load, KaggleDatasetAdapter# Kaggleデータセット取得用

def load_raw() -> pd.DataFrame:# 生データ取得
    # Kaggleから取得（CC0のScotch Whisky Dataset）
    df = dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "koki25ando/scotch-whisky-dataset",
        "whisky.csv"
    )
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 列名トリム
    df.columns = [c.strip() for c in df.columns]

    # Postcodeの余分な空白を整理
    if "Postcode" in df.columns:
        df["Postcode"] = df["Postcode"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # 解析対象の12軸
    feature_cols = [
        "Body","Sweetness","Smoky","Medicinal","Tobacco","Honey",
        "Spicy","Winey","Nutty","Malty","Fruity","Floral"
    ]

    # 数値化
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 欠損は列平均で補完
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # 使わない列を落とす（ID/位置情報など）
    drop_cols = [c for c in ["RowID","Postcode","Latitude","Longitude"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 蒸留所名を先頭にしておく
    cols = ["Distillery"] + [c for c in df.columns if c != "Distillery"]
    df = df[cols]

    return df

def main():
    raw = load_raw()
    print("Raw shape:", raw.shape)

    clean_df = clean(raw)
    print("Clean shape:", clean_df.shape)
    print(clean_df.head(3))

    out = "whisky_clean.csv"
    clean_df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Saved: {out}")

if __name__ == "__main__":
    main()
