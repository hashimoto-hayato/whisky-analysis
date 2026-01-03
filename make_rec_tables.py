import pandas as pd

DETAIL_PATH = "result/recommend_compare_detail.csv"

def top5_list(df, query, space):
    sub = df[(df["query"] == query) &
             (df["space"] == space) &
             (df["metric"] == "cosine")].sort_values("rank")
    return sub["recommendation"].head(5).tolist()

def main():
    df = pd.read_csv(DETAIL_PATH)

    queries = ["Aberfeldy", "Balmenach", "Bunnahabhain"]  # 好きに減らしてOK

    for q in queries:
        rec2 = top5_list(df, q, "PCA2")
        rec3 = top5_list(df, q, "PCA3")

        print(f"\n=== {q} ===")
        print("PCA2 + cosine:", ", ".join(rec2))
        print("PCA3 + cosine:", ", ".join(rec3))

if __name__ == "__main__":
    main()
