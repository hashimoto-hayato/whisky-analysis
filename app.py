# app.py
import streamlit as st #StreamlitはWeb UI構築用フレームワーク
import pandas as pd #表形式のデータ操作用
import plotly.express as px #インタラクティブな散布図を描画

#recommend.py から関数をインポート
from recommend import load_whisky, compute_user_vector, recommend_whisky, FEATURES

# データ読み込み
whisky_df = load_whisky()  # result/whisky_pca_clusters.csv を優先

st.set_page_config(page_title="Whisky Recommender", layout="wide") #タイトルと横幅レイアウトを設定

st.title("ウイスキー味覚マップ＆レコメンドシステム（試作）")

st.markdown("""
このアプリでは、  
- **味覚マップ（PCA）** 上でウイスキーの位置を確認したり  
- **お気に入りに近いウイスキーの推薦** を受け取ることができます。
""")

# タブで画面を分ける
tab1, tab2 = st.tabs(["🍶 お気に入りからおすすめ", "📈 味覚マップ"]) #タブを作成

# ======================
# タブ1：おすすめ表示
# ======================
with tab1:
    st.subheader("お気に入りのウイスキーを選んで、おすすめを表示")

    # 銘柄一覧から複数選択
    distillery_names = whisky_df["Distillery"].tolist()
    selected = st.multiselect(
        "お気に入り（好きな銘柄）を選択してください",
        distillery_names
    )

    if st.button("おすすめを表示"):
        if len(selected) == 0:
            st.warning("少なくとも1つはお気に入りを選んでください。")
        else:
            # 選択された銘柄のRowIDを取得
            fav_ids = whisky_df[whisky_df["Distillery"].isin(selected)]["RowID"].tolist()

            # ユーザーベクトル計算
            user_vec = compute_user_vector(fav_ids, whisky_df)

            if user_vec is None:
                st.error("ユーザー嗜好ベクトルを計算できませんでした。")
            else:
                # レコメンド実行
                rec = recommend_whisky(user_vec, whisky_df, top_n=10, exclude_ids=fav_ids)

                st.markdown("### おすすめウイスキー（上位10件）")
                st.dataframe(rec)

# ======================
# タブ2：味覚マップ表示
# ======================
with tab2:
    st.subheader("PCAによるウイスキー味覚マップ")

    if "PC1" not in whisky_df.columns or "PC2" not in whisky_df.columns:
        st.error("PC1 / PC2 の列が見つかりません。pca_cluster.py を先に実行してください。")
    else:
        # クラスタごとに色分けした散布図
        fig = px.scatter(
            whisky_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="Distillery",
            hover_data=FEATURES,
            title="ウイスキー味覚マップ（PC1 × PC2）"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
- 近くに位置する点ほど、香味特徴が似ているウイスキーを表します。  
- 色は K-means によるクラスタ（味のタイプ）を表しています。
""")
