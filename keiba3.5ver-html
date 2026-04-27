# ファイル名: koder_v35_app.py
# 実行コマンド: streamlit run koder_v35_app.py

import streamlit as st
import pandas as pd

# ==========================================
# 物理バイアス・マスターデータ（Gemini算出定数）
# ==========================================
FRAME_BIAS_MASTER = {
    '東京_芝_2000m': {1: 4.0, 2: 3.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: -1.0, 7: -4.0, 8: -6.0},
    '東京_芝_1600m': {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -1.0, 8: -2.0},
    '中山_芝_2000m': {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: -1.0, 7: -2.0, 8: -3.0},
    '中山_ダ_1200m': {1: -3.0, 2: -2.0, 3: -1.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 3.0, 8: 5.0},
    '京都_芝_1600m_外': {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -1.0, 8: -1.0},
    '阪神_芝_2200m_内': {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: -1.0, 7: -2.0, 8: -3.0}
}

# ==========================================
# コア演算エンジン
# ==========================================
def analyze_koder_v35(df, course, track_cond, pace, race_first_3f):
    race_3f_avg = race_first_3f / 3.0
    bias_map = FRAME_BIAS_MASTER.get(course, {i: 0 for i in range(1, 9)})
    
    results = []
    for _, row in df.iterrows():
        base = float(row['ベーススコア'])
        
        # 1. 自作PCI（失速耐性）
        horse_3f_avg = float(row['上がり3F']) / 3.0
        pci = 50 + (race_3f_avg - horse_3f_avg) * 10
        pci_adj = (pci - 50) * 0.4
        
        # 2. 血統アクセント（七味）
        lineage = str(row['血統'])
        lineage_bonus = 1.0
        if track_cond == '高速' and lineage == '米国型':
            lineage_bonus = 1.12
        elif (track_cond == 'タフ' or pace == 'H') and lineage == '欧州型':
            lineage_bonus = 1.12
        elif pace == 'S' and lineage == '日本型':
            lineage_bonus = 1.10
            
        # 3. 枠順物理バイアス
        frame_adj = bias_map.get(int(row['枠']), 0)
        
        # 【平面上での昇華】最終スコア
        final_score = (base * lineage_bonus) + pci_adj + frame_adj
        
        results.append({
            '馬番': int(row['馬番']),
            '枠': int(row['枠']),
            '馬名': str(row['馬名']),
            '出汁(ベース)': round(base, 1),
            'PCI粘り': round(pci, 1),
            '枠補正': frame_adj,
            '血統バフ': f"x{lineage_bonus}",
            '👑 総合スコア': round(final_score, 2)
        })
        
    # スコア順にソート
    df_res = pd.DataFrame(results).sort_values(by='👑 総合スコア', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

# ==========================================
# Web UI レイアウト設計
# ==========================================
st.set_page_config(page_title="Koder V3.5", layout="wide", initial_sidebar_state="expanded")

# ヘッダーデザイン
st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h1 style='color: #4CAF50; margin: 0;'>Koder : 競馬予測システム Ver 3.5</h1>
        <p style='color: #DDDDDD; margin: 0;'>〜全頭フィルター・高精度4頭抽出コアエンジン〜</p>
    </div>
""", unsafe_allow_html=True)

# サイドバー：環境設定とデータ入力
with st.sidebar:
    st.header("⚙️ レース環境設定")
    selected_course = st.selectbox("コース選択 (物理バイアス用)", list(FRAME_BIAS_MASTER.keys()))
    selected_cond = st.selectbox("馬場状態", ["標準", "高速", "タフ"])
    selected_pace = st.selectbox("展開予測", ["M", "S", "H"])
    input_3f_time = st.number_input("前半3F想定タイム (秒)", value=35.5, step=0.1)
    
    st.markdown("---")
    st.header("📂 データ読み込み")
    uploaded_file = st.file_uploader("本番CSVファイルをアップロード", type=['csv'])
    
    st.markdown("※CSV必須カラム: `馬番`, `馬名`, `枠`, `ベーススコア`, `上がり3F`, `血統`")

# メイン画面ロジック
if uploaded_file is not None:
    # 本番データの読み込み
    df_raw = pd.read_csv(uploaded_file)
else:
    # ダミーデータ（テスト・レイアウト確認用）
    df_raw = pd.DataFrame([
        {'馬番': 1, '馬名': 'インベタ特注馬', '枠': 1, 'ベーススコア': 72.0, '上がり3F': 35.8, '血統': '日本型'},
        {'馬番': 4, '馬名': 'システムベース軸', '枠': 2, 'ベーススコア': 80.0, '上がり3F': 35.0, '血統': '日本型'},
        {'馬番': 9, '馬名': '現行1番手(ダミー)', '枠': 5, 'ベーススコア': 82.0, '上がり3F': 34.8, '血統': '欧州型'},
        {'馬番': 10, '馬名': '大外強襲バグ馬', '枠': 5, 'ベーススコア': 75.0, '上がり3F': 34.1, '血統': '米国型'},
        {'馬番': 15, '馬名': '惜しい5着馬', '枠': 8, 'ベーススコア': 74.0, '上がり3F': 35.2, '血統': '欧州型'},
        {'馬番': 18, '馬名': '完全ノイズ馬', '枠': 8, 'ベーススコア': 60.0, '上がり3F': 36.5, '血統': '米国型'}
    ])
    st.info("💡 現在はレイアウト確認用のダミーデータを表示しています。左のサイドバーからCSVをアップロードすると本番稼働します。")

# 解析実行ボタン
if st.button("🚀 Koder V3.5 抽出エンジン起動", type="primary", use_container_width=True):
    with st.spinner('全頭の出汁と七味を融合中...'):
        df_final = analyze_koder_v35(df_raw, selected_course, selected_cond, selected_pace, input_3f_time)
        
        st.success("✅ 解析完了：コア・ポートフォリオを抽出しました。")
        
        # --- レイアウト：上位4頭のハイライト表示 ---
        st.markdown("### 🎯 最終抽出ランキング (上位4頭コア・ポートフォリオ)")
        
        # トップ4を視覚的に分割して表示 (メトリクスカード風)
        top4 = df_final.head(4)
        cols = st.columns(4)
        for i, (idx, row) in enumerate(top4.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color: #2b3a42; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50;'>
                    <h4 style='margin:0; color: #ffffff;'>{i+1}位：{row['馬名']}</h4>
                    <p style='margin:0; font-size: 24px; color: #4CAF50; font-weight: bold;'>{row['👑 総合スコア']}</p>
                    <small style='color: #aaaaaa;'>枠:{row['枠']} / PCI:{row['PCI粘り']}</small>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- レイアウト：全頭詳細データテーブル ---
        st.markdown("### 📊 全頭解析マトリクス")
        # スコアにグラデーションをかけて視覚的に強弱をアピール
        styled_df = df_final.style.background_gradient(cmap='Greens', subset=['👑 総合スコア', '出汁(ベース)', 'PCI粘り'])
        st.dataframe(styled_df, use_container_width=True, height=500)

