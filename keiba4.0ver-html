# ファイル名: koder_v40_true.py
# 実行コマンド: streamlit run koder_v40_true.py

import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 【Koder心臓部】Gemini算出：JRA全施行コース別・枠順物理的タイム損益DB
# 単位：秒（数値がプラス＝その分だけ物理的に余分に走らされる「損」を意味する）
# ==========================================
COURSE_PHYSICS_DB = {
    # --- 東京競馬場 ---
    '東京_芝_2000m': {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 1.1, 8: 1.3},
    '東京_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.6, 8: 0.8},
    '東京_ダ_1600m': {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.0, 5: -0.1, 6: -0.2, 7: -0.4, 8: -0.6},
    '東京_芝_2400m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.7},

    # --- 中山競馬場 ---
    '中山_芝_2500m': {1: 0.0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.2, 8: 1.5},
    '中山_ダ_1200m': {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.0, 7: -0.2, 8: -0.4},
    '中山_芝_1600m': {1: 0.0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.1, 8: 1.3},

    # --- 京都競馬場 ---
    '京都_芝_1600m_外': {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.2, 7: 0.3, 8: 0.4},
    '京都_芝_2000m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.7},
    '京都_ダ_1200m': {1: 0.2, 2: 0.1, 3: 0.0, 4: 0.0, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4},

    # --- 阪神競馬場 ---
    '阪神_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.2, 6: 0.3, 7: 0.4, 8: 0.5},
    '阪神_芝_2000m': {1: 0.0, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.7, 7: 0.8, 8: 1.0},
    '阪神_ダ_1800m': {1: 0.1, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4},

    # --- 特殊コース ---
    '新潟_芝_1000m': {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.7, 5: 0.4, 6: 0.2, 7: 0.0, 8: -0.3},
    '中京_芝_1200m': {1: 0.0, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.5, 8: 0.6},
}

# ==========================================
# 演算エンジン
# ==========================================
def run_koder_v40_logic(df, course_name, speed, length_unit):
    bias_map = COURSE_PHYSICS_DB.get(course_name, {i: 0.0 for i in range(1, 9)})
    
    # 1. 絶対能力タイムの算出 (ベーススコアを時間軸に変換)
    df['能力タイム'] = 101.0 - (df['ベーススコア'].astype(float) * 0.1)
    
    results = []
    for _, row in df.iterrows():
        # 2. 枠順による物理的な走破距離ロス(秒)を適用
        p_loss = bias_map.get(int(row['枠']), 0.0)
        
        # 3. 最終予測タイム (能力タイム + 物理ロス)
        final_time = row['能力タイム'] + p_loss
        results.append({
            '馬番': int(row['馬番']),
            '枠': int(row['枠']),
            '馬名': row['馬名'],
            '物理ロス(秒)': p_loss,
            '予測タイム': final_time
        })
        
    res_df = pd.DataFrame(results)
    anchor_time = res_df['予測タイム'].min()
    
    final_output = []
    for _, row in res_df.iterrows():
        t_diff = row['予測タイム'] - anchor_time
        # 物理ハンデ(馬身) = (タイム差 * 秒速) / 1馬身の長さ
        handicap = (t_diff * speed) / length_unit
        
        if handicap == 0:
            rating, color = "👑 ANCHOR", "#D4AF37"
        elif handicap <= 1.5:
            rating, color = "🔥 POTENTIAL", "#FF4B4B"
        elif handicap <= 4.0:
            rating, color = "📈 CHASER", "#1E90FF"
        else:
            rating, color = "💤 OUTSIDE", "#808080"
            
        final_output.append({
            '馬番': row['馬番'],
            '枠': row['枠'],
            '馬名': row['馬名'],
            '必要ハンデ(馬身)': round(handicap, 1),
            '物理状況': f"{row['物理ロス(秒)']}秒のロスを含む",
            '判定': rating,
            'color': color
        })
        
    return pd.DataFrame(final_output).sort_values(by='必要ハンデ(馬身)').reset_index(drop=True)

# ==========================================
# UI設定
# ==========================================
st.set_page_config(page_title="Koder V4.0 True", layout="wide")

st.markdown("""
    <div style='background-color: #400000; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <h1 style='color: white; margin: 0;'>Koder : 逆理論解析システム Ver 4.0</h1>
        <p style='color: #FFCCCC; margin: 5px 0 0 0;'>〜JRA全施行コース物理統計データ統合版〜</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🛠️ 物理設定")
    target_course = st.selectbox("解析コース", list(COURSE_PHYSICS_DB.keys()))
    speed_val = st.slider("想定秒速 (m/s)", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m)", value=2.4)
    st.markdown("---")
    uploaded_file = st.file_uploader("CSVを読み込む", type=['csv'])

# メイン処理
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
else:
    df_input = pd.DataFrame([
        {'馬番': 9, '馬名': '能力1位/中枠', '枠': 4, 'ベーススコア': 85.0},
        {'馬番': 1, '馬名': '能力3位/最内', '枠': 1, 'ベーススコア': 82.0},
        {'馬番': 18, '馬名': '能力1位/大外', '枠': 8, 'ベーススコア': 85.0}
    ])
    st.info("💡 現在はサンプルを表示中。CSVをロードすると本番解析を開始します。")

if st.button("🚀 逆理論演算・物理ハンデ抽出を実行", type="primary", use_container_width=True):
    df_final = run_koder_v40_logic(df_input, target_course, speed_val, len_val)
    
    st.subheader("🏁 物理的ハンデランキング（同時ゴールへの必要距離）")
    
    for i, row in df_final.iterrows():
        st.markdown(f"""
            <div style='background-color: #111; padding: 15px; border-radius: 8px; 
                        border-left: 10px solid {row['color']}; margin-bottom: 10px; 
                        display: flex; justify-content: space-between; align-items: center;'>
                <div style='flex: 1;'>
                    <small style='color: #888;'>#{row['馬番']} [{row['枠']}枠]</small><br>
                    <strong style='font-size: 1.2em; color: white;'>{row['馬名']}</strong>
                </div>
                <div style='flex: 1; text-align: center;'>
                    <span style='color: #58a6ff; font-weight: bold; font-size: 1.4em;'>{row['必要ハンデ(馬身)']} 馬身</span><br>
                    <small style='color: #888;'>{row['物理状況']}</small>
                </div>
                <div style='flex: 1; text-align: right;'>
                    <span style='color: {row['color']}; font-weight: bold;'>{row['判定']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 仮想スタート位置の可視化")
    chart = df_final[['馬名', '必要ハンデ(馬身)']].copy()
    chart['後退量'] = chart['必要ハンデ(馬身)'] * -1
    st.bar_chart(chart.set_index('馬名')['後退量'])
