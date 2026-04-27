　# ファイル名: koder_master_fusion.py
# 実行コマンド: streamlit run koder_master_fusion.py

import streamlit as st
import pandas as pd

# ==========================================
# 【Koder心臓部】JRA全施行コース物理統計データベース
# ==========================================
COURSE_PHYSICS_DB = {
    '東京_芝_2000m': {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 1.1, 8: 1.3},
    '東京_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.6, 8: 0.8},
    '中山_ダ_1200m': {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.0, 7: -0.2, 8: -0.4},
    '京都_芝_1600m_外': {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.2, 7: 0.3, 8: 0.4},
    '阪神_芝_2000m': {1: 0.0, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.7, 7: 0.8, 8: 1.0},
    # ※本番稼働時は全コースを追加してください
}

# ==========================================
# 右目：Ver 3.5（平面昇華・アヤの評価）
# ==========================================
def run_v35_logic(df, course, track_cond, pace, race_3f_time):
    race_3f_avg = race_3f_time / 3.0
    bias_map = COURSE_PHYSICS_DB.get(course, {i: 0 for i in range(1, 9)})
    
    results = []
    for _, row in df.iterrows():
        base = float(row['ベーススコア'])
        
        # PCIと血統の七味
        horse_3f_avg = float(row['上がり3F']) / 3.0
        pci = 50 + (race_3f_avg - horse_3f_avg) * 10
        pci_adj = (pci - 50) * 0.4
        
        lineage = str(row['血統'])
        l_bonus = 1.12 if (track_cond == '高速' and lineage == '米国型') or ((track_cond == 'タフ' or pace == 'H') and lineage == '欧州型') else 1.10 if (pace == 'S' and lineage == '日本型') else 1.0
        
        # 物理バイアスをスコア化（1秒ロス=10点減点換算）
        p_loss_sec = bias_map.get(int(row['枠']), 0.0)
        frame_adj = -(p_loss_sec * 10) 
        
        final_score = (base * l_bonus) + pci_adj + frame_adj
        results.append({'馬番': int(row['馬番']), 'V35_スコア': round(final_score, 2)})
        
    df_res = pd.DataFrame(results).sort_values(by='V35_スコア', ascending=False).reset_index(drop=True)
    df_res['V35_順位'] = df_res.index + 1
    return df_res

# ==========================================
# 左目：Ver 4.0（物理逆算・絶対距離の評価）
# ==========================================
def run_v40_logic(df, course, speed, length_unit):
    bias_map = COURSE_PHYSICS_DB.get(course, {i: 0.0 for i in range(1, 9)})
    df['能力タイム'] = 101.0 - (df['ベーススコア'].astype(float) * 0.1)
    
    results = []
    for _, row in df.iterrows():
        p_loss = bias_map.get(int(row['枠']), 0.0)
        final_time = row['能力タイム'] + p_loss
        results.append({'馬番': int(row['馬番']), '予測タイム': final_time})
        
    res_df = pd.DataFrame(results)
    anchor_time = res_df['予測タイム'].min()
    
    final_output = []
    for _, row in res_df.iterrows():
        t_diff = row['予測タイム'] - anchor_time
        handicap = (t_diff * speed) / length_unit
        final_output.append({'馬番': int(row['馬番']), 'V40_ハンデ': round(handicap, 1)})
        
    return pd.DataFrame(final_output)

# ==========================================
# コアブレイン：融合判定マトリクス
# ==========================================
def execute_fusion_matrix(df_raw, df_35, df_40):
    # データを結合
    merged = pd.merge(df_raw, df_35, on='馬番')
    merged = pd.merge(merged, df_40, on='馬番')
    
    fusion_results = []
    for _, row in merged.iterrows():
        rank = row['V35_順位']
        hc = row['V40_ハンデ']
        
        # 投資判定アルゴリズム
        if rank <= 2 and hc <= 1.0:
            grade = "S"
            label = "完全無欠の絶対軸"
            color = "#FFD700" # ゴールド
            action = "【勝負】厚め単複・連軸"
        elif rank <= 4 and hc == 0.0:
            grade = "A"
            label = "特大のオッズバグ"
            color = "#FF4B4B" # レッド
            action = "【妙味】単勝・ヒモ穴狙い"
        elif rank == 1 and hc >= 5.0:
            grade = "B"
            label = "危険な人気馬"
            color = "#9C27B0" # パープル
            action = "【警戒】投資見送り・ヒモまで"
        elif rank > 4 and hc > 4.0:
            grade = "C"
            label = "完全ノイズ"
            color = "#333333" # ダークグレー
            action = "【消し】購入対象外"
        else:
            grade = "R"
            label = "連下候補 (レギュラー)"
            color = "#1E90FF" # ブルー
            action = "【通常】相手候補として保持"
            
        fusion_results.append({
            '馬番': row['馬番'],
            '枠': row['枠'],
            '馬名': row['馬名'],
            'V35_順位': rank,
            'V40_ハンデ': hc,
            '判定': grade,
            'ステータス': label,
            '推奨アクション': action,
            'color': color
        })
        
    # S, A, R, B, C の順にソートするロジック
    sort_order = {"S": 1, "A": 2, "R": 3, "B": 4, "C": 5}
    df_fusion = pd.DataFrame(fusion_results)
    df_fusion['sort_key'] = df_fusion['判定'].map(sort_order)
    return df_fusion.sort_values(by=['sort_key', 'V40_ハンデ']).drop(columns=['sort_key']).reset_index(drop=True)

# ==========================================
# Web UI：マスターダッシュボード
# ==========================================
st.set_page_config(page_title="Koder Master Fusion", layout="wide")

st.markdown("""
    <div style='background-color: #0A0A0A; padding: 20px; border-bottom: 3px solid #00FF00; margin-bottom: 20px;'>
        <h1 style='color: #00FF00; margin: 0; font-family: monospace;'>Koder : MASTER FUSION ENGINE [AUTO-READY]</h1>
        <p style='color: #888; margin: 5px 0 0 0;'>〜Ver 3.5 × Ver 4.0 統合投資判定ダッシュボード〜</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ システム連動パラメータ")
    sel_course = st.selectbox("解析対象コース", list(COURSE_PHYSICS_DB.keys()))
    sel_cond = st.selectbox("馬場状態", ["標準", "高速", "タフ"])
    sel_pace = st.selectbox("展開予測", ["M", "S", "H"])
    sel_3f = st.number_input("前半3F想定タイム", value=35.5, step=0.1)
    st.markdown("---")
    speed_val = st.slider("想定秒速 (m/s) [V4.0用]", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m) [V4.0用]", value=2.4)
    st.markdown("---")
    uploaded_file = st.file_uploader("本番CSVをロード", type=['csv'])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    # 統合テスト用の緻密なサンプルデータ（全パターン網羅）
    df_raw = pd.DataFrame([
        {'馬番': 1, '馬名': '絶対王者(S評価テスト)', '枠': 1, 'ベーススコア': 88.0, '上がり3F': 34.0, '血統': '日本型'},
        {'馬番': 18, '馬名': '危険な人気馬(B評価テスト)', '枠': 8, 'ベーススコア': 90.0, '上がり3F': 34.2, '血統': '日本型'},
        {'馬番': 5, '馬名': '隠れバグ馬(A評価テスト)', '枠': 3, 'ベーススコア': 80.0, '上がり3F': 34.5, '血統': '米国型'},
        {'馬番': 7, '馬名': '普通の連下(R評価テスト)', '枠': 4, 'ベーススコア': 78.0, '上がり3F': 35.0, '血統': '欧州型'},
        {'馬番': 15, '馬名': '完全ノイズ(C評価テスト)', '枠': 7, 'ベーススコア': 50.0, '上がり3F': 36.5, '血統': '日本型'}
    ])
    st.info("💡 システム統合テスト稼働中。左のパネルからCSVを投入すると実稼働モードに移行します。")

if st.button("⚡ FUSION ENGINE 起動 (投資判定実行)", type="primary", use_container_width=True):
    with st.spinner('両眼解析システム照合中... 投資期待値を抽出しています...'):
        
        # 1. 右目・左目の独立演算
        df_35 = run_v35_logic(df_raw, sel_course, sel_cond, sel_pace, sel_3f)
        df_40 = run_v40_logic(df_raw, sel_course, speed_val, len_val)
        
        # 2. マトリクス統合
        df_fusion = execute_fusion_matrix(df_raw, df_35, df_40)
        
        st.subheader("🎯 最終投資判定マトリクス (自動資金配分ターゲット)")
        
        for _, row in df_fusion.iterrows():
            st.markdown(f"""
                <div style='background-color: #111; padding: 15px; border-radius: 5px; 
                            border-left: 10px solid {row['color']}; margin-bottom: 8px;
                            display: flex; justify-content: space-between; align-items: center;'>
                    
                    <div style='flex: 0.5; text-align: center;'>
                        <h2 style='margin:0; color: {row['color']};'>{row['判定']}</h2>
                    </div>
                    
                    <div style='flex: 2;'>
                        <span style='color: #888;'>#{row['馬番']} [{row['枠']}枠]</span><br>
                        <strong style='font-size: 1.2em; color: white;'>{row['馬名']}</strong>
                    </div>
                    
                    <div style='flex: 1.5;'>
                        <small style='color: #aaa;'>右目 (V3.5):</small> <b style='color: white;'>{row['V35_順位']}位</b><br>
                        <small style='color: #aaa;'>左目 (V4.0):</small> <b style='color: white;'>{row['V40_ハンデ']} 馬身差</b>
                    </div>
                    
                    <div style='flex: 2; text-align: right;'>
                        <span style='color: {row['color']}; font-weight: bold;'>{row['ステータス']}</span><br>
                        <span style='background-color: #222; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; color: #ddd;'>{row['推奨アクション']}</span>
                    </div>
                    
                </div>
            """, unsafe_allow_html=True)
