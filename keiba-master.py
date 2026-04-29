import streamlit as st
import pandas as pd
import time
import random
import io

# ==========================================
# 【Koder心臓部】JRA全施行コース物理統計データベース
# ==========================================
COURSE_PHYSICS_DB = {
    '東京_芝_2000m': {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 1.1, 8: 1.3},
    '東京_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.6, 8: 0.8},
    '中山_ダ_1200m': {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.0, 7: -0.2, 8: -0.4},
    '京都_芝_1600m_外': {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.2, 7: 0.3, 8: 0.4},
    '阪神_芝_2000m': {1: 0.0, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.7, 7: 0.8, 8: 1.0},
    '新潟_芝_1000m': {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.7, 5: 0.4, 6: 0.2, 7: 0.0, 8: -0.3},
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
        horse_3f_avg = float(row['上がり3F']) / 3.0
        pci = 50 + (race_3f_avg - horse_3f_avg) * 10
        pci_adj = (pci - 50) * 0.4
        
        lineage = str(row['血統'])
        l_bonus = 1.12 if (track_cond == '高速' and lineage == '米国型') or ((track_cond == 'タフ' or pace == 'H') and lineage == '欧州型') else 1.10 if (pace == 'S' and lineage == '日本型') else 1.0
        
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
    merged = pd.merge(df_raw, df_35, on='馬番')
    merged = pd.merge(merged, df_40, on='馬番')
    
    fusion_results = []
    for _, row in merged.iterrows():
        rank = row['V35_順位']
        hc = row['V40_ハンデ']
        
        if rank <= 2 and hc <= 1.0:
            grade, label, color, action = "S", "完全無欠の絶対軸", "#FFD700", "【勝負】厚め単複・連軸"
        elif rank <= 4 and hc == 0.0:
            grade, label, color, action = "A", "特大のオッズバグ", "#FF4B4B", "【妙味】単勝・ヒモ穴狙い"
        elif rank == 1 and hc >= 5.0:
            grade, label, color, action = "B", "危険な人気馬", "#9C27B0", "【警戒】投資見送り・ヒモまで"
        elif rank > 4 and hc > 4.0:
            grade, label, color, action = "C", "完全ノイズ", "#333333", "【消し】購入対象外"
        else:
            grade, label, color, action = "R", "連下候補 (レギュラー)", "#1E90FF", "【通常】相手候補として保持"
            
        fusion_results.append({
            '馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'],
            'V35_順位': rank, 'V40_ハンデ': hc, '判定': grade,
            'ステータス': label, '推奨アクション': action, 'color': color
        })
        
    sort_order = {"S": 1, "A": 2, "R": 3, "B": 4, "C": 5}
    df_fusion = pd.DataFrame(fusion_results)
    df_fusion['sort_key'] = df_fusion['判定'].map(sort_order)
    return df_fusion.sort_values(by=['sort_key', 'V40_ハンデ']).drop(columns=['sort_key']).reset_index(drop=True)

# ==========================================
# Web UI ＆ 演出機能
# ==========================================
st.set_page_config(page_title="Koder Master Fusion", layout="wide")

st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #0A0A0A; border: 4px solid #00FF00; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #0A0A0A; padding: 20px; border-bottom: 4px solid #00FF00; margin-bottom: 20px;'>
    <h1 style='color: #00FF00; margin: 0; font-family: monospace;'>MASTER FUSION ENGINE [AUTO-READY]</h1>
    <p style='color: #888; margin: 5px 0 0 0;'>〜Ver 3.5 × Ver 4.0 統合投資判定ダッシュボード〜</p>
</div>
""", unsafe_allow_html=True)

# 入力リセット用のセッション管理
if 'fusion_input' not in st.session_state:
    st.session_state.fusion_input = ""

def clear_text():
    st.session_state.fusion_input = ""

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🤖 1. Geminiへ指示文をコピー")
    st.info("下の文字をコピーして、出馬表の写メと一緒にGeminiに送信してください。")
    st.code("以下の出馬表画像を読み取り、統合解析用のCSVデータを作成してください。\n【必須項目】馬番, 馬名, 枠, ベーススコア, 上がり3F, 血統", language='text')

with col2:
    st.markdown("### 📥 2. データの直接貼り付け")
    st.markdown("""
<div style='border: 4px solid #00FF00; padding: 10px; border-radius: 10px 10px 0 0; background-color: rgba(0,255,0,0.05); text-align: center; border-bottom: none;'>
    <h4 style='color: #00FF00; margin:0;'>👇 ここにGeminiの解析結果を貼り付け 👇</h4>
</div>
    """, unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.fusion_input, height=150, key="fusion_area", label_visibility="collapsed", placeholder="馬番,馬名,枠,ベーススコア,上がり3F,血統\n1,馬名A,1,85.0,34.0,日本型\n...")
    
    if st.button("🔄 入力内容をクリア", on_click=clear_text, use_container_width=True):
        st.rerun()

with st.sidebar:
    st.header("⚙️ システム連動パラメータ")
    sel_course = st.selectbox("解析対象コース", list(COURSE_PHYSICS_DB.keys()))
    sel_cond = st.selectbox("馬場状態", ["標準", "高速", "タフ"])
    sel_pace = st.selectbox("展開予測", ["M", "S", "H"])
    sel_3f = st.number_input("前半3F想定タイム", value=35.5, step=0.1)
    st.markdown("---")
    speed_val = st.slider("想定秒速 (m/s) [V4.0用]", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m) [V4.0用]", value=2.4)

# ==========================================
# 🚀 実行＆スーパー特別演出
# ==========================================
if st.button("⚡ FUSION ENGINE 起動 (投資判定実行)", type="primary", use_container_width=True):
    if not pasted_data:
        st.error("解析データが貼り付けられていません！")
    else:
        try:
            df_raw = pd.read_csv(io.StringIO(pasted_data))
            df_35 = run_v35_logic(df_raw, sel_course, sel_cond, sel_pace, sel_3f)
            df_40 = run_v40_logic(df_raw, sel_course, speed_val, len_val)
            df_fusion = execute_fusion_matrix(df_raw, df_35, df_40)
            
            # S評価（完全無欠の絶対軸）がいれば大当たり確定演出！
            is_jackpot = False
            if len(df_fusion) > 0 and df_fusion.iloc[0]['判定'] == 'S':
                is_jackpot = True

            anim_stage = st.empty()
            
            # 演出1：旗振り
            anim_stage.markdown("""
<div class='stage-box'>
    <div style='text-align: center; color: #00FF00; font-size: 20px;'>[ SOUND ] 🎺 ピロリロリン♪ (G1 Fanfare)</div>
    <div class='staff-flag'><span style='font-size: 40px;'>🔴</span>👨‍💼<span style='font-size: 40px;'>🔴</span><br><small style='color: #888; font-size: 20px;'>--- 昇降台 ---</small></div>
</div>
            """, unsafe_allow_html=True)
            time.sleep(3)

            # 演出2：疾走
            horse_colors = ["🐎", "🐴", "🎠"]
            running_horse = random.choice(horse_colors)
            anim_stage.markdown(f"""
<div class='stage-box' style='background-color: #2e8b57;'>
    <div style='text-align: center; color: white; font-size: 20px;'>[ SOUND ] ⚡ ガシャン！！ (Gate Open)</div>
    <div class='horse-run'>{running_horse} 💨</div>
    <div style='text-align: center; color: white; margin-top: 50px; font-size: 24px; font-family: monospace;'>System Fusing... 3.5 x 4.0</div>
</div>
            """, unsafe_allow_html=True)
            time.sleep(6)
            
            # 演出3：大当たり（S評価降臨）
            if is_jackpot:
                anim_stage.markdown(f"""
<div class='stage-box' style='background-color: #2e8b57;'>
    <div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>[ SOUND ] 🚨 キュイン！キュイン！</div>
    <div class='horse-run'>{running_horse} 💨</div>
    <div class='cat-dash'>🐈‍⬛ 💨💨💨</div>
    <h2 style='text-align: center; color: yellow; text-shadow: 0 0 10px red;'>S評価・完全無欠の絶対軸 降臨！</h2>
</div>
                """, unsafe_allow_html=True)
                time.sleep(3)

            # 演出4：歓声
            anim_stage.markdown("""
<div class='stage-box' style='background-color: #8B0000;'>
    <div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>[ SOUND ] 🗣️ ウワーーーッ！！ (大歓声)</div>
    <div style='text-align: center; color: #FFD700; font-size: 50px; font-weight: bold; margin-top: 20px;'>FUSION COMPLETE!!!</div>
</div>
            """, unsafe_allow_html=True)
            time.sleep(2)
            anim_stage.empty()

            # リザルト表示
            st.markdown("<h2 style='color: #00FF00; margin-top: 10px; font-family: monospace; text-shadow: 0 0 10px rgba(0,255,0,0.5);'>🎯 最終投資判定マトリクス (自動資金配分ターゲット)</h2>", unsafe_allow_html=True)
            
            for _, row in df_fusion.iterrows():
                border_color = row['color']
                html_str = f"""
<div style='background-color: #0A0A0A; padding: 15px; border-radius: 5px; border-left: 10px solid {border_color}; border-bottom: 1px solid #222; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
    <div style='flex: 0.5; text-align: center;'>
        <h2 style='margin:0; color: {border_color}; font-family: monospace;'>{row['判定']}</h2>
    </div>
    <div style='flex: 2;'>
        <span style='color: #888;'>#{row['馬番']} [{row['枠']}枠]</span><br>
        <strong style='font-size: 1.2em; color: white;'>{row['馬名']}</strong>
    </div>
    <div style='flex: 1.5; background-color: #111; padding: 5px 10px; border-radius: 5px;'>
        <small style='color: #aaa;'>右目 (V3.5):</small> <b style='color: white;'>{row['V35_順位']}位</b><br>
        <small style='color: #aaa;'>左目 (V4.0):</small> <b style='color: white;'>{row['V40_ハンデ']} 馬身差</b>
    </div>
    <div style='flex: 2; text-align: right;'>
        <span style='color: {border_color}; font-weight: bold;'>{row['ステータス']}</span><br>
        <span style='background-color: #222; padding: 3px 8px; border-radius: 4px; font-size: 0.9em; color: #ddd;'>{row['推奨アクション']}</span>
    </div>
</div>
"""
                st.markdown(html_str, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"データの形式が正しくありません。貼り付けた内容を確認してください。\nエラー内容: {e}")
