import streamlit as st
import pandas as pd
import time
import random

# ==========================================
# 【Koder心臓部】JRA全施行コース別・枠順物理的タイム損益DB
# ==========================================
COURSE_PHYSICS_DB = {
    '東京_芝_2000m': {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 1.1, 8: 1.3},
    '東京_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.6, 8: 0.8},
    '中山_芝_2500m': {1: 0.0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.2, 8: 1.5},
    '京都_芝_1600m_外': {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.2, 7: 0.3, 8: 0.4},
    '阪神_芝_1600m': {1: 0.0, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.2, 6: 0.3, 7: 0.4, 8: 0.5},
    '新潟_芝_1000m': {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.7, 5: 0.4, 6: 0.2, 7: 0.0, 8: -0.3},
}

# ==========================================
# 演算エンジン (Ver 4.0)
# ==========================================
def run_koder_v40_logic(df, course_name, speed, length_unit):
    bias_map = COURSE_PHYSICS_DB.get(course_name, {i: 0.0 for i in range(1, 9)})
    df['能力タイム'] = 101.0 - (df['ベーススコア'].astype(float) * 0.1)
    
    results = []
    for _, row in df.iterrows():
        p_loss = bias_map.get(int(row['枠']), 0.0)
        final_time = row['能力タイム'] + p_loss
        results.append({
            '馬番': int(row['馬番']), '枠': int(row['枠']), '馬名': row['馬名'],
            '物理ロス(秒)': p_loss, '予測タイム': final_time
        })
        
    res_df = pd.DataFrame(results)
    anchor_time = res_df['予測タイム'].min()
    
    final_output = []
    for _, row in res_df.iterrows():
        t_diff = row['予測タイム'] - anchor_time
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
            '馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'],
            '必要ハンデ(馬身)': round(handicap, 1),
            '物理状況': f"{row['物理ロス(秒)']}秒のロスを含む",
            '判定': rating, 'color': color
        })
        
    return pd.DataFrame(final_output).sort_values(by='必要ハンデ(馬身)').reset_index(drop=True)

# ==========================================
# 演出用CSS ＆ アニメーション機能
# ==========================================
def inject_custom_css():
    # 空白判定エラーを防ぐため、CSSの記述を最適化
    st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #111; border: 4px solid #FF8C00; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
</style>
    """, unsafe_allow_html=True)

# ==========================================
# UI設定 ＆ レイアウト
# ==========================================
st.set_page_config(page_title="究極競馬場アプリ Ver 4.0", layout="wide")
inject_custom_css()

# タイトルの修正
st.markdown("""
<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 8px solid #FF4B4B; margin-bottom: 20px;'>
    <h1 style='color: #FF4B4B; margin: 0;'>究極競馬場アプリ Ver 4.0</h1>
    <p style='color: #888; margin: 5px 0 0 0;'>全馬同着からの逆算。真の実力差を「物理ハンデ（馬身）」で可視化するスーパーエンジン</p>
</div>
""", unsafe_allow_html=True)

# iPadで見やすいようにメイン画面に機能を集約！
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🤖 1. Geminiへ指示文をコピー")
    st.info("下の文字をコピーして、出馬表の写メと一緒にGeminiに送信してください。")
    st.code("以下の出馬表画像を読み取り、Ver 4.0解析用のCSVデータを作成してください。\n【必須項目】馬番, 馬名, 枠, ベーススコア", language='text')

with col2:
    st.markdown("### 📥 2. データの貼り付け")
    st.markdown("""
<div style='border: 4px solid #00FF41; padding: 15px; border-radius: 10px; background-color: rgba(0,255,65,0.05); text-align: center;'>
    <h4 style='color: #00FF41; margin-top:0;'>👇 ここにGeminiのCSVデータを投入 👇</h4>
</div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['csv'])

# 設定メニューはサイドバーに収納
with st.sidebar:
    st.header("⚙️ 詳細設定")
    target_course = st.selectbox("🎯 解析コース選択", list(COURSE_PHYSICS_DB.keys()))
    speed_val = st.slider("想定秒速 (m/s)", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m)", value=2.4)

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
else:
    df_input = pd.DataFrame([
        {'馬番': 9, '馬名': '能力1位/中枠', '枠': 4, 'ベーススコア': 85.0},
        {'馬番': 1, '馬名': '能力3位/最内', '枠': 1, 'ベーススコア': 82.0},
        {'馬番': 18, '馬名': '能力1位/大外', '枠': 8, 'ベーススコア': 85.0}
    ])

# ==========================================
# 🚀 実行＆夢のスーパー特別演出
# ==========================================
if st.button("🏇 究極スーパー特別演出・発走！", type="primary", use_container_width=True):
    df_final = run_koder_v40_logic(df_input, target_course, speed_val, len_val)
    
    is_jackpot = False
    if len(df_final) >= 2:
        diff = df_final.iloc[1]['必要ハンデ(馬身)'] - df_final.iloc[0]['必要ハンデ(馬身)']
        if diff >= 1.5:
            is_jackpot = True

    # --- 演出用コンテナ ---
    anim_stage = st.empty()
    
    # 演出1
    anim_stage.markdown("""
<div class='stage-box'>
    <div style='text-align: center; color: #FF8C00; font-size: 20px;'>[ SOUND ] 🎺 ピロリロリン♪ (G1 Fanfare)</div>
    <div class='staff-flag'>
        <span style='font-size: 40px;'>🔴</span>👨‍💼<span style='font-size: 40px;'>🔴</span><br>
        <small style='color: #888; font-size: 20px;'>--- 昇降台 ---</small>
    </div>
</div>
    """, unsafe_allow_html=True)
    time.sleep(3)

    # 演出2
    horse_colors = ["🐎", "🐴", "🎠"]
    running_horse = random.choice(horse_colors)
    anim_stage.markdown(f"""
<div class='stage-box' style='background-color: #2e8b57;'>
    <div style='text-align: center; color: white; font-size: 20px;'>[ SOUND ] ⚡ ガシャン！！ (Gate Open)</div>
    <div class='horse-run'>{running_horse} 💨</div>
    <div style='text-align: center; color: white; margin-top: 50px; font-size: 24px;'>Now Calculating...</div>
</div>
    """, unsafe_allow_html=True)
    time.sleep(6)
    
    # 演出3（大当たり）
    if is_jackpot:
        anim_stage.markdown(f"""
<div class='stage-box' style='background-color: #2e8b57;'>
    <div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>[ SOUND ] 🚨 キュイン！キュイン！</div>
    <div class='horse-run'>{running_horse} 💨</div>
    <div class='cat-dash'>🐈‍⬛ 💨💨💨</div>
    <h2 style='text-align: center; color: yellow; text-shadow: 0 0 10px red;'>超濃厚・大当たり確定演出発生！</h2>
</div>
        """, unsafe_allow_html=True)
        time.sleep(3)

    # 演出4（歓声）
    anim_stage.markdown("""
<div class='stage-box' style='background-color: #8B0000;'>
    <div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>[ SOUND ] 🗣️ ウワーーーッ！！ (大歓声)</div>
    <div style='text-align: center; color: #FFD700; font-size: 50px; font-weight: bold; margin-top: 20px;'>GOAL!!!</div>
</div>
    """, unsafe_allow_html=True)
    time.sleep(2)

    anim_stage.empty()

    # --- リザルト表示（電光掲示板） ---
    st.markdown("<h2 style='color: #FF8C00; margin-top: 10px; text-shadow: 0 0 10px rgba(255,140,0,0.5);'>📟 REKKA SCOREBOARD (LED)</h2>", unsafe_allow_html=True)
    
    # 暗号漏れを防ぐため、HTMLのインデントを完全に排除！
    for i, row in df_final.iterrows():
        border_color = row['color']
        html_str = f"""
<div style='background-color: #050505; padding: 20px; border-radius: 5px; border-left: 8px solid {border_color}; border-bottom: 2px solid #222; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;'>
    <div style='flex: 1.5;'>
        <span style='color: #00FF41; font-size: 1.2em;'>#{row['馬番']} [{row['枠']}枠]</span><br>
        <strong style='font-size: 1.6em; color: white;'>{row['馬名']}</strong>
    </div>
    <div style='flex: 2; text-align: center; background-color: #000; padding: 10px; border: 1px solid #111;'>
        <span style='color: {border_color}; font-weight: bold; font-size: 2.2em; text-shadow: 0 0 15px {border_color};'>
            +{row['必要ハンデ(馬身)']} 馬身
        </span><br>
        <small style='color: #444;'>({row['物理状況']})</small>
    </div>
    <div style='flex: 1; text-align: right;'>
        <span style='color: {border_color}; font-weight: bold; font-size: 1.2em;'>[{row['判定']}]</span>
    </div>
</div>
"""
        st.markdown(html_str, unsafe_allow_html=True)
