import streamlit as st
import pandas as pd
import time
import random
import io

# ==========================================
# 8項目連動：Ver 4.0 物理逆算エンジン
# ==========================================
def run_koder_v40_8points(df, speed, length_unit):
    results = []
    for _, row in df.iterrows():
        odds = float(row['オッズ'])
        k_rank = str(row['亀谷ランク']).upper()
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        jockey_buff = (float(row['騎手勝率']) * 0.3) + ((float(row['回収率']) - 100) * 0.1)
        
        # 8項目の力を「能力タイム」へ変換（逆理論）
        base_power = (100 - (odds * 0.5)) + rank_bonus + jockey_buff
        ability_time = 120.0 - (base_power * 0.1) 
        
        # 物理バイアス(秒)を加算して最終到達タイムを予測
        p_loss = float(row['枠バイアス'])
        final_time = ability_time + p_loss
        results.append({'馬番': int(row['馬番']), '枠': int(row['枠']), '馬名': row['馬名'], '予測タイム': final_time})
        
    res_df = pd.DataFrame(results)
    anchor_time = res_df['予測タイム'].min()
    
    final_output = []
    for _, row in res_df.iterrows():
        # 同時入着するために必要なハンデ(馬身)を逆算
        t_diff = row['予測タイム'] - anchor_time
        handicap = (t_diff * speed) / length_unit
        
        if handicap == 0: rating, color = "👑 ANCHOR", "#D4AF37"
        elif handicap <= 1.5: rating, color = "🔥 POTENTIAL", "#FF4B4B"
        elif handicap <= 4.0: rating, color = "📈 CHASER", "#1E90FF"
        else: rating, color = "💤 OUTSIDE", "#808080"
            
        final_output.append({'馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], '必要ハンデ(馬身)': round(handicap, 1), '判定': rating, 'color': color})
    return pd.DataFrame(final_output).sort_values(by='必要ハンデ(馬身)').reset_index(drop=True)

st.set_page_config(page_title="Koder V4.0 (8-Points)", layout="wide")
st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #111; border: 4px solid #FF8C00; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
.data-box { border: 6px solid #FF8C00; padding: 15px; border-radius: 10px; background-color: rgba(255,140,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='background-color: #1a1a1a; padding: 20px; border-left: 8px solid #FF4B4B; margin-bottom: 20px;'><h1 style='color: #FF4B4B; margin: 0;'>Koder : ENGINE Ver 4.0 [8項目・逆理論]</h1></div>", unsafe_allow_html=True)

if 'v40_input' not in st.session_state: st.session_state.v40_input = ""
def clear_text(): st.session_state.v40_input = ""

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🤖 1. Gemini解析指示（右上のアイコンでワンタップコピー）")
    st.code("以下の6画像（netkeiba展開/位置, 亀谷出馬表/騎手ランク, JRA出馬表/オッズ）を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出して追加すること。\n【必須項目】馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス", language='text')

with col2:
    st.markdown("### 📥 2. 解析データ投入（太枠エリア）")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.v40_input, height=150, key="v40_area", label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス\n...")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🗑️ データをオールクリア", on_click=clear_text, use_container_width=True): st.rerun()

with st.sidebar:
    st.header("⚙️ 逆理論パラメータ")
    speed_val = st.slider("想定秒速 (m/s)", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m)", value=2.4)

if st.button("🏇 究極スーパー特別演出・発走！", type="primary", use_container_width=True):
    if pasted_data:
        df_input = pd.read_csv(io.StringIO(pasted_data))
        df_final = run_koder_v40_8points(df_input, speed_val, len_val)
        is_jackpot = len(df_final) >= 2 and (df_final.iloc[1]['必要ハンデ(馬身)'] - df_final.iloc[0]['必要ハンデ(馬身)']) >= 1.5
        
        anim = st.empty()
        anim.markdown("<div class='stage-box'><div style='text-align: center; color: #FF8C00; font-size: 20px;'>🎺 ピロリロリン♪</div><div class='staff-flag'>🔴👨‍💼🔴</div></div>", unsafe_allow_html=True)
        time.sleep(3)
        h_icon = random.choice(["🐎", "🐴", "🎠"])
        anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: white; font-size: 20px;'>⚡ ガシャン！！</div><div class='horse-run'>{h_icon} 💨</div></div>", unsafe_allow_html=True)
        time.sleep(6)
        if is_jackpot:
            anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>🚨 キュイン！</div><div class='horse-run'>{h_icon} 💨</div><div class='cat-dash'>🐈‍⬛ 💨💨💨</div><h2 style='text-align: center; color: yellow;'>大当たり確定！</h2></div>", unsafe_allow_html=True)
            time.sleep(3)
        anim.markdown("<div class='stage-box' style='background-color: #8B0000;'><div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>🗣️ ウワーーーッ！！</div><div style='text-align: center; color: #FFD700; font-size: 50px;'>GOAL!!!</div></div>", unsafe_allow_html=True)
        time.sleep(2); anim.empty()

        st.markdown("<h2 style='color: #FF8C00;'>📟 REKKA SCOREBOARD (LED)</h2>", unsafe_allow_html=True)
        for _, row in df_final.iterrows():
            st.markdown(f"<div style='background-color: #050505; padding: 20px; border-left: 8px solid {row['color']}; margin-bottom: 10px; display: flex;'><div style='flex: 1.5;'><span style='color: #00FF41;'>#{row['馬番']} [{row['枠']}枠]</span><br><strong style='color: white; font-size:1.5em;'>{row['馬名']}</strong></div><div style='flex: 2; text-align: center;'><span style='color: {row['color']}; font-weight: bold; font-size: 2em;'>+{row['必要ハンデ(馬身)']} 馬身</span></div></div>", unsafe_allow_html=True)
