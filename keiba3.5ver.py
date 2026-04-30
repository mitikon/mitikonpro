import streamlit as st
import pandas as pd
import time
import random
import io

# ==========================================
# 8項目連動：Ver 3.5 確率・統計スコアリング
# ==========================================
def analyze_koder_v35_8points(df):
    results = []
    for _, row in df.iterrows():
        # 1. オッズと亀谷ランクからの基礎点
        odds = float(row['オッズ'])
        k_rank = str(row['亀谷ランク']).upper()
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        base_power = (100 - (odds * 0.5)) + rank_bonus
        
        # 2. 騎手成績（勝率・回収率）のバフ
        win_rate = float(row['騎手勝率'])
        recovery = float(row['回収率'])
        jockey_buff = (win_rate * 0.3) + ((recovery - 100) * 0.1)
        
        # 3. Gemini枠バイアス（秒を点数化：1秒ロス=10点減点）
        bias_sec = float(row['枠バイアス'])
        bias_adj = -(bias_sec * 10)
        
        # 総合スコア算出
        final_score = base_power + jockey_buff + bias_adj
        
        results.append({
            '馬番': int(row['馬番']), '枠': int(row['枠']), '馬名': str(row['馬名']),
            'オッズ': odds, '展開/位置': f"{row['展開']}/{row['ポジション']}",
            '騎手力': round(jockey_buff, 1), '👑 総合スコア': round(final_score, 2)
        })
    df_res = pd.DataFrame(results).sort_values(by='👑 総合スコア', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

# ==========================================
# UI ＆ 演出設定
# ==========================================
st.set_page_config(page_title="Koder V3.5 (8-Points)", layout="wide")
st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #0A0A0A; border: 4px solid #4CAF50; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
.data-box { border: 6px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: rgba(76,175,80,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='background-color: #1E1E1E; padding: 20px; border-left: 8px solid #4CAF50; margin-bottom: 20px;'><h1 style='color: #4CAF50; margin: 0;'>Koder : ENGINE Ver 3.5 [8項目・確率]</h1></div>", unsafe_allow_html=True)

if 'v35_input' not in st.session_state: st.session_state.v35_input = ""
def clear_text(): st.session_state.v35_input = ""

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🤖 1. Gemini解析指示（右上のアイコンでワンタップコピー）")
    st.code("以下の6画像（netkeiba展開/位置, 亀谷出馬表/騎手ランク, JRA出馬表/オッズ）を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出して追加すること。\n【必須項目】馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス", language='text')

with col2:
    st.markdown("### 📥 2. 解析データ投入（太枠エリア）")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.v35_input, height=150, key="v35_area", label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス\n...")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🗑️ データをオールクリア", on_click=clear_text, use_container_width=True): st.rerun()

if st.button("🚀 Koder V3.5 究極スーパー特別演出・発走！", type="primary", use_container_width=True):
    if pasted_data:
        df_raw = pd.read_csv(io.StringIO(pasted_data))
        df_final = analyze_koder_v35_8points(df_raw)
        is_jackpot = len(df_final) >= 2 and (df_final.iloc[0]['👑 総合スコア'] - df_final.iloc[1]['👑 総合スコア']) >= 5.0
        
        anim = st.empty()
        anim.markdown("<div class='stage-box'><div style='text-align: center; color: #4CAF50; font-size: 20px;'>🎺 ピロリロリン♪</div><div class='staff-flag'>🔴👨‍💼🔴</div></div>", unsafe_allow_html=True)
        time.sleep(3)
        h_icon = random.choice(["🐎", "🐴", "🎠"])
        anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: white; font-size: 20px;'>⚡ ガシャン！！</div><div class='horse-run'>{h_icon} 💨</div></div>", unsafe_allow_html=True)
        time.sleep(6)
        if is_jackpot:
            anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>🚨 キュイン！</div><div class='horse-run'>{h_icon} 💨</div><div class='cat-dash'>🐈‍⬛ 💨💨💨</div><h2 style='text-align: center; color: yellow;'>超濃厚・大当たり！</h2></div>", unsafe_allow_html=True)
            time.sleep(3)
        anim.markdown("<div class='stage-box' style='background-color: #8B0000;'><div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>🗣️ ウワーーーッ！！</div><div style='text-align: center; color: #FFD700; font-size: 50px;'>GOAL!!!</div></div>", unsafe_allow_html=True)
        time.sleep(2); anim.empty()

        st.markdown("<h2 style='color: #4CAF50;'>🎯 最終抽出ランキング (上位4頭)</h2>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (_, row) in enumerate(df_final.head(4).iterrows()):
            with cols[i]:
                st.markdown(f"<div style='background-color: #111; padding: 15px; border-top: 5px solid #4CAF50; text-align: center;'><h3 style='color: #aaa;'>{i+1}位 #{row['馬番']}</h3><h4 style='color: white;'>{row['馬名']}</h4><p style='font-size: 32px; color: #4CAF50; font-weight: bold;'>{row['👑 総合スコア']}</p><small style='color: #888;'>オッズ:{row['オッズ']} / {row['展開/位置']}</small></div>", unsafe_allow_html=True)
