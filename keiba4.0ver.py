import streamlit as st
import pandas as pd
import time
import io

def run_hybrid_v40(df, speed, length_unit):
    results = []
    for _, row in df.iterrows():
        tan_ret, fuku_ret = float(row['単回値']), float(row['複回値'])
        if tan_ret > 300: tan_ret = 80
        if fuku_ret > 300: fuku_ret = 70
        val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)

        spurt_bonus = 25 if (float(row['上がり3F順位']) <= 3.0 and float(row['ポジション評価']) >= 3.0) else 0
        odds_score = 100 - (float(row['オッズ']) * 0.5)
        k_rank = str(row['亀谷ランク']).upper()
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        
        base_power = odds_score + (val_score * 0.3) + rank_bonus + (float(row['騎手勝率']) * 0.3) + spurt_bonus
        
        # ハイブリッド能力をタイムへ逆算変換
        ability_time = 120.0 - (base_power * 0.1) 
        final_time = ability_time + float(row['枠バイアス(秒)'])
        results.append({'馬番': int(row['馬番']), '枠': int(row['枠']), '馬名': row['馬名'], '予測タイム': final_time})
        
    res_df = pd.DataFrame(results)
    anchor = res_df['予測タイム'].min()
    final_output = []
    for _, row in res_df.iterrows():
        handicap = ((row['予測タイム'] - anchor) * speed) / length_unit
        rating, color = ("👑 ANCHOR", "#D4AF37") if handicap == 0 else ("🔥 POTENTIAL", "#FF4B4B") if handicap <= 1.5 else ("📈 CHASER", "#1E90FF") if handicap <= 4.0 else ("💤 OUTSIDE", "#808080")
        final_output.append({'馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], '必要ハンデ(馬身)': round(handicap, 1), '判定': rating, 'color': color})
    return pd.DataFrame(final_output).sort_values(by='必要ハンデ(馬身)').reset_index(drop=True)

st.set_page_config(page_title="Koder V4.0 Hybrid", page_icon="🏇", layout="wide")
st.markdown("<style>.stage-box{height:350px;background:#111;border:4px solid #FF8C00;border-radius:10px;text-align:center;padding:20px;margin-bottom:20px;}.data-box{border:6px solid #FF8C00;padding:15px;border-radius:10px;background:rgba(255,140,0,0.05);}</style>", unsafe_allow_html=True)
st.markdown("<div style='background:#1a1a1a;padding:20px;border-left:8px solid #FF4B4B;'><h1 style='color:#FF4B4B;margin:0;'>Koder : ENGINE Ver 4.0 [ハイブリッド逆理論]</h1></div><br>", unsafe_allow_html=True)

if 'v40_input' not in st.session_state: st.session_state.v40_input = ""
def clear_text(): st.session_state.v40_input = ""

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🤖 1. Gemini解析指示（右上のアイコンでワンタップコピー）")
    st.code("以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位(1〜18),ポジション評価(1〜5),亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)", language='text')

with col2:
    st.markdown("### 📥 2. 解析データ投入（太枠エリア）")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.v40_input, height=150, label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\n...")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🗑️ データをオールクリア", on_click=clear_text, use_container_width=True): st.rerun()

with st.sidebar:
    speed_val = st.slider("想定秒速 (m/s)", 14.0, 18.0, 16.6)
    len_val = st.number_input("1馬身の定義 (m)", value=2.4)

if st.button("🏇 ハイブリッド逆理論演算・抽出を実行！", type="primary", use_container_width=True):
    if pasted_data:
        df_final = run_hybrid_v40(pd.read_csv(io.StringIO(pasted_data)), speed_val, len_val)
        anim = st.empty()
        anim.markdown("<div class='stage-box'><h2 style='color:#FF8C00;margin-top:100px;'>⚡ 物理ハンデ逆算中...</h2></div>", unsafe_allow_html=True)
        time.sleep(2); anim.empty()
        st.markdown("<h2 style='color:#FF8C00;'>📟 REKKA SCOREBOARD (LED)</h2>", unsafe_allow_html=True)
        for _, row in df_final.iterrows():
            st.markdown(f"<div style='background:#050505;padding:20px;border-left:8px solid {row['color']};margin-bottom:10px;display:flex;'><div style='flex:1.5;'><span style='color:#00FF41;'>#{row['馬番']} [{row['枠']}枠]</span><br><strong style='color:white;font-size:1.5em;'>{row['馬名']}</strong></div><div style='flex:2;text-align:center;'><span style='color:{row['color']};font-weight:bold;font-size:2em;'>+{row['必要ハンデ(馬身)']} 馬身</span></div></div>", unsafe_allow_html=True)
