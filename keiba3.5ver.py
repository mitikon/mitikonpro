import streamlit as st
import pandas as pd
import time
import random
import io

def analyze_hybrid_v35(df):
    results = []
    for _, row in df.iterrows():
        # ① ノイズカット＆期待値ベース
        tan_ret = float(row['単回値'])
        fuku_ret = float(row['複回値'])
        if tan_ret > 300: tan_ret = 80
        if fuku_ret > 300: fuku_ret = 70
        val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)

        # ② 末脚の真贋判定（ゴールデンゾーン）
        up3 = float(row['上がり3F順位'])
        pos = float(row['ポジション評価'])
        spurt_bonus = 25 if (up3 <= 3.0 and pos >= 3.0) else 0

        # ③ 8項目ベース（オッズ、亀谷、騎手）
        odds_score = 100 - (float(row['オッズ']) * 0.5)
        k_rank = str(row['亀谷ランク']).upper()
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        jockey_buff = float(row['騎手勝率']) * 0.3

        # ハイブリッド基礎能力
        base_power = odds_score + (val_score * 0.3) + rank_bonus + jockey_buff + spurt_bonus
        
        # 物理バイアス(秒)を減点（1秒=10点）
        bias_adj = -(float(row['枠バイアス(秒)']) * 10)
        final_score = base_power + bias_adj
        
        results.append({
            '馬番': int(row['馬番']), '馬名': str(row['馬名']),
            'オッズ': row['オッズ'], '真贋ボーナス': "+25" if spurt_bonus > 0 else "-",
            '👑 総合スコア': round(final_score, 2)
        })
    df_res = pd.DataFrame(results).sort_values(by='👑 総合スコア', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

st.set_page_config(page_title="Koder V3.5 Hybrid", page_icon="🐎", layout="wide")
st.markdown("<style>.stage-box{height:350px;background:#0A0A0A;border:4px solid #4CAF50;border-radius:10px;text-align:center;padding:20px;margin-bottom:20px;}.data-box{border:6px solid #4CAF50;padding:15px;border-radius:10px;background:rgba(76,175,80,0.05);}</style>", unsafe_allow_html=True)
st.markdown("<div style='background:#1E1E1E;padding:20px;border-left:8px solid #4CAF50;'><h1 style='color:#4CAF50;margin:0;'>Koder : ENGINE Ver 3.5 [ハイブリッド確率]</h1></div><br>", unsafe_allow_html=True)

if 'v35_input' not in st.session_state: st.session_state.v35_input = ""
def clear_text(): st.session_state.v35_input = ""

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🤖 1. Gemini解析指示（右上のアイコンでワンタップコピー）")
    st.code("以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位(1〜18),ポジション評価(1〜5),亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)", language='text')

with col2:
    st.markdown("### 📥 2. 解析データ投入（太枠エリア）")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.v35_input, height=150, label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\n...")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🗑️ データをオールクリア", on_click=clear_text, use_container_width=True): st.rerun()

if st.button("🚀 ハイブリッド Ver 3.5 抽出エンジン発走！", type="primary", use_container_width=True):
    if pasted_data:
        df_final = analyze_hybrid_v35(pd.read_csv(io.StringIO(pasted_data)))
        anim = st.empty()
        anim.markdown("<div class='stage-box'><h2 style='color:#4CAF50;margin-top:100px;'>🎺 解析＆真贋判定中...</h2></div>", unsafe_allow_html=True)
        time.sleep(2); anim.empty()
        st.markdown("<h2 style='color:#4CAF50;'>🎯 最終抽出ランキング (上位4頭)</h2>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (_, row) in enumerate(df_final.head(4).iterrows()):
            with cols[i]:
                st.markdown(f"<div style='background:#111;padding:15px;border-top:5px solid #4CAF50;text-align:center;'><h3 style='color:#aaa;'>{i+1}位 #{row['馬番']}</h3><h4 style='color:white;'>{row['馬名']}</h4><p style='font-size:32px;color:#4CAF50;font-weight:bold;'>{row['👑 総合スコア']}</p><small style='color:#888;'>オッズ:{row['オッズ']} / 真贋:{row['真贋ボーナス']}</small></div>", unsafe_allow_html=True)
