import streamlit as st
import pandas as pd
import time
import random
import io

def run_v35_core(df):
    results = []
    for _, row in df.iterrows():
        base = (100 - (float(row['オッズ']) * 0.5)) + (15 if str(row['亀谷ランク']).upper()=='A' else 5)
        buff = (float(row['騎手勝率']) * 0.3) + ((float(row['回収率']) - 100) * 0.1)
        score = base + buff - (float(row['枠バイアス']) * 10)
        results.append({'馬番': int(row['馬番']), 'V35スコア': score})
    res = pd.DataFrame(results).sort_values(by='V35スコア', ascending=False).reset_index(drop=True)
    res['V35順位'] = res.index + 1
    return res

def run_v40_core(df, speed=16.6, length=2.4):
    results = []
    for _, row in df.iterrows():
        base = (100 - (float(row['オッズ']) * 0.5)) + (15 if str(row['亀谷ランク']).upper()=='A' else 5)
        buff = (float(row['騎手勝率']) * 0.3) + ((float(row['回収率']) - 100) * 0.1)
        time_pred = (120.0 - ((base+buff) * 0.1)) + float(row['枠バイアス'])
        results.append({'馬番': int(row['馬番']), '予測タイム': time_pred})
    res = pd.DataFrame(results)
    anchor = res['予測タイム'].min()
    res['V40ハンデ'] = ((res['予測タイム'] - anchor) * speed) / length
    return res

def execute_fusion(df_raw, df_35, df_40):
    merged = pd.merge(df_raw, df_35, on='馬番')
    merged = pd.merge(merged, df_40, on='馬番')
    
    results = []
    for _, row in merged.iterrows():
        rank, hc = row['V35順位'], row['V40ハンデ']
        if rank <= 2 and hc <= 1.0:
            g, l, c, act = "S", "完全無欠の絶対軸", "#FFD700", "【単勝・複勝】厚め勝負 / 三連複の1列目"
        elif rank <= 4 and hc == 0.0:
            g, l, c, act = "A", "特大のオッズバグ", "#FF4B4B", "【単勝】妙味狙い / ワイドの軸"
        elif rank == 1 and hc >= 5.0:
            g, l, c, act = "B", "危険な人気馬", "#9C27B0", "【見送り】三連複のヒモまで"
        elif rank > 4 and hc > 4.0:
            g, l, c, act = "C", "完全ノイズ", "#333333", "【消し】購入対象外"
        else:
            g, l, c, act = "R", "連下・ヒモ候補", "#1E90FF", "【ワイド・三連複】相手候補として保持"
            
        results.append({'馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 'V35順位': rank, 'V40ハンデ': round(hc, 1)})
        
    order = {"S": 1, "A": 2, "R": 3, "B": 4, "C": 5}
    df_f = pd.DataFrame(results)
    df_f['sort'] = df_f['判定'].map(order)
    return df_f.sort_values(by=['sort', 'V40ハンデ']).drop(columns=['sort']).reset_index(drop=True)

st.set_page_config(page_title="Koder Master Fusion (8-Points)", layout="wide")
st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #0A0A0A; border: 4px solid #00FF00; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
.data-box { border: 6px solid #00FF00; padding: 15px; border-radius: 10px; background-color: rgba(0,255,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='background-color: #0A0A0A; padding: 20px; border-bottom: 4px solid #00FF00; margin-bottom: 20px;'><h1 style='color: #00FF00; margin: 0;'>MASTER FUSION [8項目・統合投資]</h1></div>", unsafe_allow_html=True)

if 'f_input' not in st.session_state: st.session_state.f_input = ""
def clear_text(): st.session_state.f_input = ""

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🤖 1. Gemini解析指示（右上のアイコンでワンタップコピー）")
    st.code("以下の6画像（netkeiba展開/位置, 亀谷出馬表/騎手ランク, JRA出馬表/オッズ）を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出して追加すること。\n【必須項目】馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス", language='text')

with col2:
    st.markdown("### 📥 2. 解析データ投入（太枠エリア）")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.f_input, height=150, key="f_area", label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,展開,ポジション,亀谷ランク,騎手勝率,回収率,枠バイアス\n...")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🗑️ データをオールクリア", on_click=clear_text, use_container_width=True): st.rerun()

if st.button("⚡ FUSION ENGINE 起動 (投資判定実行)", type="primary", use_container_width=True):
    if pasted_data:
        df_raw = pd.read_csv(io.StringIO(pasted_data))
        df_35 = run_v35_core(df_raw)
        df_40 = run_v40_core(df_raw)
        df_final = execute_fusion(df_raw, df_35, df_40)
        
        is_jackpot = len(df_final) > 0 and df_final.iloc[0]['判定'] == 'S'
        
        anim = st.empty()
        anim.markdown("<div class='stage-box'><div style='text-align: center; color: #00FF00; font-size: 20px;'>🎺 ピロリロリン♪</div><div class='staff-flag'>🔴👨‍💼🔴</div></div>", unsafe_allow_html=True)
        time.sleep(3)
        h_icon = random.choice(["🐎", "🐴", "🎠"])
        anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: white; font-size: 20px;'>⚡ ガシャン！！</div><div class='horse-run'>{h_icon} 💨</div></div>", unsafe_allow_html=True)
        time.sleep(6)
        if is_jackpot:
            anim.markdown(f"<div class='stage-box' style='background-color: #2e8b57;'><div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>🚨 キュイン！</div><div class='horse-run'>{h_icon} 💨</div><div class='cat-dash'>🐈‍⬛ 💨💨💨</div><h2 style='text-align: center; color: yellow;'>S評価・絶対軸 降臨！</h2></div>", unsafe_allow_html=True)
            time.sleep(3)
        anim.markdown("<div class='stage-box' style='background-color: #8B0000;'><div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>🗣️ ウワーーーッ！！</div><div style='text-align: center; color: #FFD700; font-size: 50px;'>FUSION COMPLETE!!!</div></div>", unsafe_allow_html=True)
        time.sleep(2); anim.empty()

        st.markdown("<h2 style='color: #00FF00;'>🎯 最終投資判定 (単・複・ワイド・三連複 連動)</h2>", unsafe_allow_html=True)
        for _, row in df_final.iterrows():
            st.markdown(f"<div style='background-color: #0A0A0A; padding: 15px; border-left: 10px solid {row['color']}; margin-bottom: 8px; display: flex;'><div style='flex: 0.5;'><h2 style='color: {row['color']};'>{row['判定']}</h2></div><div style='flex: 2;'><span style='color: #888;'>#{row['馬番']} [{row['枠']}枠]</span><br><strong style='color: white; font-size: 1.2em;'>{row['馬名']}</strong></div><div style='flex: 1.5;'><small style='color: #aaa;'>V3.5:</small> <b style='color: white;'>{row['V35順位']}位</b><br><small style='color: #aaa;'>V4.0:</small> <b style='color: white;'>{row['V40ハンデ']}馬身差</b></div><div style='flex: 2.5; text-align: right;'><span style='color: {row['color']}; font-weight: bold;'>{row['ステータス']}</span><br><span style='background-color: #222; padding: 3px 8px; color: #ddd; font-size: 0.9em;'>{row['推奨馬券']}</span></div></div>", unsafe_allow_html=True)
