import streamlit as st
import pandas as pd
import time
import random
import io

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
        
    df_res = pd.DataFrame(results).sort_values(by='👑 総合スコア', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

# ==========================================
# Web UI ＆ 演出機能
# ==========================================
st.set_page_config(page_title="Koder V3.5", layout="wide")

st.markdown("""
<style>
@keyframes slideUp { from { transform: translateY(100px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
@keyframes runHorse { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
@keyframes runCat { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.staff-flag { animation: slideUp 1s ease-out forwards; font-size: 80px; text-align: center; margin-top: 30px; }
.horse-run { animation: runHorse 3s linear infinite; font-size: 100px; white-space: nowrap; margin-top: 20px;}
.cat-dash { animation: runCat 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; font-size: 80px; position: absolute; bottom: 20px; z-index: 100;}
.stage-box { height: 350px; background-color: #0A0A0A; border: 4px solid #4CAF50; border-radius: 10px; position: relative; overflow: hidden; padding: 20px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #1E1E1E; padding: 20px; border-left: 8px solid #4CAF50; border-radius: 5px; margin-bottom: 20px;'>
    <h1 style='color: #4CAF50; margin: 0;'>Koder : ENGINE Ver 3.5 [確率]</h1>
    <p style='color: #DDDDDD; margin: 5px 0 0 0;'>全頭フィルター・高精度4頭抽出コアエンジン（物理統計・アヤの評価）</p>
</div>
""", unsafe_allow_html=True)

# 入力リセット用のセッション管理
if 'v35_input' not in st.session_state:
    st.session_state.v35_input = ""

def clear_text():
    st.session_state.v35_input = ""

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🤖 1. Geminiへ指示文をコピー")
    st.info("下の文字をコピーして、出馬表の写メと一緒にGeminiに送信してください。")
    st.code("以下の出馬表画像を読み取り、Ver 3.5解析用のCSVデータを作成してください。\n【必須項目】馬番, 馬名, 枠, ベーススコア, 上がり3F, 血統", language='text')

with col2:
    st.markdown("### 📥 2. データの直接貼り付け")
    st.markdown("""
<div style='border: 4px solid #4CAF50; padding: 10px; border-radius: 10px 10px 0 0; background-color: rgba(76,175,80,0.05); text-align: center; border-bottom: none;'>
    <h4 style='color: #4CAF50; margin:0;'>👇 ここにGeminiの解析結果を貼り付け 👇</h4>
</div>
    """, unsafe_allow_html=True)
    pasted_data = st.text_area("", value=st.session_state.v35_input, height=150, key="v35_area", label_visibility="collapsed", placeholder="馬番,馬名,枠,ベーススコア,上がり3F,血統\n1,馬名A,1,85.0,34.0,日本型\n...")
    
    if st.button("🔄 入力内容をクリア", on_click=clear_text, use_container_width=True):
        st.rerun()

with st.sidebar:
    st.header("⚙️ レース環境設定")
    selected_course = st.selectbox("コース選択 (物理バイアス用)", list(FRAME_BIAS_MASTER.keys()))
    selected_cond = st.selectbox("馬場状態", ["標準", "高速", "タフ"])
    selected_pace = st.selectbox("展開予測", ["M", "S", "H"])
    input_3f_time = st.number_input("前半3F想定タイム (秒)", value=35.5, step=0.1)

# ==========================================
# 🚀 解析実行 ＆ スーパー特別演出
# ==========================================
if st.button("🚀 Koder V3.5 究極スーパー特別演出・発走！", type="primary", use_container_width=True):
    if not pasted_data:
        st.error("解析データが貼り付けられていません！")
    else:
        try:
            df_raw = pd.read_csv(io.StringIO(pasted_data))
            df_final = analyze_koder_v35(df_raw, selected_course, selected_cond, selected_pace, input_3f_time)
            
            # 大当たり確定判定：1位と2位のスコア差が3.0以上
            is_jackpot = False
            if len(df_final) >= 2:
                diff = df_final.iloc[0]['👑 総合スコア'] - df_final.iloc[1]['👑 総合スコア']
                if diff >= 3.0:
                    is_jackpot = True

            anim_stage = st.empty()
            
            # 演出1：旗振り
            anim_stage.markdown("""
<div class='stage-box'>
    <div style='text-align: center; color: #4CAF50; font-size: 20px;'>[ SOUND ] 🎺 ピロリロリン♪ (G1 Fanfare)</div>
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
    <div style='text-align: center; color: white; margin-top: 50px; font-size: 24px;'>Filtering Data... V3.5</div>
</div>
            """, unsafe_allow_html=True)
            time.sleep(6)
            
            # 演出3：大当たり（黒猫）
            if is_jackpot:
                anim_stage.markdown(f"""
<div class='stage-box' style='background-color: #2e8b57;'>
    <div style='text-align: center; color: #FF00FF; font-size: 24px; font-weight: bold;'>[ SOUND ] 🚨 キュイン！キュイン！</div>
    <div class='horse-run'>{running_horse} 💨</div>
    <div class='cat-dash'>🐈‍⬛ 💨💨💨</div>
    <h2 style='text-align: center; color: yellow; text-shadow: 0 0 10px red;'>超濃厚・スコア一強確定演出！</h2>
</div>
                """, unsafe_allow_html=True)
                time.sleep(3)

            # 演出4：歓声
            anim_stage.markdown("""
<div class='stage-box' style='background-color: #8B0000;'>
    <div style='text-align: center; color: white; font-size: 30px; margin-top: 80px;'>[ SOUND ] 🗣️ ウワーーーッ！！ (大歓声)</div>
    <div style='text-align: center; color: #FFD700; font-size: 50px; font-weight: bold; margin-top: 20px;'>GOAL!!!</div>
</div>
            """, unsafe_allow_html=True)
            time.sleep(2)
            anim_stage.empty()

            # --- リザルト表示（LED電光掲示板スタイル） ---
            st.markdown("<h2 style='color: #4CAF50; margin-top: 10px; text-shadow: 0 0 10px rgba(76,175,80,0.5);'>🎯 最終抽出ランキング (上位4頭コア・ポートフォリオ)</h2>", unsafe_allow_html=True)
            
            top4 = df_final.head(4)
            cols = st.columns(4)
            for i, (idx, row) in enumerate(top4.iterrows()):
                with cols[i]:
                    html_card = f"""
<div style='background-color: #111; padding: 15px; border-radius: 8px; border: 2px solid #222; border-top: 5px solid #4CAF50; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.5);'>
    <h3 style='margin:0; color: #aaa; font-size: 16px;'>{i+1}位 #{row['馬番']}</h3>
    <h4 style='margin:10px 0; color: #ffffff; font-size: 20px;'>{row['馬名']}</h4>
    <p style='margin:0; font-size: 32px; color: #4CAF50; font-weight: bold; font-family: monospace; text-shadow: 0 0 10px rgba(76,175,80,0.5);'>{row['👑 総合スコア']}</p>
    <small style='color: #666;'>枠:{row['枠']} / PCI:{row['PCI粘り']}</small>
</div>
"""
                    st.markdown(html_card, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 全頭解析マトリクス詳細")
            styled_df = df_final.style.background_gradient(cmap='Greens', subset=['👑 総合スコア', '出汁(ベース)', 'PCI粘り'])
            st.dataframe(styled_df, use_container_width=True, height=400)

        except Exception as e:
            st.error(f"データの形式が正しくありません。貼り付けた内容を確認してください。\nエラー内容: {e}")
