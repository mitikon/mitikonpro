import streamlit as st
import pandas as pd
import math
import io
import streamlit.components.v1 as components

# ==========================================
# 1. 脳みそ：DeepROIEngineV6_Abyss (完全復元)
# ==========================================
class DeepROIEngineV6_Abyss:
    def __init__(self, temperature=1.5):
        self.temperature = temperature

    def _calculate_moisture_resistance(self, surface_type, moisture_pct):
        if surface_type == 'TURF':
            return 1.0 if moisture_pct <= 10.0 else 1.0 + 0.008 * (max(0, moisture_pct - 10.0) ** 1.2)
        return 1.0 - 0.05 * math.exp(-((moisture_pct - 15.0) ** 2) / 20.0) if surface_type == 'DIRT' else 1.0

    def analyze(self, horses, race_cond, jockey_roi):
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        pace = 'HIGH' if front_runners >= 3 else ('SLOW' if front_runners == 0 else 'MIDDLE')
        
        raw_scores = []
        for h in horses:
            # 物理・生物・気象ロジック
            track_mod = self._calculate_moisture_resistance(race_cond['surface_type'], race_cond['target_moisture_pct']) / h.get('past_resistance', 1.0)
            dist_ratio = race_cond['target_dist'] / h['past_dist']
            weight_impact = -(h['past_weight'] - h['target_weight']) * 0.15 * ((race_cond['target_dist'] / 1000.0) ** 1.1)
            
            est_time = (h['past_time'] * track_mod * (dist_ratio ** 1.04)) + weight_impact
            base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** 1.05)
            score = 100.0 - (est_time - base_time) * 3.0
            
            # エイジング(54ヶ月ピーク)
            score += (((h['past_age_months'] - 54.0) ** 2) * 0.015) - (((h['age_months'] - 54.0) ** 2) * 0.015)
            
            # ボラティリティ
            if h.get('past_scores_array'):
                score += pd.Series(h['past_scores_array']).std() * 0.8
            
            raw_scores.append(score)

        max_s = max(raw_scores) if raw_scores else 0
        exp_s = [math.exp((s - max_s) / self.temperature) for s in raw_scores]
        sum_e = sum(exp_s)
        
        results = []
        for i, h in enumerate(horses):
            win_p = exp_s[i] / sum_e
            place_p = 1 - (1 - win_p)**2.8
            ev = win_p * h['odds'] * (1.0 + math.log1p(max(0, (jockey_roi.get(h['jockey'], 80)/100.0) - 1.0)) * 0.4)
            
            results.append({
                '枠': h['draw'], '馬番': h['id'], '馬名': h['name'], 'スコア': round(raw_scores[i], 1),
                '勝率': f"{round(win_p*100, 1)}%", '複勝率': f"{round(place_p*100, 1)}%",
                'オッズ': h['odds'], '期待値(EV)': round(ev, 3),
                'バグ': "🚩発生!!" if ev > 1.5 else "-", 'raw_place': place_p
            })
        return {'pace': pace, 'rankings': sorted(results, key=lambda x: x['期待値(EV)'], reverse=True),
                'place_top': [r['馬名'] for r in sorted(results, key=lambda x: x['raw_place'], reverse=True)[:3]]}

# ==========================================
# 2. UI：Ver 4.0 オーダー完全修復
# ==========================================
def main():
    st.set_page_config(page_title="Abyss V6.0", layout="wide")

    # CSS
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; }
        .stTextArea textarea { background-color: #000000 !important; color: #00FF41 !important; border: 2px solid #cc0000; font-size: 16px; }
        h1, h3 { color: #cc0000 !important; font-weight: bold; text-align: center; }
        div.stButton > button { 
            background-color: #cc0000 !important; color: white !important; border: none !important; 
            font-size: 20px !important; font-weight: bold !important; width: 100% !important; height: 60px !important; border-radius: 10px !important;
        }
        .clear-btn > div > button { background-color: #333333 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>競馬AI投資システム 深淵-Abyss- V6.0</h1>", unsafe_allow_html=True)

    # クリア機能用カウンター
    if 'count' not in st.session_state: st.session_state.count = 0

    # 1. 赤いボタンで即コピー (JavaScript連携)
    prompt_text = "あなたはプロの競馬アナリストです。以下の項目でCSVを出力してください：馬番,馬名,枠番,単勝オッズ,過去距離,過去走破タイム,過去馬場抵抗係数,過去斤量,今回斤量,年齢月換算,過去年齢月換算,過去スコア履歴,過去ペース係数,上がり3F,脚質,ポジションスコア,騎手名,騎手ランク,逃げ馬フラグ"
    
    st.markdown(f"""
        <script>
        function copyToClipboard() {{
            const text = `{prompt_text}`;
            navigator.clipboard.writeText(text).then(() => {{
                alert("コピー完了！Geminiに貼り付けてください。");
            }});
        }}
        </script>
        <button onclick="copyToClipboard()" style="background-color: #cc0000; color: white; border: none; font-size: 20px; font-weight: bold; width: 100%; height: 60px; border-radius: 10px; cursor: pointer; margin-bottom: 20px;">
            👁️ AI用データ解析指示をコピー
        </button>
    """, unsafe_allow_html=True)

    # 2. 貼り付けエリア
    st.markdown("<h3>👀 AI抽出データをここに貼り付け 👀</h3>", unsafe_allow_html=True)
    input_csv = st.text_area("", height=250, label_visibility="collapsed", key=f"input_{st.session_state.count}")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 期待値(EV)解析を実行")
    with col2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ データをクリア"):
            st.session_state.count += 1 # キーを変えて強制消去
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # サイドバー設定
    with st.sidebar:
        st.header("🏁 条件")
        r_dist = st.number_input("距離", 1600); r_surf = st.selectbox("馬場", ["TURF", "DIRT"])
        r_moist = st.slider("含水率", 0.0, 25.0, 12.0); r_base = st.number_input("基準T", 94.0); r_3f = st.number_input("基準3F", 33.8)

    # 3. 解析実行
    if analyze_btn:
        if not input_csv: st.error("データを入力してください")
        else:
            try:
                df = pd.read_csv(io.StringIO(input_csv.strip()))
                horses = []
                for _, r in df.iterrows():
                    hist = [float(s) for s in str(r['過去スコア履歴']).split('-') if s.replace('.','').isdigit()]
                    horses.append({
                        'id': r['馬番'], 'name': r['馬名'], 'draw': r['枠番'], 'odds': r['単勝オッズ'],
                        'past_dist': r['過去距離'], 'past_time': r['過去走破タイム'], 'past_resistance': r['過去馬場抵抗係数'],
                        'past_weight': r['過去斤量'], 'target_weight': r['今回斤量'], 'age_months': r['年齢月換算'],
                        'past_age_months': r['過去年齢月換算'], 'past_scores_array': hist, 'jockey': r['騎手名'], 'rank': r['騎手ランク'],
                        'best_3f': r['上がり3F'], 'pos_type': str(r['脚質']).upper(), 'pos_score': r['ポジションスコア'],
                        'is_front_runner': str(r['逃げ馬フラグ']).lower() == 'true'
                    })
                res = DeepROIEngineV6_Abyss().analyze(horses, {'target_dist': r_dist, 'surface_type': r_surf, 'target_moisture_pct': r_moist, 'course_type': 'LONG_STRAIGHT', 'base_time_1600': r_base, 'target_3f_base': r_3f}, {h['jockey']: {'A': 110, 'B': 100, 'C': 90, 'D': 80, 'E': 70}.get(h['rank'], 80) for h in horses})
                st.success(f"解析完了！ 展開: {res['pace']} / 候補: {', '.join(res['place_top'])}")
                st.table(pd.DataFrame(res['rankings']).drop(columns=['raw_place']))
            except Exception as e: st.error(f"エラー: CSV形式を確認してください。({e})")

if __name__ == "__main__": main()
