import streamlit as st
import pandas as pd
import math
import io

# ==========================================
# 1. 脳みそ：DeepROIEngineV6_Abyss (全ロジック完全復元)
# ==========================================
class DeepROIEngineV6_Abyss:
    def __init__(self, temperature=1.5):
        self.temperature = temperature

    def _sigmoid(self, x, steepness=2.0, center=0.0):
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def _calculate_moisture_resistance(self, surface_type, moisture_pct):
        """【最重要】気象学：芝とダートで異なる物理摩擦モデル"""
        if surface_type == 'TURF':
            if moisture_pct <= 10.0: return 1.0
            return 1.0 + 0.008 * (max(0, moisture_pct - 10.0) ** 1.2)
        elif surface_type == 'DIRT':
            return 1.0 - 0.05 * math.exp(-((moisture_pct - 15.0) ** 2) / 20.0)
        return 1.0

    def analyze(self, horses, race_cond, jockey_roi_master):
        # 展開予測（逃げ馬数による動的判定）
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        if front_runners >= 3: dynamic_pace = 'HIGH'
        elif front_runners == 0: dynamic_pace = 'SLOW'
        else: dynamic_pace = 'MIDDLE'

        raw_scores = []
        for h in horses:
            # --- step1: 物理・生物・気象の統合 ---
            target_res = self._calculate_moisture_resistance(race_cond['surface_type'], race_cond['target_moisture_pct'])
            past_res = h.get('past_resistance', 1.0)
            track_modifier = target_res / past_res
            
            dist_ratio = race_cond['target_dist'] / h['past_dist']
            # 物理：斤量影響は距離の1.1乗で増幅
            weight_diff = h['past_weight'] - h['target_weight']
            weight_time_impact = - (weight_diff * 0.15) * ((race_cond['target_dist'] / 1000.0) ** 1.1)
            
            # 非線形タイム予測 (1.04/1.05乗モデル)
            est_time = (h['past_time'] * track_modifier * (dist_ratio ** 1.04)) + weight_time_impact
            base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** 1.05)
            
            score = 100.0 - (est_time - base_time) * 3.0
            
            # 生物：エイジング（54ヶ月ピークの二次曲線）
            aging_impact = (((h['past_age_months'] - 54.0) ** 2) * 0.015) - (((h['age_months'] - 54.0) ** 2) * 0.015)
            score += aging_impact 
            
            # 確率：ボラティリティ（過去スコアの標準偏差）
            if h.get('past_scores_array') and len(h['past_scores_array']) >= 2:
                std_dev = pd.Series(h['past_scores_array']).std()
                score += std_dev * 0.8
            
            # 直線適性補正
            if race_cond['course_type'] == 'LONG_STRAIGHT':
                adj_3f = h['best_3f'] + (1.0 - h.get('past_pace_index', 1.0)) * 2.0
                score += 20.0 * self._sigmoid(race_cond['target_3f_base'] - adj_3f, steepness=1.5)

            # --- step2: 環境バイアス ---
            draw = h.get('draw', 8)
            if dynamic_pace == 'HIGH':
                if h['pos_type'] == 'FRONT': score -= 15.0 if draw >= 11 else 8.0 
                elif h['pos_type'] == 'BACK': score += 15.0 if draw >= 11 else 10.0
            elif dynamic_pace == 'SLOW':
                if h['pos_type'] == 'FRONT': score += 20.0 if draw <= 4 else 10.0
                elif h['pos_type'] == 'BACK': score -= 12.0 
            
            score += h.get('pos_score', 0) * (0.5 if race_cond['course_type'] == 'LONG_STRAIGHT' else 2.0)
            raw_scores.append(score)

        # Softmax & 期待値算出
        max_s = max(raw_scores) if raw_scores else 0
        exp_s = [math.exp((s - max_s) / self.temperature) for s in raw_scores]
        sum_e = sum(exp_s)
        
        results = []
        for i, h in enumerate(horses):
            win_p = exp_s[i] / sum_e
            place_p = 1 - (1 - win_p)**2.8
            raw_roi = jockey_roi_master.get(h['jockey'], 80) / 100.0
            adj_roi = (1.0 + math.log1p(max(0, raw_roi - 1.0)) * 0.4) if raw_roi > 1.0 else (1.0 - math.log1p(max(0, 1.0 - raw_roi)) * 0.5)
            ev = win_p * h['odds'] * adj_roi
            
            results.append({
                '枠': h['draw'], '馬番': h['id'], '馬名': h['name'], 'AIスコア': round(raw_scores[i], 1),
                '勝率': f"{round(win_p*100, 1)}%", '複勝率': f"{round(place_p*100, 1)}%",
                'オッズ': h['odds'], '期待値(EV)': round(ev, 3),
                'バグ': "🚩発生!!" if ev > 1.5 else "-", 'raw_place': place_p
            })
            
        return {
            'pace': dynamic_pace, 
            'rankings': sorted(results, key=lambda x: x['期待値(EV)'], reverse=True),
            'place_top': [r['馬名'] for r in sorted(results, key=lambda x: x['raw_place'], reverse=True)[:3]]
        }

# ==========================================
# 2. 顔：UIレイアウト (Ver 4.0 完全修復オーダー)
# ==========================================
def main():
    st.set_page_config(page_title="Abyss V6.0", layout="wide")

    # CSS: オーダー通りの配色設定
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; }
        .stTextArea textarea { background-color: #1A1A1A !important; color: #00FF41 !important; border: 2px solid #cc0000; border-radius: 8px; font-weight: bold; }
        h1, h3, label { color: #cc0000 !important; font-weight: bold; text-align: center; }
        div.stButton > button:first-child { 
            background-color: #cc0000; color: white; border: none; font-size: 20px; font-weight: bold; width: 100%; height: 65px; border-radius: 12px;
        }
        div.stButton > button:hover { background-color: #ff3333; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>競馬AI投資システム 深淵-Abyss- V6.0</h1>", unsafe_allow_html=True)

    # セッション状態管理（クリア機能用）
    if 'input_box' not in st.session_state: st.session_state.input_box = ""

    # --- 1. 指示コピーボタン ---
    ai_prompt = "あなたはプロの競馬アナリストです。以下のヘッダーでCSVを出力してください：\n馬番,馬名,枠番,単勝オッズ,過去距離,過去走破タイム,過去馬場抵抗係数,過去斤量,今回斤量,年齢月換算,過去年齢月換算,過去スコア履歴,過去ペース係数,上がり3F,脚質,ポジションスコア,騎手名,騎手ランク,逃げ馬フラグ"
    
    if st.button("👁️ AI用データ解析指示 (11項目) をコピー"):
        st.info("▼ 以下のテキストをコピーしてGeminiに送信してください")
        st.code(ai_prompt, language="text")

    # --- 2. データ貼り付けエリア ---
    st.markdown("<h3>👀 AI抽出データをここに貼り付け 👀</h3>", unsafe_allow_html=True)
    # セッション状態と連動させたテキストエリア
    input_data = st.text_area("", value=st.session_state.input_box, height=250, label_visibility="collapsed", key="data_input")

    col1, col2 = st.columns(2)
    with col1:
        run_analyze = st.button("🚀 期待値(EV)解析を実行")
    with col2:
        if st.button("🗑️ データオールクリア"):
            st.session_state.input_box = "" # 値を空にする
            st.rerun()

    # サイドバー（非表示でも可）
    with st.sidebar:
        st.header("🏁 レース設定")
        r_dist = st.number_input("距離", 1600)
        r_surf = st.selectbox("馬場", ["TURF", "DIRT"])
        r_moist = st.slider("含水率(%)", 0.0, 25.0, 12.0)
        r_base = st.number_input("基準タイム", 94.0)
        r_3f = st.number_input("基準上がり", 33.8)

    # --- 3. 解析実行 ---
    if run_analyze:
        # ボタンが押された時の入力を取得
        actual_input = st.session_state.data_input if st.session_state.data_input else input_data
        if not actual_input:
            st.error("データを貼り付けてください")
        else:
            try:
                df = pd.read_csv(io.StringIO(actual_input.strip()))
                horse_list = []
                for _, r in df.iterrows():
                    s_history = [float(s) for s in str(r['過去スコア履歴']).split('-') if s.replace('.','').isdigit()]
                    horse_list.append({
                        'id': r['馬番'], 'name': r['馬名'], 'draw': r['枠番'], 'odds': r['単勝オッズ'],
                        'past_dist': r['過去距離'], 'past_time': r['過去走破タイム'], 'past_resistance': r['過去馬場抵抗係数'],
                        'past_weight': r['過去斤量'], 'target_weight': r['今回斤量'], 'age_months': r['年齢月換算'],
                        'past_age_months': r['過去年齢月換算'], 'past_scores_array': s_history, 'past_pace_index': r['過去ペース係数'],
                        'best_3f': r['上がり3F'], 'pos_type': str(r['脚質']).upper(), 'pos_score': r['ポジションスコア'],
                        'jockey': r['騎手名'], 'rank': r['騎手ランク'], 'is_front_runner': str(r['逃げ馬フラグ']).lower() == 'true'
                    })
                
                engine = DeepROIEngineV6_Abyss()
                race_cond = {'target_dist': r_dist, 'surface_type': r_surf, 'target_moisture_pct': r_moist, 
                             'course_type': 'LONG_STRAIGHT', 'base_time_1600': r_base, 'target_3f_base': r_3f}
                j_master = {h['jockey']: {'A': 110, 'B': 100, 'C': 90, 'D': 80, 'E': 70}.get(h['rank'], 80) for h in horse_list}
                
                result = engine.analyze(horse_list, race_cond, j_master)
                
                st.success(f"解析完了！ 展開予測: {result['pace']}ペース")
                st.warning(f"🥈 複勝圏内 有力候補: {', '.join(result['place_top'])}")
                st.table(pd.DataFrame(result['rankings']).drop(columns=['raw_place'], errors='ignore'))
                
            except Exception as e:
                st.error(f"解析失敗。CSVの形式を確認してください。: {e}")

if __name__ == "__main__":
    main()
