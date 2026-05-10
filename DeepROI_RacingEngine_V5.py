import streamlit as st
import pandas as pd
import math

# ==========================================
# 1. 脳みそ：DeepROIEngineV6_Abyss クラス
# ==========================================
class DeepROIEngineV6_Abyss:
    def __init__(self, temperature=1.5):
        self.temperature = temperature

    def _sigmoid(self, x, steepness=2.0, center=0.0):
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def _calculate_moisture_resistance(self, surface_type, moisture_pct):
        if surface_type == 'TURF':
            if moisture_pct <= 10.0: return 1.0
            return 1.0 + 0.008 * (max(0, moisture_pct - 10.0) ** 1.2)
        elif surface_type == 'DIRT':
            return 1.0 - 0.05 * math.exp(-((moisture_pct - 15.0) ** 2) / 20.0)
        return 1.0

    def step1_engine_potential(self, horse, race_cond):
        target_resistance = self._calculate_moisture_resistance(race_cond['surface_type'], race_cond['target_moisture_pct'])
        past_resistance = horse.get('past_resistance', 1.0)
        track_modifier = target_resistance / past_resistance
        
        dist_ratio = race_cond['target_dist'] / horse['past_dist']
        weight_diff = horse.get('past_weight', 56.0) - horse.get('target_weight', 56.0)
        weight_time_impact = - (weight_diff * 0.15) * ((race_cond['target_dist'] / 1000.0) ** 1.1)
        
        est_time = (horse['past_time'] * track_modifier * (dist_ratio ** 1.04)) + weight_time_impact
        base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** 1.05)
        
        base_score = 100.0 - (est_time - base_time) * 3.0
        
        # エイジング
        current_age_months = horse.get('age_months', 60)
        past_age_months = horse.get('past_age_months', 58)
        peak_age = 54.0 
        aging_impact = (((past_age_months - peak_age)**2) - ((current_age_months - peak_age)**2)) * 0.015
        base_score += aging_impact 
        
        # ボラティリティ
        past_scores = horse.get('past_scores_array', [])
        if len(past_scores) >= 3:
            std_dev = math.sqrt(sum((x - (sum(past_scores)/len(past_scores)))**2 for x in past_scores)/len(past_scores))
            base_score += std_dev * 0.8
        
        # 上がり3F
        ppi = horse.get('past_pace_index', 1.0)
        diff = race_cond['target_3f_base'] - (horse['best_3f'] + (1.0 - ppi) * 2.0)
        base_score += 20.0 * self._sigmoid(diff, steepness=1.5, center=0.0)
        return base_score

    def step2_environmental_bias(self, score, horse, dynamic_pace, race_cond):
        adj_score = score
        draw = horse.get('draw', 8)
        if dynamic_pace == 'HIGH':
            adj_score += (15.0 if draw >= 11 else 10.0) if horse['pos_type'] == 'BACK' else (-15.0 if draw >= 11 else -8.0)
        elif dynamic_pace == 'SLOW':
            adj_score += (20.0 if draw <= 4 else 10.0) if horse['pos_type'] == 'FRONT' else -12.0
        return adj_score

    def analyze(self, horses, race_cond, jockey_roi):
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        dynamic_pace = 'HIGH' if front_runners >= 3 else ('SLOW' if front_runners == 0 else 'MIDDLE')
        raw_scores = [self.step2_environmental_bias(self.step1_engine_potential(h, race_cond), h, dynamic_pace, race_cond) for h in horses]
        
        max_s = max(raw_scores) if raw_scores else 0
        probs = [math.exp((s - max_s) / self.temperature) for s in raw_scores]
        sum_p = sum(probs)
        
        results = []
        for i, h in enumerate(horses):
            win_prob = probs[i] / sum_p
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            adj_roi = 1.0 + math.log1p(raw_roi - 1.0) * 0.4 if raw_roi > 1.0 else 1.0 - math.log1p(1.0 - raw_roi) * 0.5
            results.append({
                '枠': h.get('draw', '-'), '馬名': h['name'], 'スコア': round(raw_scores[i], 1),
                '勝率': f"{round(win_prob * 100, 2)}%", 'オッズ': h['odds'], '期待値(EV)': round(win_prob * h['odds'] * adj_roi, 3)
            })
        return {'pace': dynamic_pace, 'rankings': sorted(results, key=lambda x: x['期待値(EV)'], reverse=True)}

# ==========================================
# 2. 顔：Webインターフェース (Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="Abyss V6.0", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #E63946;'>競馬AI投資システム 深淵-Abyss- V6.0</h1>", unsafe_allow_html=True)

    # データ貼り付けエリア
    input_data = st.text_area("👀 AI抽出データをここに貼り付け 👀", height=150, placeholder="ここにデータを入力...")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 期待値(EV)解析を実行", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ データをクリア", use_container_width=True):
            st.rerun()

    if analyze_btn:
        if not input_data:
            st.error("データが空です")
        else:
            with st.spinner('深淵アルゴリズム解析中...'):
                engine = DeepROIEngineV6_Abyss()
                # --- テスト用のダミー実行（ここにデータパース処理を入れる） ---
                # ※本来は入力テキストを解析してリスト化しますが、一旦サンプルで動かします。
                h_data = [
                    {'id': 1, 'draw': 3, 'name': 'オルネーロ', 'past_dist': 1800, 'past_time': 108.8, 'past_resistance': 1.0, 'past_weight': 57.0, 'target_weight': 56.0, 'age_months': 48, 'past_age_months': 46, 'past_scores_array': [72, 88, 70], 'past_pace_index': 0.95, 'best_3f': 33.4, 'pos_type': 'MIDDLE', 'odds': 45.3, 'rank': 'E', 'jockey': '津村明秀'},
                    {'id': 2, 'draw': 14, 'name': 'ダイヤモンド', 'past_dist': 1400, 'past_time': 80.7, 'past_resistance': 1.05, 'past_weight': 55.0, 'target_weight': 56.0, 'age_months': 72, 'past_age_months': 70, 'past_scores_array': [85, 84, 86], 'past_pace_index': 1.08, 'best_3f': 34.2, 'pos_type': 'FRONT', 'odds': 4.4, 'rank': 'A', 'jockey': '川田将雅'}
                ]
                r_cond = {'target_dist': 1600, 'course_type': 'LONG_STRAIGHT', 'surface_type': 'TURF', 'target_moisture_pct': 12.0, 'base_time_1600': 94.0, 'target_3f_base': 33.8}
                j_roi = {'津村明秀': {'E': 1287}, '川田将雅': {'A': 65}}
                
                report = engine.analyze(h_data, r_cond, j_roi)
                
                st.markdown(f"### 展開予測: **{report['pace']}**")
                st.table(pd.DataFrame(report['rankings']))

if __name__ == "__main__":
    main()
