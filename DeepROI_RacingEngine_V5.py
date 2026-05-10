import streamlit as st
import pandas as pd
import math
import io

# ==========================================
# 1. 脳みそ：DeepROIEngineV6_Abyss クラス（計算ロジック）
# ==========================================
class DeepROIEngineV6_Abyss:
    def __init__(self, ev_threshold=1.0, temperature=1.5):
        self.ev_threshold = ev_threshold
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
        alpha_horse, alpha_base = 1.04, 1.05  
        
        weight_diff = horse.get('past_weight', 56.0) - horse.get('target_weight', 56.0)
        weight_time_impact = - (weight_diff * 0.15) * ((race_cond['target_dist'] / 1000.0) ** 1.1)
        
        est_time = (horse['past_time'] * track_modifier * (dist_ratio ** alpha_horse)) + weight_time_impact
        base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** alpha_base)
        
        base_score = 100.0 - (est_time - base_time) * 3.0
        
        current_age_months = horse.get('age_months', 60)
        past_age_months = horse.get('past_age_months', 58)
        peak_age = 54.0 
        aging_impact = (((past_age_months - peak_age) ** 2) * 0.015) - (((current_age_months - peak_age) ** 2) * 0.015)
        base_score += aging_impact 
        
        past_scores = horse.get('past_scores_array', [])
        if len(past_scores) >= 3:
            mean_s = sum(past_scores) / len(past_scores)
            variance = sum((x - mean_s) ** 2 for x in past_scores) / len(past_scores)
            std_dev = math.sqrt(variance)
            base_score += std_dev * 0.8 # リスク選好度
        
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            ppi = horse.get('past_pace_index', 1.0)
            adjusted_3f = horse['best_3f'] + (1.0 - ppi) * 2.0
            diff = race_cond['target_3f_base'] - adjusted_3f
            base_score += 20.0 * self._sigmoid(diff, steepness=1.5, center=0.0)
            
        return base_score

    def step2_environmental_bias(self, score, horse, dynamic_pace, race_cond):
        adj_score = score
        draw = horse.get('draw', 8)
        
        if dynamic_pace == 'HIGH':
            if horse['pos_type'] == 'FRONT': adj_score -= 15.0 if draw >= 11 else 8.0 
            elif horse['pos_type'] == 'BACK': adj_score += 15.0 if draw >= 11 else 10.0
        elif dynamic_pace == 'SLOW':
            if horse['pos_type'] == 'FRONT': adj_score += 20.0 if draw <= 4 else 10.0
            elif horse['pos_type'] == 'BACK': adj_score -= 12.0 
        elif dynamic_pace == 'MIDDLE':
            if horse['pos_type'] == 'FRONT': adj_score += 5.0
            
        weight = 0.5 if race_cond['course_type'] == 'LONG_STRAIGHT' else 2.0
        adj_score += horse.get('pos_score', 0) * weight
        return adj_score

    def analyze(self, horses, race_cond, jockey_roi):
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        if front_runners >= 3: dynamic_pace = 'HIGH'
        elif front_runners == 0: dynamic_pace = 'SLOW'
        else: dynamic_pace = 'MIDDLE'

        raw_scores = []
        for h in horses:
            s1 = self.step1_engine_potential(h, race_cond)
            s2 = self.step2_environmental_bias(s1, h, dynamic_pace, race_cond)
            raw_scores.append(s2)

        max_score = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in raw_scores]
        sum_exp = sum(exp_scores)
        win_probs = [es / sum_exp for es in exp_scores]

        results = []
        for i, h in enumerate(horses):
            win_prob = win_probs[i]
            
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            if raw_roi > 1.0: adjusted_roi_mult = 1.0 + math.log1p(raw_roi - 1.0) * 0.4 
            else: adjusted_roi_mult = 1.0 - math.log1p(1.0 - raw_roi) * 0.5
            
            ev = win_prob * h['odds'] * adjusted_roi_mult
            
            results.append({
                '枠': h.get('draw', '-'),
                '馬番': h.get('id', '-'),
                '馬名': h['name'],
                'AIスコア': round(raw_scores[i], 1),
                '推定勝率': f"{round(win_prob * 100, 2)}%",
                'オッズ': h['odds'],
                '期待値(EV)': round(ev, 3)
            })
            
        results.sort(key=lambda x: x['期待値(EV)'], reverse=True)
        return {'predicted_pace': dynamic_pace, 'rankings': results}


# ==========================================
# 2. 顔：Webインターフェース (Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="Abyss V6.0", layout="wide")
    
    # --- サイドバー（当日のレース条件設定） ---
    st.sidebar.header("🚩 今回のレース条件設定")
    target_dist = st.sidebar.number_input("ターゲット距離 (m)", value=1600, step=100)
    surface_type = st.sidebar.selectbox("馬場種別", ["TURF", "DIRT"])
    target_moisture_pct = st.sidebar.slider("路盤含水率 (%)", 0.0, 25.0, 12.0, step=0.5)
    course_type = st.sidebar.selectbox("コース形態", ["LONG_STRAIGHT", "SHORT_STRAIGHT"])
    base_time_1600 = st.sidebar.number_input("1600m換算 基準タイム(秒)", value=94.0)
    target_3f_base = st.sidebar.number_input("基準上がり3F(秒)", value=33.8)

    race_cond = {
        'target_dist': target_dist, 'surface_type': surface_type, 
        'target_moisture_pct': target_moisture_pct, 'course_type': course_type,
        'base_time_1600': base_time_1600, 'target_3f_base': target_3f_base
    }

    # --- メイン画面 ---
    st.markdown("<h2 style='text-align: center; color: #E63946;'>競馬AI投資システム 深淵-Abyss- V6.0</h2>", unsafe_allow_html=True)
    
    st.info("💡 以下の指示文を右上のアイコンからコピーし、AI（Gemini等）に送信してCSVデータを作成させてください。")
    
    # 🌟 AIへの抽出指示プロンプト
    ai_prompt = """あなたはプロの競馬データアナリストです。指定したレースの出走馬データを収集し、以下のCSVフォーマットで出力してください。※ヘッダー行は必ず含めてください。

【指定CSVフォーマット】
馬番,馬名,枠番,単勝オッズ,過去距離,過去走破タイム,過去馬場抵抗係数,過去斤量,今回斤量,年齢月換算,過去年齢月換算,過去スコア履歴,過去ペース係数,上がり3F,脚質,ポジションスコア,騎手名,騎手ランク,逃げ馬フラグ
【出力データ例】
3,オルネーロ,3,45.3,1800,108.8,1.0,57.0,56.0,48,46,72-68-88-70,0.95,33.4,MIDDLE,7,津村明秀,E,False
7,ダイヤモンド,8,4.4,1400,80.7,1.05,55.0,56.0,72,70,85-84-86-85,1.08,34.2,FRONT,9,川田将雅,A,True"""

    # コピー可能なコードブロック
    st.code(ai_prompt, language="text")

    st.markdown("### 👀 AI抽出データをここに貼り付け 👀")
    input_text = st.text_area("", height=200, placeholder="馬番,馬名,枠番,単勝オッズ...")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 期待値(EV)解析を実行", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ データをクリア", use_container_width=True):
            st.rerun()

    # --- 解析実行ロジック ---
    if run_btn:
        if not input_text.strip():
            st.warning("データが入力されていません！")
            return
            
        with st.spinner('深淵アルゴリズムがCSVデータを解析中...'):
            try:
                # 1. 文字列をCSVとしてPandasデータフレームに読み込む
                df = pd.read_csv(io.StringIO(input_text.strip()))
                
                # 2. DataFrameをAbyssエンジン用の辞書リストに変換（パース）
                horse_data = []
                jockey_roi = {} # 簡易的に騎手ランクをA/E比に変換
                rank_to_roi = {'A': 110, 'B': 100, 'C': 90, 'D': 80, 'E': 70}

                for _, row in df.iterrows():
                    # 過去スコアをハイフン区切りの文字列からリストに変換
                    scores_str = str(row['過去スコア履歴']).split('-')
                    scores_array = [float(s) for s in scores_str if s.strip().isdigit() or s.replace('.','').isdigit()]
                    
                    horse_data.append({
                        'id': int(row['馬番']),
                        'draw': int(row['枠番']),
                        'name': str(row['馬名']),
                        'odds': float(row['単勝オッズ']),
                        'past_dist': float(row['過去距離']),
                        'past_time': float(row['過去走破タイム']),
                        'past_resistance': float(row['過去馬場抵抗係数']),
                        'past_weight': float(row['過去斤量']),
                        'target_weight': float(row['今回斤量']),
                        'age_months': int(row['年齢月換算']),
                        'past_age_months': int(row['過去年齢月換算']),
                        'past_scores_array': scores_array,
                        'past_pace_index': float(row['過去ペース係数']),
                        'best_3f': float(row['上がり3F']),
                        'pos_type': str(row['脚質']).upper(),
                        'pos_score': float(row['ポジションスコア']),
                        'jockey': str(row['騎手名']),
                        'rank': str(row['騎手ランク']),
                        'is_front_runner': str(row['逃げ馬フラグ']).strip().lower() == 'true'
                    })
                    
                    # 騎手データを登録
                    jockey_roi[str(row['騎手名'])] = {str(row['騎手ランク']): rank_to_roi.get(str(row['騎手ランク']), 80)}

                # 3. エンジン実行
                engine = DeepROIEngineV6_Abyss(temperature=1.5)
                report = engine.analyze(horse_data, race_cond, jockey_roi)
                
                # 4. 画面に出力
                st.success("解析完了！期待値(EV)ランキングを算出しました。")
                st.markdown(f"#### 🎯 深淵シミュレーション展開予測: **{report['predicted_pace']}ペース**")
                
                # スタイリングして表示（EVが1.0以上の行をハイライト）
                res_df = pd.DataFrame(report['rankings'])
                
                def highlight_high_ev(s):
                    return ['background-color: #d4edda; color: black' if v >= 1.0 else '' for v in s]
                
                st.dataframe(res_df.style.apply(highlight_high_ev, subset=['期待値(EV)']), use_container_width=True)

            except Exception as e:
                st.error(f"データの読み込みに失敗しました。CSVのフォーマットが正しいか確認してください。\n\nエラー詳細: {e}")

if __name__ == "__main__":
    main()
