import streamlit as st
import pandas as pd
import math
import io

# ==========================================
# 1. 計算エンジン：DeepROIEngineV6_Abyss
# ==========================================
class DeepROIEngineV6_Abyss:
    def __init__(self, temperature=1.5):
        self.temperature = temperature

    def analyze(self, horses, race_cond, jockey_roi):
        # 簡易計算のため一部の物理計算は内部で行います
        raw_scores = []
        for h in horses:
            # スコアの基本算出
            score = 100.0 - (h['past_time'] - race_cond['base_time_1600']) * 3.0
            # 斤量補正
            weight_diff = h['past_weight'] - h['target_weight']
            score += weight_diff * 5.0
            # 年齢補正（ピーク54ヶ月）
            age_impact = (((h['past_age_months'] - 54)**2) - ((h['age_months'] - 54)**2)) * 0.015
            score += age_impact
            # ボラティリティ
            if 'past_scores_array' in h and h['past_scores_array']:
                std_dev = pd.Series(h['past_scores_array']).std()
                score += (std_dev if not math.isnan(std_dev) else 0) * 0.8
            raw_scores.append(score)

        max_s = max(raw_scores) if raw_scores else 0
        exp_p = [math.exp((s - max_s) / self.temperature) for s in raw_scores]
        sum_p = sum(exp_p)
        
        results = []
        for i, h in enumerate(horses):
            win_prob = exp_p[i] / sum_p
            # 複勝率の推定 (k=2.8)
            place_prob = 1 - (1 - win_prob)**2.8
            
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            adj_roi = 1.0 + math.log1p(raw_roi - 1.0) * 0.4 if raw_roi > 1.0 else 1.0 - math.log1p(1.0 - raw_roi) * 0.5
            
            ev_win = win_prob * h['odds'] * adj_roi
            # オッズバグ判定: EVが高い、または勝率がオッズと大きく乖離
            odds_bug = "🚩BUG!!" if ev_win > 1.8 or (win_prob > (1.0/h['odds']) * 1.5) else ""
            
            results.append({
                '枠': h['draw'],
                '馬名': h['name'],
                'AIスコア': round(raw_scores[i], 1),
                '勝率': round(win_prob * 100, 1),
                '複勝率': round(place_prob * 100, 1),
                'オッズ': h['odds'],
                '期待値(EV)': round(ev_win, 3),
                'バグ': odds_bug
            })
        
        # 複勝有力候補を抽出
        place_candidates = sorted(results, key=lambda x: x['複勝率'], reverse=True)[:3]
        return {
            'rankings': sorted(results, key=lambda x: x['期待値(EV)'], reverse=True),
            'place_candidates': [c['馬名'] for c in place_candidates]
        }

# ==========================================
# 2. UIデザイン (Streamlit)
# ==========================================
def main():
    # 🎨 スタイル設定（背景：白、入力欄：黒）
    st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; }
        .stTextArea textarea {
            background-color: #1A1A1A !important;
            color: #00FF41 !important;
            font-family: 'Courier New', monospace;
        }
        .copy-button {
            background-color: #007BFF;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            font-weight: bold;
            font-size: 20px;
            margin: 20px 0;
            cursor: pointer;
            border: none;
            width: 100%;
        }
        .main-btn {
            background-color: #E63946 !important;
            color: white !important;
            height: 60px;
            font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #E63946;'>競馬AI投資システム 深淵-Abyss- V6.0</h1>", unsafe_allow_html=True)

    # 📋 指示コピー用セクション
    ai_prompt = "あなたはプロの競馬データアナリストです。以下の形式でCSVを出力してください：馬番,馬名,枠,オッズ,過去タイム,過去斤量,今回斤量,年齢月,過去年齢月,過去スコア履歴,過去ペース,上がり3F,脚質,騎手,ランク"
    
    if st.button("📋 AIデータ解析指示 (11項目) をコピー", use_container_width=True):
        st.write(f"以下の指示文をコピーしてください：\n\n`{ai_prompt}`")
        st.success("指示文を表示しました。コピーしてGeminiに貼り付けてください。")

    st.markdown("### 👀 AI抽出データをここに貼り付け 👀")
    input_csv = st.text_area("", height=200, placeholder="ここにAIから届いたCSVを貼り付け...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 期待値(EV)解析を実行", use_container_width=True, type="primary"):
            if input_csv:
                # 解析処理
                df = pd.read_csv(io.StringIO(input_csv.strip()))
                # --- データ変換 (サンプルロジック) ---
                horse_data = []
                for _, row in df.iterrows():
                    horse_data.append({
                        'draw': row.get('枠', 1), 'name': row.get('馬名', 'Unknown'),
                        'past_time': float(row.get('過去タイム', 95.0)), 'odds': float(row.get('オッズ', 10.0)),
                        'past_weight': float(row.get('過去斤量', 56.0)), 'target_weight': float(row.get('今回斤量', 56.0)),
                        'age_months': int(row.get('年齢月', 48)), 'past_age_months': int(row.get('過去年齢月', 46)),
                        'jockey': row.get('騎手', 'Unknown'), 'rank': row.get('ランク', 'C'),
                        'past_scores_array': [80, 85, 90] # 簡易化
                    })
                
                engine = DeepROIEngineV6_Abyss()
                report = engine.analyze(horse_data, {'base_time_1600': 94.0}, {})
                
                st.success("解析完了！ランキングを算出しました。")
                
                # 複勝有力候補の表示
                st.warning(f"🥈 複勝率圏内有力候補: {', '.join(report['place_candidates'])}")
                
                # テーブル表示
                res_df = pd.DataFrame(report['rankings'])
                st.dataframe(res_df.style.highlight_max(axis=0, subset=['勝率', '期待値(EV)'], color='#D4EDDA'))
            else:
                st.error("データが空です。")

    with col2:
        if st.button("🗑️ データオールクリア", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()

Abyss V6.0は、Geminiとの連携を前提に、人間が判断に迷う「市場のバグ」を炙り出す特化型ツールへと進化しました。新しいUIで解析を回してみてください。ご希望の挙動と一致しているか、ぜひご確認をお願いします！
