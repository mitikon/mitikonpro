"""
Project Name: DeepROI_RacingEngine_V5
Logic: 3-Step Multi-Filter Expected Value (EV) Calculation
1. Engine Potential (Time/Distance Correction)
2. Environmental Bias (Pace/Position Prediction)
3. Market Efficiency ROI (Jockey/Rank ROI Filtering)
"""

class DeepROIEngine:
    def __init__(self, ev_threshold=1.0):
        self.ev_threshold = ev_threshold

    def step1_engine_potential(self, horse, race_cond):
        """
        【第1フィルター】物理的能力の測定
        距離補正換算を行い、コース形態に応じた基礎スコアを算出
        """
        score = 100.0
        # 距離逆算ロジック
        avg_f = horse['past_time'] / (horse['past_dist'] / 200)
        est_time = avg_f * (race_cond['target_dist'] / 200)
        
        # 距離延長ペナルティ
        if race_cond['target_dist'] > horse['past_dist']:
            est_time += (race_cond['target_dist'] - horse['past_dist']) * 0.005
            
        score -= (est_time - 100)
        
        # コース別キレ味ボーナス（東京等の直線が長いコース）
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            if horse['best_3f'] <= 33.5:
                score += 15.0
        return score

    def step2_environmental_bias(self, score, horse, ai_pred, race_cond):
        """
        【第2フィルター】展開のシミュレーション
        ペースとポジションによる物理的制約をスコアに反映
        """
        adj_score = score
        # ミドルペース時の後方待機リスク
        if ai_pred['pace'] == 'MIDDLE' and horse['pos_type'] == 'BACK':
            adj_score -= 20.0
        # 逃げ馬の展開利
        if ai_pred['is_front_runner']:
            adj_score += 15.0
            
        # コース特性によるポジション重み付け
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            adj_score += horse['pos_score'] * 0.5
        else:
            adj_score += horse['pos_score'] * 2.0
        return adj_score

    def step3_market_efficiency_roi(self, score, horse, jockey_roi):
        """
        【第3フィルター】市場の歪みと期待値の算出
        騎手×人気ランク別の回収率データを適用し、最終的な投資価値を判定
        """
        win_prob = score / 1000.0 # 簡易勝率換算
        # 騎手の該当ランク回収率
        roi_multiplier = jockey_roi.get(horse['jockey'], {}).get(horse['rank'], 80) / 100.0
        
        # 期待値(EV) = 勝率 × オッズ × 人的回収率係数
        ev = win_prob * horse['odds'] * roi_multiplier
        return round(ev, 2)

    def analyze(self, horses, race_cond, ai_pred, jockey_roi):
        results = []
        for h in horses:
            s1 = self.step1_engine_potential(h, race_cond)
            s2 = self.step2_environmental_bias(s1, h, ai_pred, race_cond)
            ev = self.step3_market_efficiency_roi(s2, h, jockey_roi)
            
            results.append({
                'id': h['id'],
                'name': h['name'],
                'ev': ev,
                'odds': h['odds']
            })
        results.sort(key=lambda x: x['ev'], reverse=True)
        return results

# --- NHKマイルカップ 最終検証用 ---
if __name__ == "__main__":
    engine = DeepROIEngine()
    
    # 騎手×ランク別回収率（バグの抽出）
    j_roi = {
        '津村明秀': {'E': 1287}, # Eランク単回1287%
        '松山弘平': {'D': 146}, 
        '川田将雅': {'A': 65},
        '横山和生': {'A': 113} # 複回ベース
    }
    
    # 出走馬データ
    horse_data = [
        {'id': 3, 'name': 'オルネーロ', 'past_dist': 1800, 'past_time': 108.8, 'best_3f': 33.4, 'pos_type': 'MIDDLE', 'pos_score': 7, 'odds': 45.3, 'rank': 'E', 'jockey': '津村明秀', 'is_front_runner': False},
        {'id': 7, 'name': 'ダイヤモンドノット', 'past_dist': 1400, 'past_time': 80.7, 'best_3f': 34.2, 'pos_type': 'FRONT', 'pos_score': 9, 'odds': 4.4, 'rank': 'A', 'jockey': '川田将雅', 'is_front_runner': False}
    ]
    
    race_c = {'target_dist': 1600, 'course_type': 'LONG_STRAIGHT'}
    ai_p = {'pace': 'MIDDLE'}

    report = engine.analyze(horse_data, race_c, ai_p, j_roi)
    for r in report:
        print(f"馬番:{r['id']} {r['name']} | 期待値(EV):{r['ev']} | オッズ:{r['odds']}倍")
