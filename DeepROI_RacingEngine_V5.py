import math

class DeepROIEngineV6_Abyss:
    def __init__(self, ev_threshold=1.0, temperature=1.5):
        self.ev_threshold = ev_threshold
        # Temperature: Softmaxの確率分布を滑らかにする
        self.temperature = temperature

    def _sigmoid(self, x, steepness=2.0, center=0.0):
        """シグモイド関数：任意の中心点を持たせてスケーリング"""
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    # ==========================================
    # 【新規追加】気象学：含水率モデル
    # ==========================================
    def _calculate_moisture_resistance(self, surface_type, moisture_pct):
        """路盤の含水率(%)からタイム抵抗係数を算出する非線形モデル"""
        if surface_type == 'TURF':
            # 芝：含水率10%を超えると指数関数的に遅くなる
            if moisture_pct <= 10.0:
                return 1.0
            return 1.0 + 0.008 * (max(0, moisture_pct - 10.0) ** 1.2)
        
        elif surface_type == 'DIRT':
            # ダート：15%で砂が最も締まり抵抗が最小（速くなる）
            return 1.0 - 0.05 * math.exp(-((moisture_pct - 15.0) ** 2) / 20.0)
        
        return 1.0

    # ==========================================
    # 【完全差し替え】第1フィルター
    # ==========================================
    def step1_engine_potential(self, horse, race_cond):
        """【第1フィルター】物理・生物・気象の多次元統合解析"""
        
        # 1. 気象学：ターゲットレースの含水率に基づく抵抗係数
        target_resistance = self._calculate_moisture_resistance(
            race_cond['surface_type'], 
            race_cond['target_moisture_pct']
        )
        past_resistance = horse.get('past_resistance', 1.0)
        track_modifier = target_resistance / past_resistance
        
        # 2. 物理学：非線形スケーリングと斤量力学
        dist_ratio = race_cond['target_dist'] / horse['past_dist']
        alpha_horse = 1.04 
        alpha_base = 1.05  
        
        weight_diff = horse.get('past_weight', 56.0) - horse.get('target_weight', 56.0)
        weight_time_impact = - (weight_diff * 0.15) * ((race_cond['target_dist'] / 1000.0) ** 1.1)
        
        # 予測タイム算出
        est_time = (horse['past_time'] * track_modifier * (dist_ratio ** alpha_horse)) + weight_time_impact
        base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** alpha_base)
        
        # 基準タイムとの差分をスコア化
        base_score = 100.0 - (est_time - base_time) * 3.0
        
        # 3. 生物学：エイジング・カーブ
        current_age_months = horse.get('age_months', 60)
        past_age_months = horse.get('past_age_months', 58)
        peak_age = 54.0 
        
        past_penalty = ((past_age_months - peak_age) ** 2) * 0.015
        current_penalty = ((current_age_months - peak_age) ** 2) * 0.015
        aging_impact = past_penalty - current_penalty
        base_score += aging_impact 
        
        # 4. 確率論：標準偏差による上振れ（ボラティリティ）ボーナス
        past_scores = horse.get('past_scores_array', [])
        if len(past_scores) >= 3:
            mean_s = sum(past_scores) / len(past_scores)
            variance = sum((x - mean_s) ** 2 for x in past_scores) / len(past_scores)
            std_dev = math.sqrt(variance)
            
            risk_preference = 0.8 # リスク選好度
            volatility_bonus = std_dev * risk_preference
            base_score += volatility_bonus
        
        # 5. 上がり3Fのペース補正（相対キレ味）
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            ppi = horse.get('past_pace_index', 1.0)
            adjusted_3f = horse['best_3f'] + (1.0 - ppi) * 2.0
            diff = race_cond['target_3f_base'] - adjusted_3f
            bonus = 20.0 * self._sigmoid(diff, steepness=1.5, center=0.0)
            base_score += bonus
            
        return base_score

    def step2_environmental_bias(self, score, horse, dynamic_pace, race_cond):
        """【第2フィルター】展開とトラックバイアス"""
        adj_score = score
        draw = horse.get('draw', 8)
        
        # ペース × 脚質 × 枠順 の3Dシミュレーション
        if dynamic_pace == 'HIGH':
            if horse['pos_type'] == 'FRONT': 
                adj_score -= 15.0 if draw >= 11 else 8.0 
            elif horse['pos_type'] == 'BACK': 
                adj_score += 15.0 if draw >= 11 else 10.0
        elif dynamic_pace == 'SLOW':
            if horse['pos_type'] == 'FRONT': 
                adj_score += 20.0 if draw <= 4 else 10.0
            elif horse['pos_type'] == 'BACK': 
                adj_score -= 12.0 
        elif dynamic_pace == 'MIDDLE':
            if horse['pos_type'] == 'FRONT': adj_score += 5.0
            
        weight = 0.5 if race_cond['course_type'] == 'LONG_STRAIGHT' else 2.0
        adj_score += horse.get('pos_score', 0) * weight
        
        return adj_score

    def analyze(self, horses, race_cond, jockey_roi):
        """【最終処理】温度付きSoftmaxとベイズ的期待値（EV）の算出"""
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
            
            # 騎手のA/E比などのスコアをシミュレート（本来はここで外部マスタを読み込む）
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            
            if raw_roi > 1.0:
                adjusted_roi_mult = 1.0 + math.log1p(raw_roi - 1.0) * 0.4 
            else:
                adjusted_roi_mult = 1.0 - math.log1p(1.0 - raw_roi) * 0.5
            
            # 期待値の計算
            ev = win_prob * h['odds'] * adjusted_roi_mult
            
            results.append({
                'draw': h.get('draw', '-'),
                'name': h['name'],
                'score': round(raw_scores[i], 1),
                'win_prob_pct': round(win_prob * 100, 2),
                'adj_roi_mult': round(adjusted_roi_mult, 2),
                'ev': round(ev, 3),
                'odds': h['odds']
            })
            
        results.sort(key=lambda x: x['ev'], reverse=True)
        return {'predicted_pace': dynamic_pace, 'rankings': results}

# --- 実行テスト ---
if __name__ == "__main__":
    engine = DeepROIEngineV6_Abyss(temperature=1.5)
    
    j_roi = {
        '津村明秀': {'E': 1287}, 
        '川田将雅': {'A': 65}    
    }
    
    # 馬データに新要素を追加（含水率抵抗、斤量、年齢、スコア履歴など）
    horse_data = [
        {
            'id': 3, 'draw': 3, 'name': 'オルネーロ', 
            'past_dist': 1800, 'past_time': 108.8, 'past_resistance': 1.0, 
            'past_weight': 57.0, 'target_weight': 56.0, # 斤量1kg減（有利）
            'age_months': 48, 'past_age_months': 46,    # ピークへ向かう成長期
            'past_scores_array': [72, 68, 88, 70],      # 分散（大穴ポテンシャル）あり
            'past_pace_index': 0.95, 'best_3f': 33.4, 
            'pos_type': 'MIDDLE', 'pos_score': 7, 
            'odds': 45.3, 'rank': 'E', 'jockey': '津村明秀', 'is_front_runner': False
        },
        {
            'id': 7, 'draw': 14, 'name': 'ダイヤモンドノット', 
            'past_dist': 1400, 'past_time': 80.7, 'past_resistance': 1.05, 
            'past_weight': 55.0, 'target_weight': 56.0, # 斤量1kg増（不利）
            'age_months': 72, 'past_age_months': 70,    # ピークアウト（衰退期）
            'past_scores_array': [85, 84, 86, 85],      # 分散小（安定しているが上振れなし）
            'past_pace_index': 1.08, 'best_3f': 34.2, 
            'pos_type': 'FRONT', 'pos_score': 9, 
            'odds': 4.4, 'rank': 'A', 'jockey': '川田将雅', 'is_front_runner': True
        }
    ]
    
    # レース条件に新要素を追加（馬場種別、ターゲット含水率など）
    race_c = {
        'target_dist': 1600, 
        'course_type': 'LONG_STRAIGHT',
        'surface_type': 'TURF',         # 芝レース
        'target_moisture_pct': 12.0,    # 含水率12%（やや水分を含んで時計がかかる）
        'base_time_1600': 94.0, 
        'target_3f_base': 33.8  
    }

    report = engine.analyze(horse_data, race_c, j_roi)
    
    print(f"■ 深淵シミュレーション展開予測: {report['predicted_pace']}ペース\n")
    for r in report['rankings']:
        print(f"枠:{r['draw']:>2} | {r['name']:<10} | スコア:{r['score']:>5.1f} | 実質勝率:{r['win_prob_pct']:>5.2f}% | オッズ:{r['odds']:>5.1f}倍 | EV:{r['ev']:>5.3f}")
