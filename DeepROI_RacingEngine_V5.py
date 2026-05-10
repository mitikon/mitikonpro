import math

# (DeepROIEngineV6_Abyss クラス内のメソッドとして追加/更新)

def _calculate_moisture_resistance(self, surface_type, moisture_pct):
    """路盤の含水率(%)からタイム抵抗係数を算出する非線形モデル"""
    if surface_type == 'TURF':
        # 芝：含水率10%を超えると指数関数的に抵抗が増す（タイムが遅くなる）
        if moisture_pct <= 10.0:
            return 1.0
        return 1.0 + 0.008 * (max(0, moisture_pct - 10.0) ** 1.2)
    
    elif surface_type == 'DIRT':
        # ダート：15%で砂が最も締まり抵抗が最小（タイムが最速）になるガウス分布の底
        # 0.05は最大割引率（5%タイムが速くなる）
        return 1.0 - 0.05 * math.exp(-((moisture_pct - 15.0) ** 2) / 20.0)
    
    return 1.0

def step1_engine_potential(self, horse, race_cond):
    """【第1フィルター】物理・生物・気象の多次元統合解析"""
    
    # 1. 気象学：ターゲットレースの含水率に基づく抵抗係数（馬場差）の算出
    target_resistance = self._calculate_moisture_resistance(
        race_cond['surface_type'], 
        race_cond['target_moisture_pct']
    )
    
    # 過去走の抵抗係数（過去の含水率から逆算。データがなければ1.0）
    past_resistance = horse.get('past_resistance', 1.0)
    
    # 馬場状態による補正（過去と今回のタイム差分を相殺）
    track_modifier = target_resistance / past_resistance
    
    # 2. 物理学：非線形スケーリングと斤量力学
    dist_ratio = race_cond['target_dist'] / horse['past_dist']
    alpha_horse = 1.04 
    alpha_base = 1.05  
    
    weight_diff = horse.get('past_weight', 56.0) - horse.get('target_weight', 56.0)
    weight_time_impact = - (weight_diff * 0.15) * ((race_cond['target_dist'] / 1000.0) ** 1.1)
    
    # 過去タイムを「今回の含水率」に補正し、ターゲット距離へスケーリング＋斤量効果
    est_time = (horse['past_time'] * track_modifier * (dist_ratio ** alpha_horse)) + weight_time_impact
    
    base_time = race_cond['base_time_1600'] * ((race_cond['target_dist'] / 1600.0) ** alpha_base)
    
    # 基準タイムとの差分をスコア化 (100点満点からの減点/加点)
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
        
        # リスク選好度（1.0なら標準偏差1つ分をそのまま上乗せ。穴狙いなら高くする）
        risk_preference = 0.8 
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
