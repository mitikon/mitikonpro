import math

class DeepROIEngineV6_Abyss:
    def __init__(self, ev_threshold=1.0, temperature=1.5):
        self.ev_threshold = ev_threshold
        # Temperature（温度パラメータ）: Softmaxの確率分布を滑らかにする（高すぎると平坦、低すぎると極端）
        self.temperature = temperature

    def _sigmoid(self, x, steepness=2.0, center=0.0):
        """シグモイド関数：任意の中心点を持たせてスケーリング"""
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def step1_engine_potential(self, horse, race_cond):
        """【第1フィルター】物理的能力（馬場・クラス・適性の多次元解析）"""
        score = 100.0
        
        # 1. ペース減衰と馬場状態（Track Condition）の補正
        # 馬場が重いほどタイムはかかるため、過去走の馬場状態を係数化してタイムを標準化
        track_modifier = horse.get('past_track_cond', 1.0) 
        dist_ratio = race_cond['target_dist'] / horse['past_dist']
        decay_factor = dist_ratio ** 1.03 # 距離延長の壁を少し厚く設定
        
        # 過去タイムを「標準良馬場」に補正した上で予測タイムを算出
        normalized_past_time = horse['past_time'] / track_modifier
        est_time = normalized_past_time * decay_factor
        
        # 2. クラスレベル（相手関係）の補正
        # 基準タイムを単純な比例ではなく、レースクラスに応じて厳しくする
        base_time = race_cond['base_time_1600'] * (race_cond['target_dist'] / 1600.0)
        
        # 1秒の差をよりシビアに評価（+3.0点）
        score -= (est_time - base_time) * 3.0
        
        # 3. コース別キレ味ボーナス（クラス相対値で評価）
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            # 基準上がりタイムに対するマージンを評価
            diff = race_cond['target_3f_base'] - horse['best_3f']
            bonus = 20.0 * self._sigmoid(diff, steepness=1.5, center=0.0)
            score += bonus
            
        return score

    def step2_environmental_bias(self, score, horse, dynamic_pace, race_cond):
        """【第2フィルター】展開とトラックバイアス（枠順・脚質の力学）"""
        adj_score = score
        
        # 1. 枠順（ゲート番）によるバイアス補正
        # 内枠(1-4), 中枠(5-10), 外枠(11-) の簡易判定
        draw = horse.get('draw', 8)
        
        # 2. ペース × 脚質 × 枠順 の3Dシミュレーション
        if dynamic_pace == 'HIGH':
            if horse['pos_type'] == 'FRONT': 
                # ハイペースで外枠の逃げ/先行馬は壊滅的ダメージ
                adj_score -= 15.0 if draw >= 11 else 8.0 
            elif horse['pos_type'] == 'BACK': 
                # ハイペースで外枠の差し馬は揉まれず伸びる
                adj_score += 15.0 if draw >= 11 else 10.0
        elif dynamic_pace == 'SLOW':
            if horse['pos_type'] == 'FRONT': 
                # スローで内枠の逃げ/先行は絶好調
                adj_score += 20.0 if draw <= 4 else 10.0
            elif horse['pos_type'] == 'BACK': 
                adj_score -= 12.0 # 届かない
        elif dynamic_pace == 'MIDDLE':
            if horse['pos_type'] == 'FRONT': adj_score += 5.0
            
        # コース特性によるポジション重み付け
        weight = 0.5 if race_cond['course_type'] == 'LONG_STRAIGHT' else 2.0
        adj_score += horse['pos_score'] * weight
        
        return adj_score

    def analyze(self, horses, race_cond, jockey_roi):
        """【最終処理】温度付きSoftmaxとベイズ的期待値（EV）の算出"""
        
        # 1. 展開の動的予測
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        if front_runners >= 3: dynamic_pace = 'HIGH'
        elif front_runners == 0: dynamic_pace = 'SLOW'
        else: dynamic_pace = 'MIDDLE'

        # 2. 各馬の絶対スコア算出
        raw_scores = []
        for h in horses:
            s1 = self.step1_engine_potential(h, race_cond)
            s2 = self.step2_environmental_bias(s1, h, dynamic_pace, race_cond)
            raw_scores.append(s2)

        # 3. Temperature Scaling を用いた Softmax関数
        # スコアの差を温度(T)で割ることで、過剰な自信を抑え、大穴の勝率を現実的な確率に補正する
        max_score = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in raw_scores]
        sum_exp = sum(exp_scores)
        win_probs = [es / sum_exp for es in exp_scores]

        # 4. 市場の歪み（EV）算出
        results = []
        for i, h in enumerate(horses):
            win_prob = win_probs[i]
            
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            
            # 騎手回収率の補正: より厳密な対数圧縮 (1.0を基準に上下を圧縮)
            if raw_roi > 1.0:
                adjusted_roi_mult = 1.0 + math.log1p(raw_roi - 1.0) * 0.4 # 上振れノイズをさらに抑制
            else:
                # 1.0未満の場合も、極端に低い数値を少し持ち上げて底打ちを防ぐ
                adjusted_roi_mult = 1.0 - math.log1p(1.0 - raw_roi) * 0.5
            
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
        
        return {
            'predicted_pace': dynamic_pace,
            'rankings': results
        }

# --- 実行テスト ---
if __name__ == "__main__":
    # Temperatureを1.5に設定し、確率分布を現実に近づける
    engine = DeepROIEngineV6_Abyss(temperature=1.5)
    
    j_roi = {
        '津村明秀': {'E': 1287}, # 異常な上振れ
        '川田将雅': {'A': 65}    # 異常な下振れ
    }
    
    # 馬データに「枠番(draw)」「過去の馬場状態係数(良=1.0, 重=1.05等)」を追加
    horse_data = [
        {'id': 3, 'draw': 3, 'name': 'オルネーロ', 'past_dist': 1800, 'past_time': 108.8, 'past_track_cond': 1.0, 'best_3f': 33.4, 'pos_type': 'MIDDLE', 'pos_score': 7, 'odds': 45.3, 'rank': 'E', 'jockey': '津村明秀', 'is_front_runner': False},
        {'id': 7, 'draw': 14, 'name': 'ダイヤモンドノット', 'past_dist': 1400, 'past_time': 80.7, 'past_track_cond': 1.02, 'best_3f': 34.2, 'pos_type': 'FRONT', 'pos_score': 9, 'odds': 4.4, 'rank': 'A', 'jockey': '川田将雅', 'is_front_runner': True}
    ]
    
    # コース条件に「基準タイム」と「基準上がり3F」を追加（クラスによって変動させる想定）
    race_c = {
        'target_dist': 1600, 
        'course_type': 'LONG_STRAIGHT',
        'base_time_1600': 94.0, # 例: オープンクラスの1600m基準タイム(1分34秒0)
        'target_3f_base': 33.8  # そのクラスで求められる上がりタイム
    }

    report = engine.analyze(horse_data, race_c, j_roi)
    
    print(f"■ 深淵シミュレーション展開予測: {report['predicted_pace']}ペース\n")
    for r in report['rankings']:
        print(f"枠:{r['draw']} | {r['name']} | スコア:{r['score']} | 実質勝率:{r['win_prob_pct']}% | オッズ:{r['odds']}倍 | 騎手補正:{r['adj_roi_mult']} | EV:{r['ev']}")
