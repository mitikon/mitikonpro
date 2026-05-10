import math

class DeepROIEngineV5_1:
    def __init__(self, ev_threshold=1.0):
        self.ev_threshold = ev_threshold

    def _sigmoid(self, x, steepness=2.0):
        """シグモイド関数：閾値付近での極端な点数差を防ぎ、滑らかに評価する"""
        # 演算のオーバーフロー対策
        if x < -10: return 0.0
        if x > 10: return 1.0
        return 1.0 / (1.0 + math.exp(-steepness * x))

    def step1_engine_potential(self, horse, race_cond):
        """【第1フィルター】物理的能力の測定（非線形スケーリングと連続的評価）"""
        score = 100.0
        
        # 距離延長/短縮による非線形なペース減衰・加速を反映
        dist_ratio = race_cond['target_dist'] / horse['past_dist']
        # 距離が延びるほどペースは落ちるため、非線形の補正係数を掛ける
        decay_factor = dist_ratio ** 1.02 
        est_time = horse['past_time'] * decay_factor
        
        # 距離に応じた仮想基準タイム（簡易的に1600m=100秒としてスケーリング）
        base_time = race_cond['target_dist'] * (100.0 / 1600.0) 
        
        # 基準タイムとの差分をスコア化（1秒速い = +2.0点）
        score -= (est_time - base_time) * 2.0
        
        # コース別キレ味ボーナス（シグモイド関数で滑らかに付与）
        if race_cond['course_type'] == 'LONG_STRAIGHT':
            # 33.5秒を基準(0)とし、速いほどプラスに働く
            diff = 33.5 - horse['best_3f']
            bonus = 15.0 * self._sigmoid(diff, steepness=2.0)
            score += bonus
            
        return score

    def step2_environmental_bias(self, score, horse, dynamic_pace, race_cond):
        """【第2フィルター】展開のシミュレーション（全体ペースからの逆算）"""
        adj_score = score
        
        # 全体のペースに基づく有利不利の動的判定
        if dynamic_pace == 'HIGH':
            if horse['pos_type'] == 'FRONT': adj_score -= 10.0 # 逃げ潰れリスク
            elif horse['pos_type'] == 'BACK': adj_score += 15.0 # 差し有利
        elif dynamic_pace == 'SLOW':
            if horse['pos_type'] == 'FRONT': adj_score += 15.0 # 前残り有利
            elif horse['pos_type'] == 'BACK': adj_score -= 10.0 # 届かないリスク
        elif dynamic_pace == 'MIDDLE':
            if horse['pos_type'] == 'FRONT': adj_score += 5.0
            
        # コース特性によるポジション重み付け
        weight = 0.5 if race_cond['course_type'] == 'LONG_STRAIGHT' else 2.0
        adj_score += horse['pos_score'] * weight
        
        return adj_score

    def analyze(self, horses, race_cond, jockey_roi):
        """【最終処理】相対確率化（Softmax）と期待値（EV）の算出"""
        
        # 1. 展開の動的予測（逃げ馬の数でペース判定）
        front_runners = sum(1 for h in horses if h.get('is_front_runner', False))
        if front_runners >= 3:
            dynamic_pace = 'HIGH'
        elif front_runners == 0:
            dynamic_pace = 'SLOW'
        else:
            dynamic_pace = 'MIDDLE'

        # 2. 各馬の絶対スコア算出
        raw_scores = []
        for h in horses:
            s1 = self.step1_engine_potential(h, race_cond)
            s2 = self.step2_environmental_bias(s1, h, dynamic_pace, race_cond)
            raw_scores.append(s2)

        # 3. Softmax関数による相対勝率（確率）への変換
        max_score = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp(s - max_score) for s in raw_scores] # オーバーフロー防止
        sum_exp = sum(exp_scores)
        win_probs = [es / sum_exp for es in exp_scores]

        # 4. 市場の歪み（EV）算出
        results = []
        for i, h in enumerate(horses):
            win_prob = win_probs[i]
            
            # 騎手回収率の取得
            raw_roi = jockey_roi.get(h['jockey'], {}).get(h['rank'], 80) / 100.0
            
            # ノイズフィルター：100%(1.0)を超える極端な回収率は対数圧縮して暴走を防ぐ
            if raw_roi > 1.0:
                adjusted_roi_mult = 1.0 + math.log1p(raw_roi - 1.0) * 0.5
            else:
                adjusted_roi_mult = raw_roi # 100%未満はそのまま適用
            
            # 期待値(EV) = 実質勝率 × オッズ × 補正済み回収率係数
            ev = win_prob * h['odds'] * adjusted_roi_mult
            
            results.append({
                'id': h['id'],
                'name': h['name'],
                'score': round(raw_scores[i], 1),
                'win_prob_pct': round(win_prob * 100, 2),
                'adj_roi_mult': round(adjusted_roi_mult, 2),
                'ev': round(ev, 3),
                'odds': h['odds']
            })
            
        # 期待値順にソート
        results.sort(key=lambda x: x['ev'], reverse=True)
        
        return {
            'predicted_pace': dynamic_pace,
            'rankings': results
        }

# --- 実行テスト ---
if __name__ == "__main__":
    engine = DeepROIEngineV5_1()
    
    # 騎手×ランク別回収率（異常値のテスト）
    j_roi = {
        '津村明秀': {'E': 1287}, # 1287%のノイズデータ
        '川田将雅': {'A': 65}    # 65%の低回収率データ
    }
    
    # 出走馬データ (is_front_runnerを追加)
    horse_data = [
        {'id': 3, 'name': 'オルネーロ', 'past_dist': 1800, 'past_time': 108.8, 'best_3f': 33.4, 'pos_type': 'MIDDLE', 'pos_score': 7, 'odds': 45.3, 'rank': 'E', 'jockey': '津村明秀', 'is_front_runner': False},
        {'id': 7, 'name': 'ダイヤモンドノット', 'past_dist': 1400, 'past_time': 80.7, 'best_3f': 34.2, 'pos_type': 'FRONT', 'pos_score': 9, 'odds': 4.4, 'rank': 'A', 'jockey': '川田将雅', 'is_front_runner': True}
    ]
    
    race_c = {'target_dist': 1600, 'course_type': 'LONG_STRAIGHT'}

    report = engine.analyze(horse_data, race_c, j_roi)
    
    print(f"■ レース展開予測: {report['predicted_pace']}ペース\n")
    for r in report['rankings']:
        print(f"馬番:{r['id']} {r['name']} | 勝率:{r['win_prob_pct']}% | オッズ:{r['odds']}倍 | 補正後騎手係数:{r['adj_roi_mult']} | 期待値(EV):{r['ev']}")
