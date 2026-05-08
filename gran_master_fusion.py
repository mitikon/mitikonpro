import pandas as pd
import math
import datetime

# ==========================================
# ユーティリティ（防弾仕様のデータ変換）
# ==========================================
def safe_float(val, default_val=0.0):
    try:
        s = str(val).replace('%', '').strip()
        if s in ['-', 'ー', '', 'None', 'null', 'nan']:
            return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# 実稼働用セーフティネット（検証終了後に連携）
# ==========================================
def is_duplicate_order(race_id, horse_number, executed_orders_set):
    """
    重複注文禁止コード
    ※システム検証稼働を終了後、実際のAPI発注処理の直前に組み込んでフタをします。
    """
    order_key = f"{race_id}_{horse_number}"
    if order_key in executed_orders_set:
        return True # すでに注文済み
    return False

# ==========================================
# コアエンジン（V4.2 デュアルエンジン仕様）
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    
    for _, row in df_raw.iterrows():
        # 馬番の取得とスキップ判定
        baban_val = safe_float(row.get('馬番', 0), 0)
        if baban_val == 0: continue
        baban = int(baban_val)
        
        bamei = str(row.get('馬名', '不明')).strip()
        waku = int(safe_float(row.get('枠', 0), 0))
        odds = safe_float(row.get('オッズ', 10.0), 10.0)
        
        # --- 基礎データ抽出 ---
        tan_ret = min(safe_float(row.get('単回値', 0), 0), 250)
        fuku_ret = min(safe_float(row.get('複回値', 0), 0), 180)
        up3 = safe_float(row.get('上がり3F順位', 10), 10)
        j_win = safe_float(row.get('騎手勝率', 0), 0)
        bias = safe_float(row.get('枠バイアス(秒)', 0), 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
        tokuchu = str(row.get('特注評価', 'C')).upper().strip()

        # ポジション評価の数値化
        p_val = str(row.get('ポジション評価', 3)).strip()
        if '逃' in p_val: pos = 4.5
        elif '先' in p_val: pos = 5.0
        elif '差' in p_val: pos = 3.0
        elif '追' in p_val: pos = 1.5
        else: pos = safe_float(p_val, 3.0)

        # ==========================================
        # エンジンA：純粋な勝率（実力）算出モジュール
        # ==========================================
        # 1. 連続的（滑らか）な適性ボーナス計算（クリフエッジの排除）
        # 上がりが速いほど、ポジションが前なほど滑らかに加点
        spurt_score = max(0, 25.0 - (up3 * 1.5) - ((5.0 - pos) * 2.0))
        
        # 2. 定性評価のスコア化
        rank_score = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        tokuchu_score = 18 if tokuchu == 'A' else 7 if tokuchu == 'B' else 0
        
        # 3. 基礎パワーの算出（オッズ・回収率は一切見ない）
        base_power = spurt_score + rank_score + tokuchu_score + (j_win * 0.5) - (bias * 12)
        
        # 4. 基礎パワーを疑似的な勝率(%)に変換 (上限を約45%に設定)
        pseudo_win_rate = min(max(base_power * 0.45, 1.0), 45.0) / 100.0

        # ==========================================
        # エンジンB：期待値（EV）算出モジュール
        # ==========================================
        # 1. 過去統計に基づく「隠れた妙味係数」
        # 単回値・複回値が高い＝オッズ以上に走る傾向がある条件
        historical_value_factor = max(((tan_ret * 0.4) + (fuku_ret * 0.6)) / 100.0, 0.5) 
        
        # 2. 最終期待値（EV）の算出 = 勝率 × オッズ × 統計的妙味
        # EVが1.0を超えれば理論上プラスになる
        expected_value = (pseudo_win_rate * odds) * historical_value_factor

        results.append({
            '馬番': baban, '枠': waku, '馬名': bamei, 'オッズ': odds,
            '勝率予測': round(pseudo_win_rate * 100, 1), 
            'EV値': round(expected_value, 2), 
            '特注': tokuchu
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty: return df_calc
    
    # 評価の主軸を「EV値（期待値）」に変更
    df_calc['総合順位'] = df_calc['EV値'].rank(ascending=False, method='min').astype(int)

    final_output = []
    for _, r in df_calc.iterrows():
        rank = r['総合順位']
        ev = r['EV値']
        win_prob = r['勝率予測']
        odds_val = r['オッズ']
        
        # --- 2軸マトリクス判定（期待値 × 勝率） ---
        if ev >= 1.5 and win_prob >= 15.0:
            g, l, c, act = "S", "的中・回収の完全両立", "#d32f2f", "【単・複】厚め勝負"
        elif ev >= 1.5 and win_prob < 15.0:
            g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単・ワイド】妙味狙い"
        elif ev < 0.8 and win_prob >= 20.0:
            g, l, c, act = "B", "過剰人気の危険馬", "#9c27b0", "【見送り】またはヒモまで"
        elif ev < 0.5:
            g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else:
            g, l, c, act = "R", "連下・相手候補", "#0056b3", "【通常】展開次第"
        
        final_output.append({
            '総合順位': rank, '馬番': r['馬番'], '枠': r['枠'], '馬名': r['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            '勝率予測(%)': win_prob, 'EV値': ev, 'オッズ': odds_val, '特注': r['特注']
        })
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)
