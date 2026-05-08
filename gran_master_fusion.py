# ==========================================
# 4. コアエンジン（特注評価ボーナス＆V2.0チューニング実装）
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    for _, row in df_raw.iterrows():
        tan_ret = safe_float(row.get('単回値', 0), 0)
        fuku_ret = safe_float(row.get('複回値', 0), 0)
        odds = safe_float(row.get('オッズ', 10), 10)
        up3 = safe_float(row.get('上がり3F順位', 10), 10) 
        
        p_val = str(row.get('ポジション評価', 3)).strip()
        if p_val == '逃げ': pos = 4.0
        elif p_val == '先行': pos = 5.0
        elif p_val == '差し': pos = 3.0
        elif p_val in ['追込', '追い込み']: pos = 1.0
        elif p_val in ['-', '']: pos = 3.0 
        else: pos = safe_float(p_val, 3.0)
            
        j_win = safe_float(row.get('騎手勝率', 0), 0)
        bias = safe_float(row.get('枠バイアス(秒)', 0), 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
        tokuchu = str(row.get('特注評価', 'C')).upper().strip()
        
        baban_val = safe_float(row.get('馬番', 0), 0)
        if baban_val == 0: continue 
        baban = int(baban_val)
        
        bamei = str(row.get('馬名', '不明')).strip()
        waku = int(safe_float(row.get('枠', 0), 0))

        # --- 脈動エンジン V2.0 チューニング ---

        # 【改善1】オッズペナルティを廃止し、投資妙味ゾーン（期待値）をブースト
        if odds <= 3.0: odds_mod = -5            # 過剰人気は期待値が低いため少し割引
        elif 5.0 <= odds <= 15.0: odds_mod = 5   # 狙い目の単勝適正オッズ
        elif 15.0 < odds <= 40.0: odds_mod = 12  # ★最も美味しい中穴ゾーンを評価アップ
        else: odds_mod = -15                     # 100倍超え等の極端な大穴は勝率的にカット
        
        base_score = 80 + odds_mod

        # 【改善2】上がり3Fとポジションのハイブリッド評価（追込馬の救済）
        spurt_bonus = 0
        if up3 <= 3.0:
            if pos >= 4.0: spurt_bonus = 25   # 逃げ先行＋速い上がり＝完全軸（バケモノ）
            elif pos == 3.0: spurt_bonus = 15 # 差し＋速い上がり＝王道
            else: spurt_bonus = 12            # 追込＋速い上がり＝一発の破壊力（旧式0点から修正）

        # 【改善3】回収値（投資妙味）の絶対値評価を追加
        if tan_ret > 300: tan_ret = 80
        if fuku_ret > 300: fuku_ret = 70
        
        # 回収率が90%を超えている馬は「馬券的な旨味がある」として直接ボーナス
        miaomi_bonus = 10 if (tan_ret >= 90 or fuku_ret >= 90) else 0
        val_score = ((tan_ret * 0.5) + (fuku_ret * 0.5)) * 0.3 + miaomi_bonus

        # 定性評価・騎手力
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        tokuchu_bonus = 15 if tokuchu == 'A' else 5 if tokuchu == 'B' else 0
        j_bonus = j_win * 0.4 
        
        # 【算出1】旧システム評価（バイアス抜きの基礎力）
        old_score = base_score + val_score + rank_bonus + j_bonus + spurt_bonus + tokuchu_bonus
        
        # 【算出2】システム3.5評価（枠バイアスを組み込んだ総合期待値）
        v35_score = old_score - (bias * 10)
        
        results.append({
            '馬番': baban, '枠': waku, '馬名': bamei, 'オッズ': odds,
            '旧評価点': round(old_score, 1),
            'V35点': round(v35_score, 1),
            '特注': tokuchu
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty:
        return df_calc
    
    df_calc['総合順位'] = df_calc['V35点'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['V35点'].max()
    df_calc['V40馬身'] = round(((max_score - df_calc['V35点']) * 0.1 * 16.6) / 2.4, 1)

    # 【改善4】最終判定ロジックの適正化（オッズバグの条件緩和と危険な人気馬の厳格化）
    final_output = []
    for _, r in df_calc.iterrows():
        rank = r['総合順位']
        hc = r['V40馬身']
        odds = r['オッズ']
        
        if rank <= 2 and hc <= 1.5: 
            g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単・複】厚め勝負"
        elif rank <= 4 and hc <= 3.0 and odds >= 15.0: 
            g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単・複】妙味狙い"
        elif rank == 1 and odds <= 2.5 and hc >= 2.0: 
            g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】ヒモまで"
        elif rank > 6 and hc > 5.0: 
            g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else: 
            g, l, c, act = "R", "連下・相手候補", "#0056b3", "【通常】相手候補"
        
        final_output.append({
            '総合順位': rank, '馬番': r['馬番'], '枠': r['枠'], '馬名': r['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            '旧評価点': r['旧評価点'], 'V35点': r['V35点'], 'V40馬身': hc, '特注': r['特注']
        })
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)
