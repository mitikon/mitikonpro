import numpy as np
import pandas as pd

class ContextualSubspacePredictorV5:
    def __init__(self, base_lambda=0.9, K=3):
        """
        Ver 5.0: 文脈理解型・部分空間正則化モデル
        base_lambda: 基本の正則化パラメータ
        K: 抽出する主成分(ファクター)の数
        """
        self.base_lambda = base_lambda
        self.K = K
        self.top_eigenvectors = None

    def _normalize_features(self, df):
        """データを標準化（偏差値化：別路線からの比較を可能にする）"""
        df_norm = df.copy()
        for col in df.columns:
            df_norm[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df_norm

    def apply_context_filters(self, df):
        """
        [Ver 5.0 コアロジック] 馬ごとの文脈（ローテーション、格）を数値に翻訳し補正する
        """
        df_context = df.copy()
        
        # ① 格・対戦相手補正（Race Level Adjustment）
        # 前走が格下なら直近スコアを割引、牡馬混合重賞などハイレベルなら割増
        df_context['補正後_上がり3F'] = df_context['上がり3F'] * df_context['前走レースレベル係数']
        df_context['補正後_惜敗度'] = df_context['惜敗度(タイム差)'] * df_context['前走レースレベル係数']

        # ② ローテーション・休養補正（Dynamic Penalty）
        # 休み明け（例: 3ヶ月=約90日以上）は直近データの信頼度を下げる
        decay_factor = np.exp(-df_context['前走からの日数'] / 180.0) # 時間減衰関数
        
        # ③ ステップレース特例（Target Step Exemption）
        # 指定された王道ローテ（フラグ=1）なら、減衰ペナルティを無効化し本気度を加算
        step_bonus = df_context['王道ステップフラグ'] * 1.2
        final_decay = np.where(df_context['王道ステップフラグ'] == 1, 1.0, decay_factor)

        # 直近の物理データを文脈で最終補正
        df_context['最終_直近パフォーマンス'] = (df_context['補正後_上がり3F'] + df_context['補正後_惜敗度']) * final_decay * (1 + step_bonus)
        
        return df_context

    def fit_and_predict(self, df_today, C_0_matrix, win_weights):
        """文脈補正後のデータで主成分を抽出し、スコアを計算"""
        # 文脈フィルターを適用
        df_filtered = self.apply_context_filters(df_today)
        
        # モデルに投入する特徴量を選択
        features = ['オッズ期待勝率', '最終_直近パフォーマンス', '距離・コース適性']
        X_input = df_filtered[features]
        X_norm = self._normalize_features(X_input)
        
        # 相関行列と正則化
        C_t = np.corrcoef(X_norm.T)
        C_reg = (1 - self.base_lambda) * C_t + self.base_lambda * C_0_matrix
        
        # 固有値分解 (次元圧縮)
        eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
        idx = np.argsort(eigenvalues)[::-1]
        self.top_eigenvectors = eigenvectors[:, idx][:, :self.K]
        
        # スコア算出
        factor_scores = np.dot(X_norm.values, self.top_eigenvectors)
        # K個のファクターの重みが足りない場合は自動調整する安全装置
        actual_k = factor_scores.shape[1]
        win_scores = np.dot(factor_scores, win_weights[:actual_k])
        
        return win_scores

def calculate_ev(df, scores, threshold_ev=1.5):
    """スコアから期待値(EV)を算出し判定マークをつける"""
    exp_scores = np.exp(scores - np.max(scores)) # Softmax
    win_probs = (exp_scores / exp_scores.sum())
    
    actual_odds = 1 / df['オッズ期待勝率']
    ev = win_probs * actual_odds
    
    results = df[['馬名', 'オッズ期待勝率']].copy()
    results['単勝オッズ'] = actual_odds
    results['期待値(EV)'] = ev
    
    def apply_mark(ev_val):
        if ev_val >= threshold_ev: return "🔴【超抜🉐買い!!】"
        elif ev_val >= 1.0: return "🟡【ヒモ候補】"
        else: return "❌ (見送り)"
        
    results['判定'] = results['期待値(EV)'].apply(apply_mark)
    return results

# ==========================================
# 実行ブロック（新潟大賞典の「エラー」をどう処理するか検証）
# ==========================================
if __name__ == "__main__":
    # 新潟大賞典でシステムが間違えた14番と、勝った3番のシミュレーションデータ
    data = {
        "馬名": ["グランディア(3番)", "シンハナーダ(14番)", "別路線からの刺客"],
        "オッズ期待勝率": [1/11.7, 1/8.6, 1/20.0], # 6番人気, 4番人気, 穴馬
        "上がり3F": [-33.7, -33.5, -34.0], # 全馬直近の脚は使えている(マイナスは速い意味)
        "惜敗度(タイム差)": [0.0, 0.0, -0.2], 
        "距離・コース適性": [0.9, 0.9, 0.5],
        
        # --- Ver 5.0 追加コンテキスト ---
        "前走レースレベル係数": [1.0, 0.6, 1.5], # 3番は重賞レベル, 14番は3勝クラス(格下割引), 刺客は牡馬混合G2(割増)
        "前走からの日数": [30, 105, 45], # 14番は3ヶ月半の休み明け
        "王道ステップフラグ": [1, 0, 0] # 3番は王道ローテ
    }
    df_test = pd.DataFrame(data)
    
    # 3x3の事前知識行列 (オッズ, 直近パフォーマンス, 適性)
    C_0 = np.array([
        [1.0,  0.5,  0.4],
        [0.5,  1.0,  0.3],
        [0.4,  0.3,  1.0]
    ])
    
    predictor = ContextualSubspacePredictorV5(base_lambda=0.9, K=2)
    win_weights = np.array([0.7, 0.3]) # 主成分への重み
    
    scores = predictor.fit_and_predict(df_test, C_0, win_weights)
    result_df = calculate_ev(df_test, scores, threshold_ev=1.5)
    
    print("=========================================================")
    print(" 🏇 Ver 5.0 [完全コンテキスト解析版] 予測結果")
    print("=========================================================")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(result_df[['判定', '馬名', '単勝オッズ', '期待値(EV)']].sort_values('期待値(EV)', ascending=False).to_string(index=False))
