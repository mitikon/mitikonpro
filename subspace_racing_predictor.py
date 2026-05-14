import numpy as np
import pandas as pd

class SubspaceRacingPredictor:
    def __init__(self, lambda_param=0.9, K=3):
        """
        部分空間正則化を用いた競馬予測モデル (Ver 4.5)
        lambda_param: 0.9 (人間の経験則・オッズ・適性を90%重視し、直近データで10%補正)
        """
        self.lambda_param = lambda_param
        self.K = K
        self.top_eigenvectors = None
        
    def _normalize_features(self, df):
        """データを標準化する（平均0, 分散1）"""
        df_norm = df.copy()
        for col in df.columns:
            df_norm[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df_norm

    def fit_extract_factors(self, X_recent, C_0):
        """直近データと事前知識(C_0)から最強ファクター空間を抽出"""
        X_norm = self._normalize_features(X_recent)
        C_t = np.corrcoef(X_norm.T) 
        
        # 式(13): 正則化 (直近データ10% + 普遍的経験則90%)
        C_reg = (1 - self.lambda_param) * C_t + self.lambda_param * C_0
        
        # 式(14): 固有値分解 (K次元への圧縮によりバランス崩壊を防ぐ)
        eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
        idx = np.argsort(eigenvalues)[::-1]
        self.top_eigenvectors = eigenvectors[:, idx][:, :self.K]
        return self.top_eigenvectors

    def predict_scores(self, X_today, win_weights):
        """抽出されたファクターから単勝スコアを算出"""
        X_norm = self._normalize_features(X_today)
        factor_scores = np.dot(X_norm.values, self.top_eigenvectors)
        win_scores = np.dot(factor_scores, win_weights)
        return win_scores

def calculate_expected_value_and_mark(df, threshold_ev=1.5):
    """
    スコアから予測勝率と期待値(EV)を割り出し、印をつける
    """
    scores = df['単勝スコア'].values
    exp_scores = np.exp(scores - np.max(scores)) # Softmax
    df['予測勝率(%)'] = (exp_scores / exp_scores.sum()) * 100
    
    df['実際の単勝オッズ'] = 1 / df['オッズ期待勝率']
    df['期待値(EV)'] = (df['予測勝率(%)'] / 100) * df['実際の単勝オッズ']
    
    def apply_mark(row):
        base_name = row.name
        if row['期待値(EV)'] >= threshold_ev:
            return f"🔴【超抜🉐買い!!】 {base_name}"
        elif row['期待値(EV)'] >= 1.0:
            return f"🟡【ヒモ候補】 {base_name}"
        else:
            return f"　 {base_name} (見送り)"
            
    df['判定馬名'] = df.apply(apply_mark, axis=1)
    return df

# ==========================================
# メイン実行ブロック
# ==========================================
if __name__ == "__main__":
    np.random.seed(42) # 再現性のため固定
    N_horses = 18
    
    # --- 1. 出馬表データ作成 (第6のファクター「距離適性」を追加) ---
    data = {
        "オッズ期待勝率": np.random.uniform(0.01, 0.4, N_horses),
        "人気ギャップ": np.random.uniform(-5, 5, N_horses),     
        "惜敗度(タイム差)": np.random.uniform(-1.5, 0, N_horses),
        "上がり3F": -np.random.uniform(33.0, 36.5, N_horses),    
        "通過順位平均": -np.random.uniform(1, 15, N_horses),     
        "距離適性スコア": np.random.uniform(0.0, 0.5, N_horses)  # NEW: 0.0〜1.0で評価
    }
    horse_names = [f"馬番{str(i).zfill(2)}" for i in range(1, N_horses + 1)]
    df_today = pd.DataFrame(data, index=horse_names)
    
    # ----------------------------------------------------
    # 🎯 特殊シミュレーション：NHKマイルCの「11番」と「8番」を再現
    # ----------------------------------------------------
    
    # 【馬番08】(先ほどの激走穴馬パターン)
    # オッズは低く、距離適性も普通だが、直近の末脚と惜敗度が異常に高い
    df_today.loc["馬番08", "オッズ期待勝率"] = 0.03 (オッズ約33倍)
    df_today.loc["馬番08", "上がり3F"] = -33.2
    df_today.loc["馬番08", "惜敗度(タイム差)"] = -0.1
    df_today.loc["馬番08", "距離適性スコア"] = 0.3
    
    # 【馬番11】(今回システムが取りこぼした「距離戻り」のG1馬パターン)
    # 前走大敗でオッズ・末脚・位置取りは最悪。しかし「距離適性スコア」だけが満点(1.0)
    df_today.loc["馬番11", "オッズ期待勝率"] = 0.05 (オッズ20倍)
    df_today.loc["馬番11", "上がり3F"] = -36.5  (前走大敗のノイズ)
    df_today.loc["馬番11", "通過順位平均"] = -16 (前走大敗のノイズ)
    df_today.loc["馬番11", "距離適性スコア"] = 1.0  (同距離複勝率100％を数値化)
    
    # --- 2. C_0 (事前知識の相関行列) の 6x6 拡張 ---
    # 右端と下端に「距離適性」の行と列を追加。
    # 距離適性は「オッズ(人気)」や「惜敗度」と少し連動すると定義する
    C_0_matrix = np.array([
        [1.0,  0.6,  0.4,  0.1,  0.2,  0.5], # オッズ
        [0.6,  1.0,  0.5,  0.0,  0.1,  0.3], # ギャップ
        [0.4,  0.5,  1.0,  0.3,  0.2,  0.4], # 惜敗度
        [0.1,  0.0,  0.3,  1.0, -0.4,  0.2], # 上がり3F
        [0.2,  0.1,  0.2, -0.4,  1.0,  0.1], # 通過順位
        [0.5,  0.3,  0.4,  0.2,  0.1,  1.0]  # 距離適性スコア (NEW)
    ])
    
    # --- 3. モデルの実行とスコア算出 ---
    predictor = SubspaceRacingPredictor(lambda_param=0.9, K=3)
    predictor.fit_extract_factors(df_today, C_0_matrix)
    
    # 3つの主要ファクターに対する重み付け (総合力, 過去の不遇度, 能力ポテンシャル)
    win_weights = np.array([0.6, 0.4, 0.7]) 
    
    df_today['単勝スコア'] = predictor.predict_scores(df_today, win_weights)
    result_df = calculate_expected_value_and_mark(df_today, threshold_ev=1.5)
    
    # --- 4. 結果表示 ---
    display_cols = ['判定馬名', '実際の単勝オッズ', '予測勝率(%)', '期待値(EV)', '単勝スコア']
    final_output = result_df.sort_values('期待値(EV)', ascending=False)[display_cols]
    
    print("=========================================================")
    print(" 🏇 Ver 4.5 [距離適性実装版] 予測結果")
    print("=========================================================")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(final_output.head(8).to_string(index=False))
