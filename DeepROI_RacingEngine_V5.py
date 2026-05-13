import numpy as np
import pandas as pd

class SubspaceRacingPredictor:
    def __init__(self, lambda_param=0.9, K=3):
        """
        部分空間正則化を用いた競馬予測モデル
        lambda_param: 0.9 (人間の経験・オッズを90%重視し、直近データで10%補正する)
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
        
        # 式(13): 正則化
        C_reg = (1 - self.lambda_param) * C_t + self.lambda_param * C_0
        
        # 式(14): 固有値分解
        eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
        idx = np.argsort(eigenvalues)[::-1]
        self.top_eigenvectors = eigenvectors[:, idx][:, :self.K]
        return self.top_eigenvectors

    def predict_scores(self, X_today, win_weights):
        """単勝スコアの算出"""
        X_norm = self._normalize_features(X_today)
        factor_scores = np.dot(X_norm.values, self.top_eigenvectors)
        win_scores = np.dot(factor_scores, win_weights)
        return win_scores

def calculate_expected_value_and_mark(df, threshold_ev=1.5):
    """
    スコアから予測勝率を計算し、期待値(EV)を割り出す。
    基準を超えた「本当の買い馬」には派手な印をつける関数。
    """
    # 1. スコアを「予測勝率 (0〜100%)」に変換 (Softmax関数を使用し精度向上)
    scores = df['単勝スコア'].values
    exp_scores = np.exp(scores - np.max(scores)) # オーバーフロー対策
    df['予測勝率(%)'] = (exp_scores / exp_scores.sum()) * 100
    
    # 2. 実際のオッズを計算 (オッズ期待勝率の逆数)
    df['実際の単勝オッズ'] = 1 / df['オッズ期待勝率']
    
    # 3. 期待値(EV)の計算 = (予測勝率 / 100) * 実際の単勝オッズ
    df['期待値(EV)'] = (df['予測勝率(%)'] / 100) * df['実際の単勝オッズ']
    
    # 4. 【ハイライト処理】期待値が閾値を超える馬にド派手な印をつける
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
    np.random.seed(123) # 再現性のためシード固定
    N_horses = 18
    
    # --- 1. 当日の出馬表データ作成 (ダミー) ---
    data = {
        "オッズ期待勝率": np.random.uniform(0.01, 0.4, N_horses), # 人間の評価 (大きいほど人気)
        "人気ギャップ": np.random.uniform(-5, 5, N_horses),     # 過去の不遇度
        "惜敗度(タイム差)": np.random.uniform(-1.5, 0, N_horses),# 0に近いほど僅差
        "上がり3F": -np.random.uniform(33.0, 36.5, N_horses),    # 物理データ (マイナスにして大きいほど速い評価に)
        "通過順位平均": -np.random.uniform(1, 15, N_horses)     # 物理データ (マイナスにして大きいほど先行評価に)
    }
    # 分かりやすく馬名を設定
    horse_names = [f"馬番{str(i).zfill(2)}" for i in range(1, N_horses + 1)]
    df_today = pd.DataFrame(data, index=horse_names)
    
    # 意図的に「人気はないが、実力値(上がり3Fや惜敗度)が良い穴馬」を馬番07に仕込む
    df_today.loc["馬番07", "オッズ期待勝率"] = 0.05  # オッズ20倍(不人気)
    df_today.loc["馬番07", "人気ギャップ"] = 4.0   # 過去不当に負けている
    df_today.loc["馬番07", "惜敗度(タイム差)"] = -0.1  # 実は僅差
    df_today.loc["馬番07", "上がり3F"] = -33.1     # 末脚は最速クラス
    
    # --- 2. C_0 (事前知識の相関行列) ---
    C_0_matrix = np.array([
        [1.0,  0.6,  0.4,  0.1,  0.2],
        [0.6,  1.0,  0.5,  0.0,  0.1],
        [0.4,  0.5,  1.0,  0.3,  0.2],
        [0.1,  0.0,  0.3,  1.0, -0.4],
        [0.2,  0.1,  0.2, -0.4,  1.0]
    ])
    
    # --- 3. 予測の実行 (lambda=0.9) ---
    predictor = SubspaceRacingPredictor(lambda_param=0.9, K=3)
    predictor.fit_extract_factors(df_today, C_0_matrix)
    
    # ウェイト設定 (総合力, 過去の不遇度, 末脚の鋭さ)
    win_weights = np.array([0.6, 0.4, 0.7]) 
    
    # スコア算出
    df_today['単勝スコア'] = predictor.predict_scores(df_today, win_weights)
    
    # --- 4. 期待値(EV)の算出と、買い目判定 (閾値 1.5倍) ---
    result_df = calculate_expected_value_and_mark(df_today, threshold_ev=1.5)
    
    # --- 5. 結果を【期待値(EV)が高い順】に並び替えて表示 ---
    # 表示する列を整理
    display_cols = ['判定馬名', '実際の単勝オッズ', '予測勝率(%)', '期待値(EV)', '単勝スコア']
    final_output = result_df.sort_values('期待値(EV)', ascending=False)[display_cols]
    
    print("=========================================================")
    print(" 🏇 Ver 4.0 期待値ベース 予測結果 (部分空間正則化)")
    print("=========================================================")
    # Pandasの表示オプションを調整して見やすくする
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(final_output.to_string(index=False))
