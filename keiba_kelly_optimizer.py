import numpy as np
import pandas as pd
import lightgbm as lgb
import itertools
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

class QuantsHorseRacingSystem:
    def __init__(self, target_type='top3'):
        """
        AI競馬予測・自動資金配分システム
        target_type: 'win'(単勝狙い) または 'top3'(複勝・ワイド狙い)
        """
        self.target_type = target_type
        self.calibrated_model = None
        self.features = ['pca_score', 'odds', 'is_g1', 'is_dirt', 'distance']
        
    def train(self, historical_data):
        """過去データからAIを学習し、確率を較正(キャリブレーション)する"""
        print("--- [1] AIモデルの学習と確率較正を開始 ---")
        X = historical_data[self.features]
        # target_typeに応じて目的変数を切り替え
        y = historical_data['is_win'] if self.target_type == 'win' else historical_data['is_top3']
        
        # 学習用・較正用にデータを分割
        X_temp, X_valid, y_temp, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # LightGBMベースモデル
        base_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
        base_model.fit(X_train, y_train)
        
        # 確率較正（シグモイド補正）
        self.calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(X_calib, y_calib)
        print("完了: 確率較正済みモデルの構築に成功しました。\n")

    def predict_probabilities(self, race_df):
        """出馬表データに対して、較正済みの正確な確率を付与する"""
        df = race_df.copy()
        # predict_probaの [:, 1] がクラス1(対象達成)の確率
        df['ai_prob'] = self.calibrated_model.predict_proba(df[self.features])[:, 1]
        return df

    def get_single_bets(self, race_df, bankroll, fraction=0.25, edge_threshold=1.10):
        """単勝の買い目とケリー基準に基づく最適ベット額を計算"""
        df = self.predict_probabilities(race_df)
        
        # 期待値の計算
        df['expected_return'] = df['ai_prob'] * df['odds']
        value_bets = df[df['expected_return'] >= edge_threshold].copy()
        
        if value_bets.empty:
            return pd.DataFrame()
            
        # ケリー基準
        b = value_bets['odds'] - 1.0
        p = value_bets['ai_prob']
        q = 1.0 - p
        
        value_bets['full_kelly'] = np.maximum(0, p - (q / b))
        value_bets['bet_jpy'] = (bankroll * (value_bets['full_kelly'] * fraction) / 100).apply(np.floor) * 100
        
        return value_bets[value_bets['bet_jpy'] > 0].sort_values('expected_return', ascending=False)

    def get_wide_bets(self, race_df, bankroll, fraction=0.25, edge_threshold=1.10):
        """ワイド（ペア）の買い目とケリー基準に基づく最適ベット額を計算"""
        df = self.predict_probabilities(race_df)
        
        # 単勝予測確率が低すぎる馬(5%未満など)は計算から除外して高速化
        df = df[df['ai_prob'] >= 0.05]
        
        pairs = list(itertools.combinations(df.index.tolist(), 2))
        wide_records = []
        
        for idx_A, idx_B in pairs:
            horse_A, horse_B = df.loc[idx_A], df.loc[idx_B]
            
            # ワイド確率の簡易推定 (独立事象と仮定し、微小な相性補正をかける)
            prob = min(horse_A['ai_prob'] * horse_B['ai_prob'] * 1.2, 1.0)
            # ダミーのワイド下限オッズ生成 (実運用ではJRAのオッズAPIから取得)
            min_odds = max(1.1, round(np.sqrt(horse_A['odds'] * horse_B['odds']) * 0.8, 1))
            
            wide_records.append({
                'horse_A': horse_A['horse_name'], 'horse_B': horse_B['horse_name'],
                'wide_min_odds': min_odds, 'ai_wide_prob': prob
            })
            
        wide_df = pd.DataFrame(wide_records)
        if wide_df.empty: return pd.DataFrame()
        
        wide_df['expected_return'] = wide_df['ai_wide_prob'] * wide_df['wide_min_odds']
        value_bets = wide_df[wide_df['expected_return'] >= edge_threshold].copy()
        
        if value_bets.empty: return pd.DataFrame()
            
        b = value_bets['wide_min_odds'] - 1.0
        p = value_bets['ai_wide_prob']
        q = 1.0 - p
        
        value_bets['full_kelly'] = np.maximum(0, p - (q / b))
        value_bets['bet_jpy'] = (bankroll * (value_bets['full_kelly'] * fraction) / 100).apply(np.floor) * 100
        
        return value_bets[value_bets['bet_jpy'] > 0].sort_values('expected_return', ascending=False)


# ==========================================
# 実運用シミュレーション（全自動フロー）
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    # 1. 過去5年分のJRAデータ（ダミー）を用意
    n_history = 5000
    mock_history = pd.DataFrame({
        'pca_score': np.random.uniform(0, 50, n_history),
        'odds': np.random.uniform(1.5, 100.0, n_history),
        'is_g1': np.random.choice([0, 1], n_history, p=[0.9, 0.1]),
        'is_dirt': np.random.choice([0, 1], n_history),
        'distance': np.random.choice([1200, 1600, 2000, 2400], n_history),
        'is_win': np.random.choice([0, 1], n_history, p=[0.9, 0.1]),
        'is_top3': np.random.choice([0, 1], n_history, p=[0.7, 0.3])
    })

    # 2. システムの初期化と学習（今回はワイド・複勝狙いのため target_type='top3'）
    system = QuantsHorseRacingSystem(target_type='top3')
    system.train(mock_history)

    # 3. 今週末の出馬表データを取得（AIに推論させる）
    weekend_race = pd.DataFrame({
        'horse_name': ['エフフォーリア', 'タイトルホルダー', 'ドウデュース', 'ディープボンド', 'パンサラッサ'],
        'odds': [2.5, 3.8, 5.0, 15.0, 25.0],
        'pca_score': [48.0, 45.0, 40.0, 35.0, 10.0],
        'is_g1': [1, 1, 1, 1, 1], 'is_dirt': [0, 0, 0, 0, 0], 'distance': [2500, 2500, 2500, 2500, 2500]
    })

    # 4. 全自動で買い目と金額を計算（手持ち資金10万円）
    bankroll = 100000
    print(f"--- [2] 今週末のレース予測と資金配分 (総資金: ¥{bankroll:,}) ---")
    
    wide_recommendations = system.get_wide_bets(weekend_race, bankroll=bankroll, fraction=0.25, edge_threshold=1.10)
    
    if wide_recommendations.empty:
        print("※期待値が基準を満たす買い目はありませんでした（見送り推奨）")
    else:
        print("【推奨ワイド買い目】")
        display_cols = ['horse_A', 'horse_B', 'wide_min_odds', 'ai_wide_prob', 'expected_return', 'bet_jpy']
        fmt_df = wide_recommendations[display_cols].copy()
        fmt_df['ai_wide_prob'] = fmt_df['ai_wide_prob'].apply(lambda x: f"{x:.1%}")
        fmt_df['expected_return'] = fmt_df['expected_return'].apply(lambda x: f"{x:.2f}")
        fmt_df['bet_jpy'] = fmt_df['bet_jpy'].apply(lambda x: f"¥{int(x):,}")
        print(fmt_df.to_string(index=False))
