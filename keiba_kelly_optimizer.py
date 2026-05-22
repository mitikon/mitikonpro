import numpy as np
import pandas as pd
import lightgbm as lgb
import itertools
from sklearn.calibration import CalibratedClassifierCV

class QuantsHorseRacingSystem:
    def __init__(self, target_type='top3'):
        """
        AI競馬予測・自動資金配分システム（改善版）
        target_type: 'win'(単勝狙い) または 'top3'(複勝・ワイド狙い)
        """
        self.target_type = target_type
        self.calibrated_model = None
        self.features = ['pca_score', 'odds', 'is_g1', 'is_dirt', 'distance']
        
    def train(self, historical_data):
        """過去データからAIを学習し、確率を較正(キャリブレーション)する"""
        print("--- [1] AIモデルの学習と確率較正を開始 ---")
        
        # 【改善1】データの順序が時系列順(古い順)であると仮定し、時系列で分割
        # ランダム分割(train_test_split)は時系列リークを起こすため使用しない
        n_samples = len(historical_data)
        train_idx = int(n_samples * 0.6)
        valid_idx = int(n_samples * 0.8)
        
        df_train = historical_data.iloc[:train_idx]
        df_valid = historical_data.iloc[train_idx:valid_idx]
        df_calib = historical_data.iloc[valid_idx:]
        
        target_col = 'is_win' if self.target_type == 'win' else 'is_top3'
        
        X_train, y_train = df_train[self.features], df_train[target_col]
        X_valid, y_valid = df_valid[self.features], df_valid[target_col]
        X_calib, y_calib = df_calib[self.features], df_calib[target_col]
        
        # 【改善2】Early Stoppingを導入し、過学習を防止
        base_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=4, random_state=42, verbose=-1)
        
        # LightGBMのコールバック仕様変更に対応
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        base_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=callbacks
        )
        
        # 確率較正（シグモイド補正）
        self.calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(X_calib, y_calib)
        print("完了: 確率較正済みモデルの構築に成功しました。\n")

    def predict_probabilities(self, race_df):
        df = race_df.copy()
        df['ai_prob'] = self.calibrated_model.predict_proba(df[self.features])[:, 1]
        return df

    def _apply_kelly_criterion(self, df, prob_col, odds_col, bankroll, fraction, max_exposure):
        """ケリー基準の計算と、1レースあたりの投資上限(max_exposure)の制御"""
        b = df[odds_col] - 1.0
        p = df[prob_col]
        q = 1.0 - p
        
        # ハーフ・ケリーなどのフラクションを適用
        df['full_kelly'] = np.maximum(0, p - (q / b))
        df['target_fraction'] = df['full_kelly'] * fraction
        
        # 【改善3】同一レースでの過剰ベットを防ぐための正規化（上限キャップ）
        total_fraction = df['target_fraction'].sum()
        if total_fraction > max_exposure:
            # 上限を超える場合は、買い目の比率を維持したまま全体をスケールダウン
            scale_factor = max_exposure / total_fraction
            df['target_fraction'] = df['target_fraction'] * scale_factor
            
        df['bet_jpy'] = (bankroll * df['target_fraction'] / 100).apply(np.floor) * 100
        return df

    def get_single_bets(self, race_df, bankroll, fraction=0.25, edge_threshold=1.10, max_exposure=0.15):
        df = self.predict_probabilities(race_df)
        df['expected_return'] = df['ai_prob'] * df['odds']
        value_bets = df[df['expected_return'] >= edge_threshold].copy()
        
        if value_bets.empty:
            return pd.DataFrame()
            
        value_bets = self._apply_kelly_criterion(value_bets, 'ai_prob', 'odds', bankroll, fraction, max_exposure)
        return value_bets[value_bets['bet_jpy'] > 0].sort_values('expected_return', ascending=False)

    def get_wide_bets(self, race_df, bankroll, fraction=0.25, edge_threshold=1.10, max_exposure=0.15):
        df = self.predict_probabilities(race_df)
        df = df[df['ai_prob'] >= 0.05]
        
        pairs = list(itertools.combinations(df.index.tolist(), 2))
        wide_records = []
        
        for idx_A, idx_B in pairs:
            horse_A, horse_B = df.loc[idx_A], df.loc[idx_B]
            
            # 【改善4】ワイド確率の論理的補正
            # 馬Aが3着以内に入る確率(P(A)) × 馬Aが入った前提で馬Bが残り2枠に入る確率(近似)
            # P(B | A) = P(B) * (2/3) をベースに、少し保守的に見積もる
            prob_A = horse_A['ai_prob']
            prob_B = horse_B['ai_prob']
            
            # 独立事象の単純乗算よりは現実に近い補正
            prob = prob_A * prob_B * 0.85
            
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
            
        # 安全装置付きのケリー基準を適用
        value_bets = self._apply_kelly_criterion(value_bets, 'ai_wide_prob', 'wide_min_odds', bankroll, fraction, max_exposure)
        
        return value_bets[value_bets['bet_jpy'] > 0].sort_values('expected_return', ascending=False)


# ==========================================
# 実運用シミュレーション
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    # 1. 過去5年分のJRAデータ（ダミー：時系列順に並んでいる想定）
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

    system = QuantsHorseRacingSystem(target_type='top3')
    system.train(mock_history)

    weekend_race = pd.DataFrame({
        'horse_name': ['エフフォーリア', 'タイトルホルダー', 'ドウデュース', 'ディープボンド', 'パンサラッサ'],
        'odds': [2.5, 3.8, 5.0, 15.0, 25.0],
        'pca_score': [48.0, 45.0, 40.0, 35.0, 10.0],
        'is_g1': [1, 1, 1, 1, 1], 'is_dirt': [0, 0, 0, 0, 0], 'distance': [2500, 2500, 2500, 2500, 2500]
    })

    bankroll = 100000
    print(f"--- [2] 今週末のレース予測と資金配分 (総資金: ¥{bankroll:,}) ---")
    
    # fraction=0.25 (1/4ケリー), max_exposure=0.15 (1レースへの最大投下資金は総資金の15%まで)
    wide_recommendations = system.get_wide_bets(weekend_race, bankroll=bankroll, fraction=0.25, edge_threshold=1.10)
    
    if wide_recommendations.empty:
        print("※期待値が基準を満たす買い目はありませんでした（見送り推奨）")
    else:
        print("【推奨ワイド買い目】")
        display_cols = ['horse_A', 'horse_B', 'wide_min_odds', 'ai_wide_prob', 'expected_return', 'bet_jpy']
        fmt_df = wide_recommendations[display_cols].copy()
        
        # 見やすいようにフォーマット処理
        fmt_df['ai_wide_prob'] = fmt_df['ai_wide_prob'].map("{:.1%}".format)
        fmt_df['expected_return'] = fmt_df['expected_return'].map("{:.2f}".format)
        fmt_df['bet_jpy'] = fmt_df['bet_jpy'].map("¥{:,.0f}".format)
        
        print(fmt_df.to_string(index=False))
