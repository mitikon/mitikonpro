import numpy as np
import pandas as pd
from typing import Optional

class SubspaceRegularizationModelV6:
    """
    完全コンテキスト解析モデル Ver 6.0 (ワイド特化運用仕様・堅牢化版)
    """
    def __init__(self):
        # 普遍的経験則(0.9)と直近データ(0.1)の部分空間正則化（黄金比コア）
        self.lambda_recent = 0.1
        self.lambda_universal = 0.9

    def apply_context_filters_v6(self, df: pd.DataFrame, current_class_avg_time: float) -> pd.DataFrame:
        """
        [Ver 6.0] コアを保護しつつ、無印馬の激走を捕捉する動的文脈フィルター
        ※注意：current_class_avg_timeとの比較を行うため、dfには「同距離・同馬場」に
        正規化された前走走破タイムが入力されていることを前提とします。
        """
        df_context = df.copy()
        
        # 必須カラムの存在チェック
        required_cols = ['前走走破タイム', '前走クラス', '今回クラス', '前走レースレベル係数', 
                         '前走からの日数', '年齢', 'トップ生産牧場フラグ', '距離短縮フラグ', 
                         'スローペース予測確率', '上がり3F', '惜敗度(タイム差)']
        missing = [col for col in required_cols if col not in df_context.columns]
        if missing:
            raise KeyError(f"入力データに必須カラムが不足しています: {missing}")

        # ① 動的・昇級ペナルティ (Breakthrough Z-Score)
        time_diff = current_class_avg_time - df_context['前走走破タイム'] 
        
        df_context['実質レースレベル係数'] = np.where(
            (df_context['前走クラス'] < df_context['今回クラス']) & (time_diff > 0),
            1.0, 
            df_context['前走レースレベル係数']
        )
        
        # ② 現代版・休養補正 (Modern Layoff Coefficient)
        # 指数関数的減衰: decay = exp(-days / 180.0)
        base_decay = np.exp(-df_context['前走からの日数'] / 180.0) 
        
        growth_flag = (df_context['年齢'] <= 5) & (df_context['トップ生産牧場フラグ'] == 1) & (df_context['前走からの日数'] <= 150)
        final_decay = np.where(growth_flag, 1.0, base_decay)

        # ③ 展開（ペース）適性バッファ
        pace_bonus = df_context['距離短縮フラグ'] * df_context['スローペース予測確率'] * 1.1

        # ④ 直近パフォーマンススコア (Ct) の適正化
        # ※本来はレース条件に応じて動的な最大値(max_spurt等)を与えるのが望ましい
        base_spurt_score = np.maximum(0, 45.0 - df_context['上がり3F'])
        base_margin_score = np.maximum(0, 5.0 - df_context['惜敗度(タイム差)'])
        
        df_context['補正後_上がりスコア'] = base_spurt_score * df_context['実質レースレベル係数']
        df_context['補正後_惜敗スコア'] = base_margin_score * df_context['実質レースレベル係数']

        df_context['Ct_score'] = (df_context['補正後_上がりスコア'] + df_context['補正後_惜敗スコア']) * final_decay * (1 + pace_bonus)
        
        return df_context

    def calculate_expected_value(self, df_context: pd.DataFrame) -> pd.DataFrame:
        """
        部分空間正則化コアによる最終期待値（EV）算出
        方程式: $C_{reg} = \lambda_{recent} \times C_t + \lambda_{universal} \times C_0$
        """
        if 'C0_score' not in df_context.columns or '予想オッズ' not in df_context.columns:
            raise KeyError("期待値の算出には 'C0_score' と '予想オッズ' カラムが必要です。")

        # コアエンジンによる融合計算
        df_context['EV_score'] = (self.lambda_recent * df_context['Ct_score']) + \
                                 (self.lambda_universal * df_context['C0_score'])
        
        df_context['最終期待値_EV'] = df_context['EV_score'] * df_context['予想オッズ']
        
        return df_context.sort_values(by='最終期待値_EV', ascending=False)

    def format_output_for_wide_bet(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """
        特注仕様：ワイド特化型・3〜4頭動的選出フォーマット生成
        """
        if df_sorted.empty:
            print("警告: 評価対象のデータがありません。")
            return df_sorted

        top_horses = df_sorted.head(4).copy()
        s_score = top_horses.iloc[0]['最終期待値_EV']
        
        # 少頭数（3頭以下）の場合のフェイルセーフを実装
        if len(top_horses) >= 4:
            fourth_score = top_horses.iloc[3]['最終期待値_EV']
            selected_count = 4 if fourth_score >= (s_score * 0.5) else 3
        else:
            selected_count = len(top_horses)
        
        final_selection = top_horses.head(selected_count).copy() 
        
        eval_labels = ['🔴 [S評価]', '🔴 [A評価]', '🟡 [B評価]', '🟡 [B評価]']
        final_selection['評価'] = eval_labels[:selected_count]
        
        print("📊 Ver 6.0 最終解析（ワイド特化型）")
        print("-" * 40)
        for index, row in final_selection.iterrows():
            print(f"{row['評価']} {int(row['馬番']):02d}番 {row['馬名']}")
        
        # ボックス点数の動的計算（n C 2）
        points = int(selected_count * (selected_count - 1) / 2)
        horse_numbers = ", ".join([str(int(x)) for x in final_selection['馬番'].tolist()])
        
        print("-" * 40)
        print(f"🎯 【推奨買い目：ワイド {selected_count}頭ボックス】")
        print(f"対象馬：{horse_numbers}")
        print(f"投資点数：合計 {points}点")
        
        return final_selection
