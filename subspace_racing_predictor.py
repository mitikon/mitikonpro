import numpy as np
import pandas as pd

class SubspaceRegularizationModelV6:
    """
    完全コンテキスト解析モデル Ver 6.0 (ワイド特化運用仕様)
    """
    def __init__(self):
        # 普遍的経験則(0.9)と直近データ(0.1)の部分空間正則化（黄金比コア）
        # ※絶対にこの比率（ウェイト）は変更しないこと
        self.lambda_recent = 0.1
        self.lambda_universal = 0.9

    def apply_context_filters_v6(self, df, current_class_avg_time):
        """
        [Ver 6.0] コアを保護しつつ、無印馬の激走を捕捉する動的文脈フィルター
        """
        df_context = df.copy()
        
        # -------------------------------------------------------------
        # ① 動的・昇級ペナルティ (Breakthrough Z-Score)
        # -------------------------------------------------------------
        # 前走の時計が、今回のクラスの平均を上回っているか（仮想の標準化）
        time_diff = current_class_avg_time - df_context['前走走破タイム'] 
        
        # 時計が優秀なら、昇級の壁(クラスペナルティ)を無効化(1.0)
        df_context['実質レースレベル係数'] = np.where(
            (df_context['前走クラス'] < df_context['今回クラス']) & (time_diff > 0),
            1.0, 
            df_context['前走レースレベル係数']
        )
        
        df_context['補正後_上がり3F'] = df_context['上がり3F'] * df_context['実質レースレベル係数']
        df_context['補正後_惜敗度'] = df_context['惜敗度(タイム差)'] * df_context['実質レースレベル係数']

        # -------------------------------------------------------------
        # ② 現代版・休養補正 (Modern Layoff Coefficient)
        # -------------------------------------------------------------
        # 基本の時間減衰ペナルティ
        base_decay = np.exp(-df_context['前走からの日数'] / 180.0) 
        
        # 外厩・成長特例: 5歳以下のトップ牧場生産馬の休養(150日以内)は減衰させない
        growth_flag = (df_context['年齢'] <= 5) & (df_context['トップ生産牧場フラグ'] == 1) & (df_context['前走からの日数'] <= 150)
        final_decay = np.where(growth_flag, 1.0, base_decay)

        # -------------------------------------------------------------
        # ③ 展開（ペース）適性バッファ
        # -------------------------------------------------------------
        # スローペース予測時、距離短縮組にスタミナ加点
        pace_bonus = df_context['距離短縮フラグ'] * df_context['スローペース予測確率'] * 1.1

        # 最終直近パフォーマンススコア (Ct) を算出
        df_context['Ct_score'] = (df_context['補正後_上がり3F'] + df_context['補正後_惜敗度']) * final_decay * (1 + pace_bonus)
        
        return df_context

    def calculate_expected_value(self, df_context):
        """
        部分空間正則化コアによる最終期待値（EV）算出
        C_reg = 0.1 * Ct + 0.9 * C0
        """
        # C0: 普遍的経験則スコア（血統、騎手、外厩などのベース評価。外部で算出済とする）
        
        # コアエンジンによる融合計算
        df_context['EV_score'] = (self.lambda_recent * df_context['Ct_score']) + \
                                 (self.lambda_universal * df_context['C0_score'])
        
        # オッズによる最終的な期待値（旨味）の算出
        df_context['最終期待値_EV'] = df_context['EV_score'] * df_context['予想オッズ']
        
        # 期待値の高い順にソート
        return df_context.sort_values(by='最終期待値_EV', ascending=False)

    def format_output_for_wide_bet(self, df_sorted):
        """
        特注仕様：ワイド特化型・3〜4頭動的選出フォーマット生成
        """
        # 上位4頭を暫定抽出
        top_horses = df_sorted.head(4).copy()
        
        # 動的選出ロジック：4頭目の期待値がS評価(1位)の50%未満なら、ノイズと判定し3頭切りにする
        s_score = top_horses.iloc[0]['最終期待値_EV']
        fourth_score = top_horses.iloc[3]['最終期待値_EV']
        
        selected_count = 4 if fourth_score >= (s_score * 0.5) else 3
        final_selection = top_horses.head(selected_count)
        
        # 評価ラベルの付与
        eval_labels = ['🔴 [S評価]', '🔴 [A評価]', '🟡 [B評価]', '🟡 [B評価]']
        final_selection['評価'] = eval_labels[:selected_count]
        
        # フォーマット出力
        print("📊 Ver 6.0 最終解析（ワイド特化型）")
        print("-" * 40)
        for index, row in final_selection.iterrows():
            print(f"{row['評価']} {row['馬番']:02d}番 {row['馬名']}")
        
        points = 3 if selected_count == 3 else 6
        horse_numbers = ", ".join([str(x) for x in final_selection['馬番'].tolist()])
        print("-" * 40)
        print(f"🎯 【推奨買い目：ワイド {selected_count}頭ボックス】")
        print(f"対象馬：{horse_numbers}")
        print(f"投資点数：合計 {points}点")
        
        return final_selection

# ==========================================
# 実行用メインプロセス
# ==========================================
# model = SubspaceRegularizationModelV6()
# filtered_data = model.apply_context_filters_v6(raw_race_data, class_avg_time)
# ev_data = model.calculate_expected_value(filtered_data)
# final_output = model.format_output_for_wide_bet(ev_data)
