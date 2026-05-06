import pandas as pd
import numpy as np
import datetime
import os
import yfinance as yf
import warnings

# yfinanceの一部警告を非表示にする
warnings.filterwarnings('ignore')

# ==========================================
# [GitHub管理用] プロジェクト名: Dual-Alpha-Project
# ファイル名: jp_paper_trade_v1.py (テスト稼働版)
# 目的: 日米波及理論のフォワードテスト（ペーパートレード）
# ==========================================

# 1. 初期設定とパラメータ
INITIAL_CAPITAL = 1000000      # 初期テスト資金（100万円）
MAX_RISK_RATIO = 0.30          # 1回の最大発注比率（30%）
ASSET_FILE = "paper_asset_jp.txt"
EXCEL_FILE = "paper_trade_report_jp.xlsx"

# 2. 日米セクターETFのマッピング（相関ペア）
SECTOR_MAP = {
    "XLK": "1618.T",  # 米国テクノロジー -> 日本 情報通信・サービス
    "XLF": "1615.T",  # 米国金融 -> 日本 銀行
    "XLE": "1610.T",  # 米国エネルギー -> 日本 エネルギー資源
    "XLV": "1621.T",  # 米国ヘルスケア -> 日本 医薬品
    "XLI": "1624.T"   # 米国資本財 -> 日本 機械
}

def fetch_us_signals():
    """
    【前日の米国市場】のデータから、本日の日本市場のターゲットを決定する
    """
    print("🇺🇸 前日の米国セクターETFの騰落率を取得中...")
    us_tickers = list(SECTOR_MAP.keys())
    
    # 直近5日分のデータを取得し、最新の前日比を計算
    data = yf.download(us_tickers, period="5d", interval="1d", progress=False)['Close']
    if data.empty:
        raise ValueError("米国データの取得に失敗しました。休日の可能性があります。")
        
    returns = data.pct_change().dropna().iloc[-1]
    
    # 簡易PCA/モメンタムロジック（最強をロング、最弱をショート）
    sorted_returns = returns.sort_values(ascending=False)
    strongest_us = sorted_returns.index[0]
    weakest_us = sorted_returns.index[-1]
    
    target_long_jp = SECTOR_MAP[strongest_us]
    target_short_jp = SECTOR_MAP[weakest_us]
    
    print(f"  -> 米国最強: {strongest_us} ({sorted_returns[strongest_us]*100:.2f}%)")
    print(f"  -> 米国最弱: {weakest_us} ({sorted_returns[weakest_us]*100:.2f}%)")
    return target_long_jp, target_short_jp

def execute_paper_trade(long_ticker, short_ticker, long_budget, short_budget):
    """
    【本日の日本市場】の実際の「始値」と「終値」を取得し、
    寄り付きでエントリーし大引けで決済したと仮定した「正確な利益」を計算する
    """
    print("🇯🇵 本日の日本市場の実際の価格データ（始値・終値）を取得中...")
    
    try:
        # 本日のデータを取得
        long_data = yf.download(long_ticker, period="1d", interval="1d", progress=False)
        short_data = yf.download(short_ticker, period="1d", interval="1d", progress=False)
        
        # 始値(Open)と終値(Close)を抽出
        long_open = float(long_data['Open'].iloc[-1])
        long_close = float(long_data['Close'].iloc[-1])
        
        short_open = float(short_data['Open'].iloc[-1])
        short_close = float(short_data['Close'].iloc[-1])
        
        # --- 利益計算（ペーパートレード） ---
        # ロング（買い）の利益率: (終値 - 始値) / 始値
        long_return = (long_close - long_open) / long_open
        long_profit = long_budget * long_return
        
        # ショート（空売り）の利益率: (始値 - 終値) / 始値 （※下がると利益になる）
        short_return = (short_open - short_close) / short_open
        short_profit = short_budget * short_return
        
        # 仮の手数料・スリップページ（往復で約0.1%のコストを差し引く厳しめのテスト）
        trading_cost = (long_budget + short_budget) * 0.001
        
        total_profit = long_profit + short_profit - trading_cost
        
        print(f"  -> {long_ticker} (Long) : 始値 {long_open:.1f} -> 終値 {long_close:.1f} (利益: ¥{long_profit:,.0f})")
        print(f"  -> {short_ticker} (Short): 始値 {short_open:.1f} -> 終値 {short_close:.1f} (利益: ¥{short_profit:,.0f})")
        print(f"  -> 推定コスト: ¥{-trading_cost:,.0f}")
        
        return total_profit
        
    except Exception as e:
        print(f"【警告】本日の日本株データの取得に失敗しました。祝日の可能性があります。詳細: {e}")
        return 0

def main():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"[{datetime.datetime.now()}] ペーパートレード検証システム起動")
    print("-" * 50)

    # 1. 資産の読み込み
    if os.path.exists(ASSET_FILE):
        with open(ASSET_FILE, "r") as f:
            total_asset = float(f.read())
    else:
        total_asset = INITIAL_CAPITAL

    trade_budget = total_asset * MAX_RISK_RATIO
    long_budget = trade_budget / 2
    short_budget = trade_budget / 2

    # 2. 米国データからシグナル抽出
    try:
        long_ticker, short_ticker = fetch_us_signals()
    except Exception as e:
        print(e)
        return

    # 3. 日本市場の実際の価格から損益をシミュレーション（15:00以降に実行される前提）
    daily_profit = execute_paper_trade(long_ticker, short_ticker, long_budget, short_budget)
    
    if daily_profit == 0:
        print("取引は実行されませんでした（データなし）。")
        return

    # 4. 資産の更新
    total_asset += daily_profit
    with open(ASSET_FILE, "w") as f:
        f.write(str(total_asset))

    print("-" * 50)
    print(f"📊 本日のテスト運用結果: ¥{daily_profit:,.0f}")
    print(f"💰 現在の仮想総資産: ¥{total_asset:,.0f}")
    print("-" * 50)

    # 5. エクセルへ記録
    report_data = {
        "日付": [today_str],
        "仮想総資産": [total_asset],
        "Long銘柄": [long_ticker],
        "Short銘柄": [short_ticker],
        "投資予算": [trade_budget],
        "本日の損益": [daily_profit]
    }
    df_report = pd.DataFrame(report_data)

    if os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(EXCEL_FILE, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            startrow = writer.sheets['Sheet1'].max_row
            df_report.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df_report.to_excel(EXCEL_FILE, index=False)

    print("✅ エクセルへのテストレポート記録が完了しました。")

if __name__ == "__main__":
    main()
