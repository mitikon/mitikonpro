import pandas as pd
import numpy as np
import datetime
import os
import yfinance as yf
import warnings
import pytz
import pandas_market_calendars as mcal

# yfinanceの一部警告を非表示にする
warnings.filterwarnings('ignore')

# ==========================================
# [GitHub管理用] プロジェクト名: Dual-Alpha-Project
# ファイル名: jp_paper_trade_v1.py (テスト稼働版)
# 目的: 日米波及理論のフォワードテスト（米国市場→日本市場）
# 機能: mcal統合・大引け前フライング防止搭載
# ==========================================

# 1. 初期設定とパラメータ
INITIAL_CAPITAL = 1000000      # 初期テスト資金（100万円）
MAX_RISK_RATIO = 0.30          # 1回の最大発注比率（30%）
KILL_SWITCH_FILE = "STOP_JP.txt" # 🚨 日本版用の緊急停止ボタン
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

def is_jpx_open_today():
    """
    【高度カレンダー機能】pandas_market_calendarsを使ってJPX（東証）の営業日を判定し、
    15:30の大引けを確実に過ぎているかを確認する。
    """
    tz_jp = pytz.timezone('Asia/Tokyo')
    now_jp = datetime.datetime.now(tz_jp)
    jp_today = now_jp.date()
    
    jpx = mcal.get_calendar('JPX')
    schedule = jpx.schedule(start_date=jp_today, end_date=jp_today)
    
    if schedule.empty:
        print(f"🗓️ カレンダー判定: 本日 {jp_today} は日本市場の休場日（土日・祝日等）です。")
        return False
        
    open_time_utc = schedule.iloc[0]['market_open']
    close_time_utc = schedule.iloc[0]['market_close']
    
    open_time_jp = open_time_utc.astimezone(tz_jp)
    close_time_jp = close_time_utc.astimezone(tz_jp)
    
    print(f"🗓️ カレンダー判定: 本日 {jp_today} は日本市場の営業日です。")
    print(f"  -> 予定開場時間: {open_time_jp.strftime('%H:%M')}")
    print(f"  -> 予定閉場時間: {close_time_jp.strftime('%H:%M')}")
    
    if now_jp < close_time_jp:
        print(f"⚠️ 【警告】現在時刻 ({now_jp.strftime('%H:%M')}) は、まだ日本市場の営業時間中です！")
        print("終値(Close)が確定していないため、誤った計算を防ぐためにシステムを緊急停止します。")
        return False
        
    print("✅ 日本市場の閉場（大引け）を確認しました。トレード処理を進行します。")
    return True

def fetch_us_signals():
    print("🇺🇸 前日の米国セクターETFの騰落率を取得中...")
    us_tickers = list(SECTOR_MAP.keys())
    
    data = yf.download(us_tickers, period="5d", interval="1d", progress=False)['Close']
    if data.empty:
        raise ValueError("米国データの取得に失敗しました。")
        
    returns = data.pct_change().dropna().iloc[-1]
    
    sorted_returns = returns.sort_values(ascending=False)
    strongest_us = sorted_returns.index[0]
    weakest_us = sorted_returns.index[-1]
    
    target_long_jp = SECTOR_MAP[strongest_us]
    target_short_jp = SECTOR_MAP[weakest_us]
    
    print(f"  -> 米国最強: {strongest_us} ({sorted_returns[strongest_us]*100:.2f}%)")
    print(f"  -> 米国最弱: {weakest_us} ({sorted_returns[weakest_us]*100:.2f}%)")
    print(f"  -> 🎯 本日の日本市場ターゲット: LONG買い [{target_long_jp}] / SHORT売り [{target_short_jp}]")
    
    return target_long_jp, target_short_jp

def execute_paper_trade(long_ticker, short_ticker, long_budget, short_budget):
    print("🇯🇵 本日の日本市場の実際の価格データ（始値・終値）を取得中...")
    try:
        long_data = yf.download(long_ticker, period="1d", interval="1d", progress=False)
        short_data = yf.download(short_ticker, period="1d", interval="1d", progress=False)
        
        long_open = float(long_data['Open'].iloc[-1])
        long_close = float(long_data['Close'].iloc[-1])
        short_open = float(short_data['Open'].iloc[-1])
        short_close = float(short_data['Close'].iloc[-1])
        
        long_return = (long_close - long_open) / long_open
        long_profit = long_budget * long_return
        
        short_return = (short_open - short_close) / short_open
        short_profit = short_budget * short_return
        
        trading_cost = (long_budget + short_budget) * 0.001
        total_profit = long_profit + short_profit - trading_cost
        
        print(f"  -> {long_ticker} (Long) : 始値 {long_open:.1f} -> 終値 {long_close:.1f} (利益: ¥{long_profit:,.0f})")
        print(f"  -> {short_ticker} (Short): 始値 {short_open:.1f} -> 終値 {short_close:.1f} (利益: ¥{short_profit:,.0f})")
        print(f"  -> 推定コスト: ¥{-trading_cost:,.0f}")
        
        return total_profit
    except Exception as e:
        print(f"【警告】本日の日本株データの取得に失敗しました。詳細: {e}")
        return 0

def main():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"[{datetime.datetime.now()}] 日本版ペーパートレード検証システム起動")
    print("-" * 50)

    if os.path.exists(KILL_SWITCH_FILE):
        print("【警告】STOP_JP.txtが検出されました。本日の日本版トレードを強制停止します。")
        return

    if not is_jpx_open_today():
        print("システムを安全に終了します。")
        return
    print("-" * 50)

    if os.path.exists(ASSET_FILE):
        with open(ASSET_FILE, "r") as f:
            total_asset = float(f.read())
    else:
        total_asset = INITIAL_CAPITAL

    trade_budget = total_asset * MAX_RISK_RATIO
    long_budget = trade_budget / 2
    short_budget = trade_budget / 2

    try:
        long_ticker, short_ticker = fetch_us_signals()
    except Exception as e:
        print(f"シグナル抽出エラー: {e}")
        return

    daily_profit = execute_paper_trade(long_ticker, short_ticker, long_budget, short_budget)
    
    if daily_profit == 0:
        print("取引は実行されませんでした。")
        return

    total_asset += daily_profit
    with open(ASSET_FILE, "w") as f:
        f.write(str(total_asset))

    print("-" * 50)
    print(f"📊 本日の日本テスト運用結果: ¥{daily_profit:,.0f}")
    print(f"💰 現在の仮想総資産(JP枠): ¥{total_asset:,.0f}")
    print("-" * 50)

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

    print("✅ エクセルへの日本版テストレポート記録が完了しました。")

if __name__ == "__main__":
    main()
