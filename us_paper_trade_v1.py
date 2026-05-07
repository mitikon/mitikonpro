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
# ファイル名: us_paper_trade_v1.py (テスト稼働版)
# 目的: 日米逆・波及理論のフォワードテスト（日本市場→米国市場）
# 機能: 夏時間対応・高度カレンダー判定搭載
# ==========================================

# 1. 初期設定と安全装置のパラメータ
INITIAL_CAPITAL = 1000000      # 初期テスト資金（100万円）
MAX_RISK_RATIO = 0.30          # 1回の最大発注比率（30%）
KILL_SWITCH_FILE = "STOP.txt"  # 🚨 物理的緊急停止ボタン
ASSET_FILE = "paper_asset_us.txt"
EXCEL_FILE = "paper_trade_report_us.xlsx"

# 2. 逆マッピング（日本の動向を先行指標とし、米国のターゲットを決定）
REVERSE_SECTOR_MAP = {
    "1618.T": "XLK",  # 日本 情報通信 -> 米国 テクノロジー
    "1615.T": "XLF",  # 日本 銀行 -> 米国 金融
    "1610.T": "XLE",  # 日本 エネルギー資源 -> 米国 エネルギー
    "1621.T": "XLV",  # 日本 医薬品 -> 米国 ヘルスケア
    "1624.T": "XLI"   # 日本 機械 -> 米国 資本財
}

def is_nyse_open_today():
    """
    【高度カレンダー機能】現在の米国時間を基準に休場日を判定し、
    夏時間/冬時間を自動判定して、市場が確実に閉場しているかを確認する。
    """
    tz_us = pytz.timezone('US/Eastern')
    tz_jp = pytz.timezone('Asia/Tokyo')
    
    now_jp = datetime.datetime.now(tz_jp)
    now_us = now_jp.astimezone(tz_us)
    us_today = now_us.date()
    
    is_dst = now_us.dst() != datetime.timedelta(0)
    season_str = "夏時間/EDT" if is_dst else "冬時間/EST"
    
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=us_today, end_date=us_today)
    
    if schedule.empty:
        print(f"🗓️ カレンダー判定: 米国時間 {us_today} ({season_str}) は休場日です。")
        return False
        
    open_time_utc = schedule.iloc[0]['market_open']
    close_time_utc = schedule.iloc[0]['market_close']
    
    open_time_jp = open_time_utc.astimezone(tz_jp)
    close_time_jp = close_time_utc.astimezone(tz_jp)
    
    print(f"🗓️ カレンダー判定: 米国時間 {us_today} は営業日です。({season_str} 適用中)")
    print(f"  -> 本日の開場(日本時間): {open_time_jp.strftime('%Y-%m-%d %H:%M')}")
    print(f"  -> 本日の閉場(日本時間): {close_time_jp.strftime('%Y-%m-%d %H:%M')}")
    
    if now_jp < close_time_jp:
        print(f"⚠️ 【警告】現在時刻 ({now_jp.strftime('%H:%M')}) は、まだ米国市場の営業時間中です！")
        print("終値(Close)が確定していないため、誤った計算を防ぐためにシステムを緊急停止します。")
        return False
        
    print("✅ 米国市場の閉場を確認しました。トレード処理を進行します。")
    return True

def fetch_jp_signals():
    print("🇯🇵 今日の日本市場のデータ（先行指標）を取得中...")
    jp_tickers = list(REVERSE_SECTOR_MAP.keys())
    
    data = yf.download(jp_tickers, period="5d", interval="1d", progress=False)['Close']
    if data.empty:
        raise ValueError("日本データの取得に失敗しました。")
        
    returns = data.pct_change().dropna().iloc[-1]
    
    sorted_returns = returns.sort_values(ascending=False)
    strongest_jp = sorted_returns.index[0]
    weakest_jp = sorted_returns.index[-1]
    
    target_long_us = REVERSE_SECTOR_MAP[strongest_jp]
    target_short_us = REVERSE_SECTOR_MAP[weakest_jp]
    
    print(f"  -> 日本最強: {strongest_jp} ({sorted_returns[strongest_jp]*100:.2f}%)")
    print(f"  -> 日本最弱: {weakest_jp} ({sorted_returns[weakest_jp]*100:.2f}%)")
    print(f"  -> 🎯 今夜の米国市場ターゲット: LONG買い [{target_long_us}] / SHORT売り [{target_short_us}]")
    
    return target_long_us, target_short_us

def execute_paper_trade_us(long_ticker, short_ticker, long_budget, short_budget):
    print("🇺🇸 昨晩の米国市場の実際の価格データ（始値・終値）を取得中...")
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
        
        trading_cost = (long_budget + short_budget) * 0.0015
        total_profit = long_profit + short_profit - trading_cost
        
        print(f"  -> {long_ticker} (Long) : 始値 {long_open:.2f} -> 終値 {long_close:.2f} (利益: ¥{long_profit:,.0f})")
        print(f"  -> {short_ticker} (Short): 始値 {short_open:.2f} -> 終値 {short_close:.2f} (利益: ¥{short_profit:,.0f})")
        print(f"  -> 推定コスト: ¥{-trading_cost:,.0f}")
        
        return total_profit
    except Exception as e:
        print(f"【警告】米国データの取得に失敗しました。詳細: {e}")
        return 0

def main():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"[{datetime.datetime.now()}] 米国版ペーパートレード検証システム起動")
    print("-" * 50)

    if os.path.exists(KILL_SWITCH_FILE):
        print("【警告】STOP.txtが検出されました。本日の米国版トレードを強制停止します。")
        return

    if not is_nyse_open_today():
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
        long_ticker, short_ticker = fetch_jp_signals()
    except Exception as e:
        print(f"シグナル抽出エラー: {e}")
        return

    daily_profit = execute_paper_trade_us(long_ticker, short_ticker, long_budget, short_budget)
    
    if daily_profit == 0:
        print("取引は実行されませんでした。")
        return

    total_asset += daily_profit
    with open(ASSET_FILE, "w") as f:
        f.write(str(total_asset))

    print("-" * 50)
    print(f"📊 本日の米国テスト運用結果: ¥{daily_profit:,.0f}")
    print(f"💰 現在の仮想総資産(US枠): ¥{total_asset:,.0f}")
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

    print("✅ エクセルへの米国版テストレポート記録が完了しました。")

if __name__ == "__main__":
    main()
