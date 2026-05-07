import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime
import time
import schedule
import jpholiday  # 日本の祝祭日を自動判定するライブラリ
import warnings

warnings.filterwarnings('ignore')

# 本日の保有ポジションを記憶するリスト
todays_positions = []

# ==========================================
# 日本市場（東証）の開場日を判定する関数（自動更新カレンダー）
# ==========================================
def is_tse_open(check_date):
    """
    指定された日付が東京証券取引所の開場日（営業日）かどうかを判定する
    """
    # 1. 土日判定（5:土曜, 6:日曜）
    if check_date.weekday() >= 5:
        return False
        
    # 2. 祝日判定（jpholidayが毎年自動で最新の祝日を計算）
    if jpholiday.is_holiday(check_date):
        return False
        
    # 3. 東証の年末年始休場ルール（12月31日 〜 1月3日）
    month = check_date.month
    day = check_date.day
    if (month == 12 and day == 31) or (month == 1 and day <= 3):
        return False
        
    # 上記すべてに当てはまらない日が「開場日」
    return True

# ==========================================
# 証券会社API連携用のダミー関数
# ==========================================
def order_market_buy(ticker):
    print(f"💸 【買付発注】 証券コード {ticker} の成行買い注文を送信しました！")
    todays_positions.append(ticker)

def order_market_sell(ticker):
    print(f"💰 【決済発注】 証券コード {ticker} の成行売り注文を送信しました！")

# ==========================================
# データ取得・分析ロジック
# ==========================================
def fetch_data(tickers, days=150):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    return data

def check_market_trend(topix_data, window=25):
    sma_25 = topix_data.rolling(window=window).mean()
    latest_close = topix_data.iloc[-1]
    latest_sma = sma_25.iloc[-1]
    return latest_close >= latest_sma, latest_close, latest_sma

def calculate_pca_signals(data):
    returns = data.pct_change().dropna()
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    scaled_df = pd.DataFrame(scaled_returns, index=returns.index, columns=returns.columns)
    
    pca = PCA(n_components=1)
    pca.fit(scaled_df)
    
    reconstructed_scaled = pca.inverse_transform(pca.transform(scaled_df))
    reconstructed_df = pd.DataFrame(reconstructed_scaled, index=returns.index, columns=returns.columns)
    
    residuals = scaled_df - reconstructed_df
    return residuals.iloc[-1]

# ==========================================
# ジョブ①：朝のシグナル判定とエントリー（毎朝08:30実行）
# ==========================================
def morning_entry_job():
    global todays_positions
    todays_positions = [] # ポジションをリセット
    now = datetime.datetime.now()
    
    # 【追加】日本のカレンダーに基づく休場日判定
    if not is_tse_open(now.date()):
        print(f"[{now.strftime('%Y-%m-%d')}] 本日は土日・祝日、または年末年始のため東証は休場です。取引をスキップします。")
        return

    print(f"\n🌅 [{now.strftime('%Y-%m-%d %H:%M')}] 朝のシグナル判定・自動発注プロセスを開始します...")

    sector_tickers = ['1617.T', '1618.T', '1619.T', '1620.T', '1621.T', '1622.T',
                      '1623.T', '1624.T', '1625.T', '1626.T', '1627.T', '1628.T',
                      '1629.T', '1630.T', '1631.T', '1632.T', '1633.T']
    topix_ticker = '1306.T'

    try:
        all_tickers = sector_tickers + [topix_ticker]
        historical_data = fetch_data(all_tickers)
        
        topix_series = historical_data[topix_ticker].dropna()
        is_uptrend, _, _ = check_market_trend(topix_series)
        
        if not is_uptrend:
            print("⚠️ 【トレンド判定: 下落】 本日の買いは見送ります。")
            return

        sector_data = historical_data[sector_tickers]
        latest_residuals = calculate_pca_signals(sector_data)
        
        threshold = -1.5
        long_targets = latest_residuals[latest_residuals <= threshold].sort_values(ascending=True)

        if long_targets.empty:
            print(f"⚠️ 残差 {threshold} 以下のシグナルなし。本日の取引は見送ります。")
        else:
            top_n = min(3, len(long_targets))
            for i in range(top_n):
                ticker = long_targets.index[i].replace('.T', '')
                score = long_targets.iloc[i]
                print(f"🎯 シグナル検出: 【 {ticker} 】 (Zスコア: {score:.2f})")
                order_market_buy(ticker)
                
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

# ==========================================
# ジョブ②：大引け前の全決済（毎日14:50実行）
# ==========================================
def afternoon_exit_job():
    global todays_positions
    now = datetime.datetime.now()
    
    # 【追加】休場日判定
    if not is_tse_open(now.date()):
        return

    print(f"\n🌇 [{now.strftime('%Y-%m-%d %H:%M')}] 大引け前の全決済プロセスを開始します...")
    
    if not todays_positions:
        print("保有ポジションはありません。明日の朝まで待機します。")
        return
        
    for ticker in todays_positions:
        order_market_sell(ticker)
        
    todays_positions = [] 
    print("✅ 本日の取引がすべて完了しました。明日の朝まで待機します。")

# ==========================================
# メインシステム（常駐ループ）
# ==========================================
def run_trading_system():
    print("===========================================")
    print("🤖 日本版ETFシステム 【完全自動運用・日本カレンダー対応モード】 起動")
    print("※システムは稼働中です。停止するには Ctrl+C を押してください。")
    print("===========================================\n")

    schedule.every().day.at("08:30").do(morning_entry_job)
    schedule.every().day.at("14:50").do(afternoon_exit_job)

    print("⏳ 次のスケジュールを待機中...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run_trading_system()
