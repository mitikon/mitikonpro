import yfinance as yf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime
import time
import jpholiday
import pytz
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# グローバル変数（ペーパートレード用状態管理）
# ==========================================
# 保有ポジションを辞書型で記憶（例: [{'ticker': '1617.T', 'entry_price': 2500.5}]）
todays_positions = []

# 当日すでに実行したかを判定するためのフラグ
last_run_morning = None
last_run_afternoon = None

# ==========================================
# ユーティリティ関数
# ==========================================
def get_now_jst():
    """常に正確な日本時間(JST)を取得する"""
    return datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

def is_tse_open(check_date):
    """指定された日付が東京証券取引所の開場日かどうかを判定する"""
    if check_date.weekday() >= 5: # 土日
        return False
    if jpholiday.is_holiday(check_date): # 祝日
        return False
    month, day = check_date.month, check_date.day
    if (month == 12 and day == 31) or (month == 1 and day <= 3): # 年末年始
        return False
    return True

# ==========================================
# データ取得・分析ロジック
# ==========================================
def fetch_data_with_retry(tickers, days=150, retries=3):
    """yfinanceの接続不安定対策のためのリトライ付きデータ取得"""
    end_date = get_now_jst().date()
    start_date = end_date - datetime.timedelta(days=days)
    
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
            if not data.empty:
                return data
        except Exception as e:
            print(f"⚠️ データ取得エラー (試行 {attempt+1}/{retries}): {e}")
            time.sleep(5)
    
    raise ConnectionError("データの取得に規定回数失敗しました。")

def get_current_price(ticker):
    """ペーパートレード用：直近の価格（現在値または前日終値）を取得"""
    try:
        # 1日分のデータを1分足で取得し、最新の価格を取る
        ticker_data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not ticker_data.empty:
            return float(ticker_data['Close'].iloc[-1])
        return None
    except:
        return None

def check_market_trend(topix_data, window=25):
    """TOPIXの25日移動平均線によるトレンド判定"""
    sma_25 = topix_data.rolling(window=window).mean()
    latest_close = topix_data.iloc[-1]
    latest_sma = sma_25.iloc[-1]
    return latest_close >= latest_sma, latest_close, latest_sma

def calculate_pca_signals(data):
    """PCAを用いたシグナル算出（NaN対策強化版）"""
    # 欠損値が多すぎる銘柄を除外(90%以上データがあるものだけ残す)、その後前日終値で穴埋め
    clean_data = data.dropna(axis=1, thresh=int(len(data)*0.9)).ffill().bfill()
    
    if clean_data.empty or len(clean_data.columns) < 2:
        raise ValueError("PCA計算に十分な有効データがありません。")
        
    returns = clean_data.pct_change().dropna()
    
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
# ペーパートレード用 注文実行関数
# ==========================================
def paper_order_market_buy(ticker, score):
    """仮想買い注文"""
    price = get_current_price(ticker)
    if price is None:
        print(f"⚠️ {ticker} の価格取得に失敗したため、買いを見送ります。")
        return

    print(f"💸 【仮想買付】 {ticker} (Zスコア: {score:.2f}) | 約定価格: ¥{price:.2f}")
    todays_positions.append({'ticker': ticker, 'entry_price': price})

def paper_order_market_sell():
    """仮想決済注文と損益(PnL)計算"""
    global todays_positions
    if not todays_positions:
        print("決済するポジションはありません。")
        return

    total_pnl = 0
    for pos in todays_positions:
        ticker = pos['ticker']
        entry_price = pos['entry_price']
        exit_price = get_current_price(ticker)
        
        if exit_price is None:
            print(f"⚠️ {ticker} の価格取得に失敗しました。仮にエントリー価格で決済したものとします。")
            exit_price = entry_price
            
        profit = exit_price - entry_price
        profit_percent = (profit / entry_price) * 100
        total_pnl += profit
        
        print(f"💰 【仮想決済】 {ticker} | 買値: ¥{entry_price:.2f} -> 売値: ¥{exit_price:.2f} | 損益: ¥{profit:.2f} ({profit_percent:.2f}%)")
        
    print(f"📊 本日の仮想トータル損益（1株あたり換算）: ¥{total_pnl:.2f}")
    todays_positions = [] # ポジションクリア

# ==========================================
# ジョブ処理
# ==========================================
def morning_entry_job():
    global todays_positions
    todays_positions = [] 
    now = get_now_jst()

    print(f"\n🌅 [{now.strftime('%Y-%m-%d %H:%M')}] 朝のシグナル判定・仮想エントリーを開始します...")

    sector_tickers = ['1617.T', '1618.T', '1619.T', '1620.T', '1621.T', '1622.T',
                      '1623.T', '1624.T', '1625.T', '1626.T', '1627.T', '1628.T',
                      '1629.T', '1630.T', '1631.T', '1632.T', '1633.T']
    topix_ticker = '1306.T'

    try:
        all_tickers = sector_tickers + [topix_ticker]
        historical_data = fetch_data_with_retry(all_tickers)
        
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
                ticker = long_targets.index[i]
                score = long_targets.iloc[i]
                paper_order_market_buy(ticker, score)
                
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

def afternoon_exit_job():
    now = get_now_jst()
    print(f"\n🌇 [{now.strftime('%Y-%m-%d %H:%M')}] 大引け前の仮想全決済プロセスを開始します...")
    paper_order_market_sell()
    print("✅ 本日のペーパートレードが完了しました。")

# ==========================================
# メインシステム（常駐ループ）
# ==========================================
def run_trading_system():
    global last_run_morning, last_run_afternoon
    
    print("===========================================")
    print("🤖 日本版ETFシステム 【ペーパートレード・検証モード】 起動")
    print("※日本時間(JST)基準で動作します。停止するには Ctrl+C を押してください。")
    print("===========================================\n")

    while True:
        now = get_now_jst()
        today_str = now.strftime('%Y-%m-%d')
        
        # 1. カレンダー判定（休場日は何もしない）
        if not is_tse_open(now.date()):
            time.sleep(3600) # 1時間待機して再チェック（CPU負荷軽減）
            continue
            
        # 2. 朝のエントリージョブ (09:05実行に変更)
        # ※市場が開いて価格がつくのを待つため、08:30から09:05に変更しています
        if now.hour == 9 and now.minute >= 5 and now.hour < 10:
            if last_run_morning != today_str:
                morning_entry_job()
                last_run_morning = today_str
                
        # 3. 大引け前の決済ジョブ (14:50実行)
        if now.hour == 14 and now.minute >= 50:
            if last_run_afternoon != today_str:
                afternoon_exit_job()
                last_run_afternoon = today_str
                
        # 次の判定まで待機（1分おきに時刻チェック）
        time.sleep(60)

if __name__ == "__main__":
    run_trading_system()
