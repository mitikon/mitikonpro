from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


# 無料の日次価格ソースで取得できる範囲。記号名は内部で固定して特徴量側と共有する。
DEFAULT_UNIVERSE: dict[str, str] = {
    # 主要指数・ベンチマーク
    "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "RSP": "RSP",
    # 11セクター
    "XLC": "XLC", "XLY": "XLY", "XLP": "XLP", "XLE": "XLE",
    "XLF": "XLF", "XLV": "XLV", "XLI": "XLI", "XLB": "XLB",
    "XLRE": "XLRE", "XLK": "XLK", "XLU": "XLU",
    # 市場内部・信用
    "SMH": "SMH", "HYG": "HYG", "LQD": "LQD", "IWM": "IWM",
    # ボラティリティ・金利
    "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M",
    "TNX": "^TNX", "IRX": "^IRX",
    # 商品・為替（先物の期近継続系列）
    "GOLD": "GC=F", "OIL": "CL=F", "COPPER": "HG=F",
    "USDJPY": "JPY=X", "DXY": "DX-Y.NYB",
}


@dataclass(frozen=True)
class MarketDataset:
    close: pd.DataFrame
    volume: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    unadjusted_close: pd.DataFrame | None = None

    def save_csv(self, directory: str | Path) -> tuple[Path, Path]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        close_path = destination / "daily_close.csv"
        volume_path = destination / "daily_volume.csv"
        self.close.to_csv(close_path, index_label="date")
        self.volume.to_csv(volume_path, index_label="date")
        for name, frame in (
            ("daily_open.csv", self.open),
            ("daily_high.csv", self.high),
            ("daily_low.csv", self.low),
            ("daily_unadjusted_close.csv", self.unadjusted_close),
        ):
            if frame is not None:
                frame.to_csv(destination / name, index_label="date")
        return close_path, volume_path


class DailyMarketCollector:
    """確定済み日次バーだけを収集するプロバイダー分離型コレクター。"""

    def __init__(
        self,
        universe: dict[str, str] | None = None,
        downloader: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self.universe = universe or DEFAULT_UNIVERSE.copy()
        self._downloader = downloader

    def collect(self, start: str, end_exclusive: str) -> MarketDataset:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end_exclusive).normalize()
        if start_ts >= end_ts:
            raise ValueError("start must be earlier than end_exclusive")
        downloader = self._downloader or self._yfinance_download
        raw = downloader(
            tickers=list(self.universe.values()),
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise RuntimeError("daily data provider returned no rows")
        close = self._extract(raw, "Adj Close", fallback="Close")
        volume = self._extract(raw, "Volume", allow_missing=True)
        open_price = self._extract(raw, "Open", allow_missing=True)
        high = self._extract(raw, "High", allow_missing=True)
        low = self._extract(raw, "Low", allow_missing=True)
        unadjusted_close = self._extract(raw, "Close", allow_missing=True)
        close = self._rename_and_bound(close, end_ts)
        volume = self._rename_and_bound(volume, end_ts).reindex(close.index)
        open_price = self._rename_and_bound(open_price, end_ts).reindex(close.index)
        high = self._rename_and_bound(high, end_ts).reindex(close.index)
        low = self._rename_and_bound(low, end_ts).reindex(close.index)
        unadjusted_close = self._rename_and_bound(unadjusted_close, end_ts).reindex(close.index)
        required = {"SPY", "QQQ", "RSP", "SMH", "HYG", "LQD", "XLY", "XLP"}
        missing = sorted(symbol for symbol in required if symbol not in close or close[symbol].dropna().empty)
        if missing:
            raise RuntimeError(f"required daily series missing: {missing}")
        # 欠損は可視化したまま保存する。将来値によるbackfillはしない。
        return MarketDataset(
            close.sort_index(),
            volume.sort_index(),
            open_price.sort_index(),
            high.sort_index(),
            low.sort_index(),
            unadjusted_close.sort_index(),
        )

    def _rename_and_bound(self, frame: pd.DataFrame, end_ts: pd.Timestamp) -> pd.DataFrame:
        reverse = {ticker: name for name, ticker in self.universe.items()}
        renamed = frame.rename(columns=reverse)
        renamed.index = pd.to_datetime(renamed.index).tz_localize(None).normalize()
        return renamed.loc[renamed.index < end_ts]

    @staticmethod
    def _extract(
        raw: pd.DataFrame,
        field: str,
        fallback: str | None = None,
        allow_missing: bool = False,
    ) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            available = raw.columns.get_level_values(0)
            selected = field if field in available else fallback
            if selected and selected in available:
                return raw[selected].copy()
        elif field in raw.columns:
            return raw[[field]].copy()
        if allow_missing:
            tickers = raw.columns.get_level_values(-1).unique() if isinstance(raw.columns, pd.MultiIndex) else []
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
        raise RuntimeError(f"provider response has no {field!r} data")

    @staticmethod
    def _yfinance_download(**kwargs) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError("install the data extra: pip install '.[data]'") from error
        return yf.download(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect completed US-market daily bars")
    parser.add_argument("--start", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-exclusive", required=True, help="exclusive YYYY-MM-DD")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    dataset = DailyMarketCollector().collect(args.start, args.end_exclusive)
    close_path, volume_path = dataset.save_csv(args.output)
    print(f"saved {len(dataset.close)} daily rows: {close_path}, {volume_path}")


if __name__ == "__main__":
    main()
