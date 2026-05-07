"""
Diagnostic: test each concept INDEPENDENTLY on Nifty 750 data (long-only).
Goal: find which single concept has genuine edge before combining.
Tests:
  A) VWAP Mean Reversion — buy when price dips X% below VWAP, sell on reversion
  B) FVG Fill — buy when bullish FVG forms, target: gap fill
  C) Volume Spike Reversal — buy on high-volume bearish candles (capitulation)
  D) Liquidity Sweep Longs — buy after sweep of lows with bullish close
All tested LONG-ONLY with simple next-day / 3-day / 5-day forward returns.
"""

import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/sahil/Developer/Quantitative Finance/data/YF Stock Data Nifty 750"
CUTOFF = "2024-12-31"


def load_stocks(sample_n=100):
    """Load a random sample for speed, or all if sample_n=0."""
    files = glob.glob(os.path.join(DATA_DIR, "*_daily.csv"))
    stocks = []
    for f in files:
        ticker = os.path.basename(f).replace("_daily.csv", "")
        df = pd.read_csv(f, parse_dates=['date'])
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'] <= CUTOFF].copy()
        if len(df) < 200 or df['volume'].tail(60).mean() < 100_000:
            continue
        # Forward returns
        df['ret_1d'] = df['close'].shift(-1) / df['close'] - 1
        df['ret_3d'] = df['close'].shift(-3) / df['close'] - 1
        df['ret_5d'] = df['close'].shift(-5) / df['close'] - 1
        stocks.append((ticker, df))
    if sample_n and len(stocks) > sample_n:
        np.random.seed(42)
        idx = np.random.choice(len(stocks), sample_n, replace=False)
        stocks = [stocks[i] for i in idx]
    print(f"Loaded {len(stocks)} stocks")
    return stocks


def test_vwap_reversion(stocks):
    """A) Buy when close dips X% below rolling VWAP. Test multiple thresholds."""
    print("\n" + "=" * 60)
    print("TEST A: VWAP Mean Reversion (Long-Only)")
    print("=" * 60)
    for period in [10, 20]:
        for thresh in [0.02, 0.03, 0.05]:
            signals_1d, signals_3d, signals_5d = [], [], []
            for ticker, df in stocks:
                tp = (df['high'] + df['low'] + df['close']) / 3
                vwap = (tp * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
                mask = (df['close'] < vwap * (1 - thresh)) & df['ret_1d'].notna()
                signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
                signals_3d.extend(df.loc[mask, 'ret_3d'].dropna().tolist())
                signals_5d.extend(df.loc[mask, 'ret_5d'].dropna().tolist())
            n = len(signals_1d)
            if n < 20:
                continue
            wr = np.mean([r > 0 for r in signals_1d]) * 100
            avg1 = np.mean(signals_1d) * 100
            avg3 = np.mean(signals_3d) * 100 if signals_3d else 0
            avg5 = np.mean(signals_5d) * 100 if signals_5d else 0
            print(f"  VWAP({period}) dip>{thresh*100:.0f}%: N={n:5d} | "
                  f"WR={wr:.1f}% | 1d={avg1:+.3f}% | 3d={avg3:+.3f}% | 5d={avg5:+.3f}%")


def test_fvg_fill(stocks):
    """B) Buy on bullish FVG, expect gap fill / continuation."""
    print("\n" + "=" * 60)
    print("TEST B: Bullish FVG (Long-Only)")
    print("=" * 60)
    signals_1d, signals_3d, signals_5d = [], [], []
    for ticker, df in stocks:
        for i in range(1, len(df) - 5):
            if df['low'].iloc[i + 1] > df['high'].iloc[i - 1]:
                # Bullish FVG at bar i
                if pd.notna(df['ret_1d'].iloc[i + 1]):
                    signals_1d.append(df['ret_1d'].iloc[i + 1])
                if pd.notna(df['ret_3d'].iloc[i + 1]):
                    signals_3d.append(df['ret_3d'].iloc[i + 1])
                if pd.notna(df['ret_5d'].iloc[i + 1]):
                    signals_5d.append(df['ret_5d'].iloc[i + 1])
    n = len(signals_1d)
    wr = np.mean([r > 0 for r in signals_1d]) * 100
    avg1 = np.mean(signals_1d) * 100
    avg3 = np.mean(signals_3d) * 100
    avg5 = np.mean(signals_5d) * 100
    print(f"  Bullish FVG entry: N={n:5d} | WR={wr:.1f}% | "
          f"1d={avg1:+.3f}% | 3d={avg3:+.3f}% | 5d={avg5:+.3f}%")

    # Also test: buy when price revisits (fills) a prior bullish FVG
    signals_1d2 = []
    for ticker, df in stocks:
        fvg_zones = []
        for i in range(1, len(df) - 1):
            if df['low'].iloc[i + 1] > df['high'].iloc[i - 1]:
                fvg_zones.append((df['high'].iloc[i - 1], df['low'].iloc[i + 1], i))
        for gap_lo, gap_hi, fvg_idx in fvg_zones:
            for j in range(fvg_idx + 2, min(fvg_idx + 15, len(df) - 5)):
                if df['low'].iloc[j] <= gap_hi and df['close'].iloc[j] > df['open'].iloc[j]:
                    if pd.notna(df['ret_1d'].iloc[j]):
                        signals_1d2.append(df['ret_1d'].iloc[j])
                    break
    if signals_1d2:
        n2 = len(signals_1d2)
        wr2 = np.mean([r > 0 for r in signals_1d2]) * 100
        avg = np.mean(signals_1d2) * 100
        print(f"  FVG Fill + Bullish close: N={n2:5d} | WR={wr2:.1f}% | 1d={avg:+.3f}%")


def test_volume_spike(stocks):
    """C) Buy after high-volume bearish candle (capitulation buy)."""
    print("\n" + "=" * 60)
    print("TEST C: Volume Spike Reversal (Long-Only)")
    print("=" * 60)
    for vol_mult in [1.5, 2.0, 3.0]:
        signals_1d, signals_3d, signals_5d = [], [], []
        for ticker, df in stocks:
            vol_avg = df['volume'].rolling(20).mean()
            # High volume + bearish candle + next day we buy
            mask = ((df['volume'] > vol_mult * vol_avg) &
                    (df['close'] < df['open']) &
                    df['ret_1d'].notna())
            signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
            signals_3d.extend(df.loc[mask, 'ret_3d'].dropna().tolist())
            signals_5d.extend(df.loc[mask, 'ret_5d'].dropna().tolist())
        n = len(signals_1d)
        if n < 20:
            continue
        wr = np.mean([r > 0 for r in signals_1d]) * 100
        avg1 = np.mean(signals_1d) * 100
        avg3 = np.mean(signals_3d) * 100
        avg5 = np.mean(signals_5d) * 100
        print(f"  VolSpike>{vol_mult}x + bearish: N={n:5d} | "
              f"WR={wr:.1f}% | 1d={avg1:+.3f}% | 3d={avg3:+.3f}% | 5d={avg5:+.3f}%")

    # Also test: high volume BULLISH candle (momentum)
    print("  --- Volume Spike Momentum (buy strong up candle) ---")
    for vol_mult in [1.5, 2.0, 3.0]:
        signals_1d = []
        for ticker, df in stocks:
            vol_avg = df['volume'].rolling(20).mean()
            body = (df['close'] - df['open']) / df['open']
            mask = ((df['volume'] > vol_mult * vol_avg) &
                    (body > 0.02) &
                    df['ret_1d'].notna())
            signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
        n = len(signals_1d)
        if n < 20:
            continue
        wr = np.mean([r > 0 for r in signals_1d]) * 100
        avg1 = np.mean(signals_1d) * 100
        print(f"  VolSpike>{vol_mult}x + bullish body>2%: N={n:5d} | "
              f"WR={wr:.1f}% | 1d={avg1:+.3f}%")


def test_sweep_longs(stocks):
    """D) Liquidity sweep of lows — long only."""
    print("\n" + "=" * 60)
    print("TEST D: Liquidity Sweep of Lows (Long-Only)")
    print("=" * 60)
    for lookback in [10, 20, 40]:
        signals_1d, signals_3d, signals_5d = [], [], []
        for ticker, df in stocks:
            rolling_low = df['low'].rolling(lookback).min().shift(1)
            mask = ((df['low'] < rolling_low) &
                    (df['close'] > df['open']) &
                    df['ret_1d'].notna())
            signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
            signals_3d.extend(df.loc[mask, 'ret_3d'].dropna().tolist())
            signals_5d.extend(df.loc[mask, 'ret_5d'].dropna().tolist())
        n = len(signals_1d)
        wr = np.mean([r > 0 for r in signals_1d]) * 100
        avg1 = np.mean(signals_1d) * 100
        avg3 = np.mean(signals_3d) * 100
        avg5 = np.mean(signals_5d) * 100
        print(f"  Sweep lows({lookback}): N={n:5d} | "
              f"WR={wr:.1f}% | 1d={avg1:+.3f}% | 3d={avg3:+.3f}% | 5d={avg5:+.3f}%")

    # Sweep + VWAP filter
    print("  --- Sweep(20) + Below VWAP(20) ---")
    signals_1d = []
    for ticker, df in stocks:
        tp = (df['high'] + df['low'] + df['close']) / 3
        vwap = (tp * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        rolling_low = df['low'].rolling(20).min().shift(1)
        mask = ((df['low'] < rolling_low) &
                (df['close'] > df['open']) &
                (df['close'] < vwap) &
                df['ret_1d'].notna())
        signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
    n = len(signals_1d)
    if n > 0:
        wr = np.mean([r > 0 for r in signals_1d]) * 100
        avg = np.mean(signals_1d) * 100
        print(f"  Sweep(20)+below VWAP: N={n:5d} | WR={wr:.1f}% | 1d={avg:+.3f}%")


def test_combined_best(stocks):
    """E) Test promising combos found above."""
    print("\n" + "=" * 60)
    print("TEST E: VWAP Dip + Volume Spike Combo")
    print("=" * 60)
    for thresh in [0.02, 0.03, 0.05]:
        for vol_mult in [1.5, 2.0]:
            signals_1d, signals_3d = [], []
            for ticker, df in stocks:
                tp = (df['high'] + df['low'] + df['close']) / 3
                vwap = (tp * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
                vol_avg = df['volume'].rolling(20).mean()
                mask = ((df['close'] < vwap * (1 - thresh)) &
                        (df['volume'] > vol_mult * vol_avg) &
                        df['ret_1d'].notna())
                signals_1d.extend(df.loc[mask, 'ret_1d'].tolist())
                signals_3d.extend(df.loc[mask, 'ret_3d'].dropna().tolist())
            n = len(signals_1d)
            if n < 10:
                continue
            wr = np.mean([r > 0 for r in signals_1d]) * 100
            avg1 = np.mean(signals_1d) * 100
            avg3 = np.mean(signals_3d) * 100
            print(f"  VWAP dip>{thresh*100:.0f}% + vol>{vol_mult}x: N={n:5d} | "
                  f"WR={wr:.1f}% | 1d={avg1:+.3f}% | 3d={avg3:+.3f}%")


if __name__ == '__main__':
    stocks = load_stocks(sample_n=0)  # Use all stocks
    test_vwap_reversion(stocks)
    test_fvg_fill(stocks)
    test_volume_spike(stocks)
    test_sweep_longs(stocks)
    test_combined_best(stocks)
    print("\n✅ Diagnostic complete. Pick the concept(s) with best WR + avg return.")
