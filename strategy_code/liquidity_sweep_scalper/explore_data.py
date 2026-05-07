"""
Data exploration script to understand price action patterns for:
- Liquidity sweeps (false breakouts beyond key levels)
- Volume profile (high volume nodes, POC)
- VWAP (anchored volume weighted average price)
- FVG (Fair Value Gaps - imbalances in price delivery)

Goal: Identify how often these patterns co-occur and lead to reversals,
to validate a scalping strategy hypothesis.
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/sahil/Developer/Quantitative Finance/data/YF Stock Data Nifty 750'

# Pick diverse, liquid stocks for analysis
SAMPLE_STOCKS = [
    'RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK',
    'TATASTEEL', 'SBIN', 'BAJFINANCE', 'ADANIENT', 'HINDALCO',
    'AXISBANK', 'KOTAKBANK', 'LT', 'MARUTI', 'SUNPHARMA',
    'WIPRO', 'HCLTECH', 'BHARTIARTL', 'ITC', 'ASIANPAINT'
]


def load_stock(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def compute_vwap(df, period=20):
    """Rolling VWAP over N periods."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = typical_price * df['volume']
    df['vwap'] = tp_vol.rolling(period).sum() / df['volume'].rolling(period).sum()
    return df


def detect_fvg(df):
    """
    Fair Value Gap: A 3-candle pattern where candle[i-1].high < candle[i+1].low (bullish FVG)
    or candle[i-1].low > candle[i+1].high (bearish FVG).
    """
    bullish_fvg = []
    bearish_fvg = []
    for i in range(1, len(df) - 1):
        # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
        if df['low'].iloc[i + 1] > df['high'].iloc[i - 1]:
            bullish_fvg.append(i)
        # Bearish FVG: gap between candle[i-1] low and candle[i+1] high
        if df['high'].iloc[i + 1] < df['low'].iloc[i - 1]:
            bearish_fvg.append(i)
    return bullish_fvg, bearish_fvg


def detect_liquidity_sweep(df, lookback=20):
    """
    Liquidity sweep: price briefly exceeds a recent high/low then reverses.
    - Sweep of highs: high[i] > max(high[i-lookback:i]) but close[i] < open[i] (bearish close)
    - Sweep of lows: low[i] < min(low[i-lookback:i]) but close[i] > open[i] (bullish close)
    """
    sweep_highs = []
    sweep_lows = []
    for i in range(lookback, len(df)):
        recent_high = df['high'].iloc[i - lookback:i].max()
        recent_low = df['low'].iloc[i - lookback:i].min()

        # Sweep of highs: wick above, close below open (rejection)
        if df['high'].iloc[i] > recent_high and df['close'].iloc[i] < df['open'].iloc[i]:
            sweep_highs.append(i)

        # Sweep of lows: wick below, close above open (rejection)
        if df['low'].iloc[i] < recent_low and df['close'].iloc[i] > df['open'].iloc[i]:
            sweep_lows.append(i)

    return sweep_highs, sweep_lows


def compute_volume_profile_poc(df, lookback=20):
    """
    Simple volume profile: compute Point of Control (POC) as the price level 
    with highest volume over lookback period using price bins.
    """
    pocs = []
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        price_min = window['low'].min()
        price_max = window['high'].max()
        if price_max == price_min:
            pocs.append(np.nan)
            continue
        n_bins = 20
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_volumes = np.zeros(n_bins)
        for j in range(len(window)):
            row = window.iloc[j]
            # Distribute volume across bins that the candle spans
            for b in range(n_bins):
                if row['low'] <= bins[b + 1] and row['high'] >= bins[b]:
                    bin_volumes[b] += row['volume'] / n_bins
        poc_bin = np.argmax(bin_volumes)
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
        pocs.append(poc_price)
    return [np.nan] * lookback + pocs


def analyze_stock(ticker):
    """Analyze a single stock for pattern occurrences and forward returns."""
    df = load_stock(ticker)
    
    # Need sufficient data
    if len(df) < 100:
        return None

    # Use last 2 years of data for relevance
    df = df.tail(500).reset_index(drop=True)

    df = compute_vwap(df, period=20)
    df['poc'] = compute_volume_profile_poc(df, lookback=20)

    bullish_fvg, bearish_fvg = detect_fvg(df)
    sweep_highs, sweep_lows = detect_liquidity_sweep(df, lookback=20)

    # Compute forward returns (1-day, 3-day, 5-day)
    df['ret_1d'] = df['close'].shift(-1) / df['close'] - 1
    df['ret_3d'] = df['close'].shift(-3) / df['close'] - 1
    df['ret_5d'] = df['close'].shift(-5) / df['close'] - 1

    results = {
        'ticker': ticker,
        'total_bars': len(df),
        'bullish_fvg_count': len(bullish_fvg),
        'bearish_fvg_count': len(bearish_fvg),
        'sweep_highs_count': len(sweep_highs),
        'sweep_lows_count': len(sweep_lows),
    }

    # === LONG SIGNAL: Sweep of lows + near VWAP/POC + bullish FVG nearby ===
    long_signals = []
    for idx in sweep_lows:
        if idx >= len(df) - 5:
            continue
        row = df.iloc[idx]
        # Check: price near VWAP (within 1.5%)
        if pd.isna(row['vwap']):
            continue
        vwap_dist = abs(row['close'] - row['vwap']) / row['vwap']
        near_vwap = vwap_dist < 0.015

        # Check: price near POC (within 2%)
        if pd.isna(row['poc']):
            continue
        poc_dist = abs(row['close'] - row['poc']) / row['poc']
        near_poc = poc_dist < 0.02

        # Check: bullish FVG within last 3 bars
        has_fvg = any(f in range(idx - 3, idx + 1) for f in bullish_fvg)

        # Volume spike (>1.5x 20-bar avg)
        if idx >= 20:
            avg_vol = df['volume'].iloc[idx - 20:idx].mean()
            vol_spike = row['volume'] > 1.5 * avg_vol
        else:
            vol_spike = False

        if near_vwap or near_poc:
            signal_quality = sum([near_vwap, near_poc, has_fvg, vol_spike])
            long_signals.append({
                'idx': idx,
                'date': row['date'],
                'quality': signal_quality,
                'near_vwap': near_vwap,
                'near_poc': near_poc,
                'has_fvg': has_fvg,
                'vol_spike': vol_spike,
                'ret_1d': df['ret_1d'].iloc[idx],
                'ret_3d': df['ret_3d'].iloc[idx],
                'ret_5d': df['ret_5d'].iloc[idx],
            })

    # === SHORT SIGNAL: Sweep of highs + near VWAP/POC + bearish FVG nearby ===
    short_signals = []
    for idx in sweep_highs:
        if idx >= len(df) - 5:
            continue
        row = df.iloc[idx]
        if pd.isna(row['vwap']) or pd.isna(row['poc']):
            continue

        vwap_dist = abs(row['close'] - row['vwap']) / row['vwap']
        near_vwap = vwap_dist < 0.015
        poc_dist = abs(row['close'] - row['poc']) / row['poc']
        near_poc = poc_dist < 0.02
        has_fvg = any(f in range(idx - 3, idx + 1) for f in bearish_fvg)

        if idx >= 20:
            avg_vol = df['volume'].iloc[idx - 20:idx].mean()
            vol_spike = row['volume'] > 1.5 * avg_vol
        else:
            vol_spike = False

        if near_vwap or near_poc:
            signal_quality = sum([near_vwap, near_poc, has_fvg, vol_spike])
            short_signals.append({
                'idx': idx,
                'date': row['date'],
                'quality': signal_quality,
                'near_vwap': near_vwap,
                'near_poc': near_poc,
                'has_fvg': has_fvg,
                'vol_spike': vol_spike,
                'ret_1d': -df['ret_1d'].iloc[idx],  # Invert for short
                'ret_3d': -df['ret_3d'].iloc[idx],
                'ret_5d': -df['ret_5d'].iloc[idx],
            })

    results['long_signals'] = len(long_signals)
    results['short_signals'] = len(short_signals)

    if long_signals:
        long_df = pd.DataFrame(long_signals)
        results['long_avg_ret_1d'] = long_df['ret_1d'].mean() * 100
        results['long_avg_ret_3d'] = long_df['ret_3d'].mean() * 100
        results['long_win_rate_1d'] = (long_df['ret_1d'] > 0).mean() * 100
        # High quality signals (3+ confluences)
        hq = long_df[long_df['quality'] >= 3]
        results['long_hq_count'] = len(hq)
        results['long_hq_avg_ret_1d'] = hq['ret_1d'].mean() * 100 if len(hq) > 0 else np.nan
        results['long_hq_win_rate'] = (hq['ret_1d'] > 0).mean() * 100 if len(hq) > 0 else np.nan
    else:
        results['long_avg_ret_1d'] = np.nan
        results['long_avg_ret_3d'] = np.nan
        results['long_win_rate_1d'] = np.nan
        results['long_hq_count'] = 0
        results['long_hq_avg_ret_1d'] = np.nan
        results['long_hq_win_rate'] = np.nan

    if short_signals:
        short_df = pd.DataFrame(short_signals)
        results['short_avg_ret_1d'] = short_df['ret_1d'].mean() * 100
        results['short_avg_ret_3d'] = short_df['ret_3d'].mean() * 100
        results['short_win_rate_1d'] = (short_df['ret_1d'] > 0).mean() * 100
        hq = short_df[short_df['quality'] >= 3]
        results['short_hq_count'] = len(hq)
        results['short_hq_avg_ret_1d'] = hq['ret_1d'].mean() * 100 if len(hq) > 0 else np.nan
        results['short_hq_win_rate'] = (hq['ret_1d'] > 0).mean() * 100 if len(hq) > 0 else np.nan
    else:
        results['short_avg_ret_1d'] = np.nan
        results['short_avg_ret_3d'] = np.nan
        results['short_win_rate_1d'] = np.nan
        results['short_hq_count'] = 0
        results['short_hq_avg_ret_1d'] = np.nan
        results['short_hq_win_rate'] = np.nan

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("LIQUIDITY SWEEP + VOLUME PROFILE + VWAP + FVG SCALPING STRATEGY EXPLORATION")
    print("=" * 80)

    all_results = []
    for ticker in SAMPLE_STOCKS:
        try:
            result = analyze_stock(ticker)
            if result:
                all_results.append(result)
                print(f"\n--- {ticker} ---")
                print(f"  Bars: {result['total_bars']} | "
                      f"Bullish FVGs: {result['bullish_fvg_count']} | "
                      f"Bearish FVGs: {result['bearish_fvg_count']}")
                print(f"  Sweep Highs: {result['sweep_highs_count']} | "
                      f"Sweep Lows: {result['sweep_lows_count']}")
                print(f"  LONG signals: {result['long_signals']} | "
                      f"Avg 1d ret: {result['long_avg_ret_1d']:.3f}% | "
                      f"Win rate: {result['long_win_rate_1d']:.1f}%")
                print(f"  SHORT signals: {result['short_signals']} | "
                      f"Avg 1d ret: {result['short_avg_ret_1d']:.3f}% | "
                      f"Win rate: {result['short_win_rate_1d']:.1f}%")
                print(f"  HQ Long (3+ confluences): {result['long_hq_count']} | "
                      f"Avg ret: {result['long_hq_avg_ret_1d']:.3f}% | "
                      f"Win rate: {result['long_hq_win_rate']:.1f}%"
                      if result['long_hq_count'] > 0 else
                      f"  HQ Long: 0")
                print(f"  HQ Short (3+ confluences): {result['short_hq_count']} | "
                      f"Avg ret: {result['short_hq_avg_ret_1d']:.3f}% | "
                      f"Win rate: {result['short_hq_win_rate']:.1f}%"
                      if result['short_hq_count'] > 0 else
                      f"  HQ Short: 0")
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(all_results)
    print(f"\nTotal stocks analyzed: {len(summary_df)}")
    print(f"Total LONG signals: {summary_df['long_signals'].sum()}")
    print(f"Total SHORT signals: {summary_df['short_signals'].sum()}")
    print(f"\nAvg LONG 1d return: {summary_df['long_avg_ret_1d'].mean():.3f}%")
    print(f"Avg LONG win rate: {summary_df['long_win_rate_1d'].mean():.1f}%")
    print(f"Avg SHORT 1d return: {summary_df['short_avg_ret_1d'].mean():.3f}%")
    print(f"Avg SHORT win rate: {summary_df['short_win_rate_1d'].mean():.1f}%")
    
    hq_long = summary_df[summary_df['long_hq_count'] > 0]
    hq_short = summary_df[summary_df['short_hq_count'] > 0]
    print(f"\nHQ LONG avg return: {hq_long['long_hq_avg_ret_1d'].mean():.3f}%")
    print(f"HQ LONG avg win rate: {hq_long['long_hq_win_rate'].mean():.1f}%")
    print(f"HQ SHORT avg return: {hq_short['short_hq_avg_ret_1d'].mean():.3f}%")
    print(f"HQ SHORT avg win rate: {hq_short['short_hq_win_rate'].mean():.1f}%")
