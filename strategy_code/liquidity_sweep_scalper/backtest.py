"""
Liquidity Sweep Reversal Scalper (LSRS) — Full Backtest Engine
Generates: metrics.csv + trades.csv in results folder
"""

import pandas as pd
import numpy as np
import os, glob, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from config import *


def load_all_stocks():
    """Load all CSVs, filter by date and liquidity."""
    files = glob.glob(os.path.join(DATA_DIR, "*_daily.csv"))
    stocks = {}
    for f in files:
        ticker = os.path.basename(f).replace("_daily.csv", "")
        df = pd.read_csv(f, parse_dates=['date'])
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'] <= TRAIN_END_DATE].copy()
        if len(df) < MIN_BARS_REQUIRED:
            continue
        if df['volume'].tail(60).mean() < MIN_AVG_VOLUME:
            continue
        stocks[ticker] = df
    print(f"Loaded {len(stocks)} stocks (filtered by date <= {TRAIN_END_DATE}, min bars, min volume)")
    return stocks


def compute_indicators(df):
    """Add VWAP, POC, FVG flags to dataframe (vectorized for speed)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = tp * df['volume']
    df['vwap'] = tp_vol.rolling(VWAP_PERIOD).sum() / df['volume'].rolling(VWAP_PERIOD).sum()

    # Volume Profile POC — vectorized with numpy
    lows = df['low'].values
    highs = df['high'].values
    volumes = df['volume'].values
    n = len(df)
    pocs = np.full(n, np.nan)

    for i in range(VP_LOOKBACK, n):
        w_lo = lows[i - VP_LOOKBACK:i]
        w_hi = highs[i - VP_LOOKBACK:i]
        w_vol = volumes[i - VP_LOOKBACK:i]
        lo, hi = w_lo.min(), w_hi.max()
        if hi == lo:
            pocs[i] = lo; continue
        bins = np.linspace(lo, hi, VP_NUM_BINS + 1)
        # Vectorized: for each bin, check overlap with all candles at once
        bin_lo = bins[:-1]  # (VP_NUM_BINS,)
        bin_hi = bins[1:]   # (VP_NUM_BINS,)
        # overlap: candle_low <= bin_hi AND candle_high >= bin_lo
        overlap = (w_lo[:, None] <= bin_hi[None, :]) & (w_hi[:, None] >= bin_lo[None, :])
        bv = (overlap * (w_vol[:, None] / VP_NUM_BINS)).sum(axis=0)
        best = np.argmax(bv)
        pocs[i] = (bin_lo[best] + bin_hi[best]) / 2
    df['poc'] = pocs

    # FVG flags — vectorized with shifted arrays
    hi_prev = df['high'].shift(1)
    lo_next = df['low'].shift(-1)
    hi_next = df['high'].shift(-1)
    lo_prev = df['low'].shift(1)
    df['bullish_fvg'] = (lo_next > hi_prev).fillna(False)
    df['bearish_fvg'] = (hi_next < lo_prev).fillna(False)

    # Rolling volume average
    df['vol_avg'] = df['volume'].rolling(VOL_AVG_PERIOD).mean()

    # Rolling high/low for sweep detection
    df['rolling_high'] = df['high'].rolling(SWEEP_LOOKBACK).max().shift(1)
    df['rolling_low'] = df['low'].rolling(SWEEP_LOOKBACK).min().shift(1)

    return df


def score_signal(row, df, idx, direction):
    """Return confluence score (0-5) for a signal at given index."""
    score = 0
    # Factor 1: Sweep is already confirmed by caller (always 1)
    score += 1

    # Factor 2: VWAP proximity
    if pd.notna(row['vwap']) and row['vwap'] > 0:
        if abs(row['close'] - row['vwap']) / row['vwap'] < VWAP_PROXIMITY_PCT:
            score += 1

    # Factor 3: POC proximity
    if pd.notna(row['poc']) and row['poc'] > 0:
        if abs(row['close'] - row['poc']) / row['poc'] < POC_PROXIMITY_PCT:
            score += 1

    # Factor 4: FVG within lookback
    start = max(0, idx - FVG_LOOKBACK_BARS)
    col = 'bullish_fvg' if direction == 'long' else 'bearish_fvg'
    if df[col].iloc[start:idx+1].any():
        score += 1

    # Factor 5: Volume spike
    if pd.notna(row['vol_avg']) and row['vol_avg'] > 0:
        if row['volume'] > VOL_SPIKE_MULTIPLIER * row['vol_avg']:
            score += 1

    return score


def run_backtest(stocks):
    """Run the full backtest across all stocks."""
    all_trades = []
    total = len(stocks)

    for count, (ticker, df) in enumerate(stocks.items(), 1):
        if count % 50 == 0 or count == total:
            print(f"  Processing {count}/{total}: {ticker}...")
        df = compute_indicators(df)

        for i in range(SWEEP_LOOKBACK + 1, len(df) - 5):
            row = df.iloc[i]

            # === LONG: sweep of lows ===
            if (pd.notna(row['rolling_low']) and
                row['low'] < row['rolling_low'] and
                row['close'] > row['open']):

                sc = score_signal(row, df, i, 'long')
                if sc >= MIN_CONFLUENCE_SCORE:
                    size_mult = 0.5 if sc == HALF_SIZE_THRESHOLD else 1.0
                    entry_price = row['close'] * (1 + SLIPPAGE_PER_SIDE)
                    sl_price = entry_price * (1 - STOP_LOSS_PCT)
                    tp1_price = entry_price * (1 + TAKE_PROFIT_1_PCT)
                    tp2_price = entry_price * (1 + TAKE_PROFIT_2_PCT)

                    # Simulate forward
                    trade = _simulate_trade(df, i, entry_price, sl_price,
                                            tp1_price, tp2_price, 'long',
                                            ticker, sc, size_mult, row['date'])
                    if trade:
                        all_trades.append(trade)

            # === SHORT: sweep of highs ===
            if (pd.notna(row['rolling_high']) and
                row['high'] > row['rolling_high'] and
                row['close'] < row['open']):

                sc = score_signal(row, df, i, 'short')
                if sc >= MIN_CONFLUENCE_SCORE:
                    size_mult = 0.5 if sc == HALF_SIZE_THRESHOLD else 1.0
                    entry_price = row['close'] * (1 - SLIPPAGE_PER_SIDE)
                    sl_price = entry_price * (1 + STOP_LOSS_PCT)
                    tp1_price = entry_price * (1 - TAKE_PROFIT_1_PCT)
                    tp2_price = entry_price * (1 - TAKE_PROFIT_2_PCT)

                    trade = _simulate_trade(df, i, entry_price, sl_price,
                                            tp1_price, tp2_price, 'short',
                                            ticker, sc, size_mult, row['date'])
                    if trade:
                        all_trades.append(trade)

    return pd.DataFrame(all_trades)


def _simulate_trade(df, entry_idx, entry_price, sl, tp1, tp2,
                     direction, ticker, score, size_mult, entry_date):
    """Simulate a trade bar-by-bar. Dual TP with partial exits."""
    tp1_hit = False
    exit_price = None
    exit_date = None
    exit_reason = None
    max_hold = 10  # max bars to hold

    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        bar = df.iloc[j]

        if direction == 'long':
            # Check SL first (worst case)
            if bar['low'] <= sl:
                if tp1_hit:
                    exit_price = sl
                    exit_reason = 'sl_after_tp1'
                else:
                    exit_price = sl
                    exit_reason = 'stopped_out'
                exit_date = bar['date']
                break
            # Check TP1
            if not tp1_hit and bar['high'] >= tp1:
                tp1_hit = True
            # Check TP2
            if bar['high'] >= tp2:
                exit_price = tp2
                exit_reason = 'tp2'
                exit_date = bar['date']
                break
        else:  # short
            if bar['high'] >= sl:
                if tp1_hit:
                    exit_price = sl
                    exit_reason = 'sl_after_tp1'
                else:
                    exit_price = sl
                    exit_reason = 'stopped_out'
                exit_date = bar['date']
                break
            if not tp1_hit and bar['low'] <= tp1:
                tp1_hit = True
            if bar['low'] <= tp2:
                exit_price = tp2
                exit_reason = 'tp2'
                exit_date = bar['date']
                break

    # Time exit if no SL/TP hit
    if exit_price is None:
        last_idx = min(entry_idx + max_hold, len(df) - 1)
        bar = df.iloc[last_idx]
        exit_price = bar['close'] * (1 - SLIPPAGE_PER_SIDE if direction == 'long'
                                     else 1 + SLIPPAGE_PER_SIDE)
        exit_date = bar['date']
        exit_reason = 'time_exit'

    # Calculate returns
    if direction == 'long':
        # Blended return considering partial TP1 exit
        if tp1_hit and exit_reason != 'stopped_out':
            ret_tp1 = (tp1 - entry_price) / entry_price  # 50% at TP1
            ret_rest = (exit_price - entry_price) / entry_price  # 50% at final
            raw_ret = 0.5 * ret_tp1 + 0.5 * ret_rest
        else:
            raw_ret = (exit_price - entry_price) / entry_price
    else:
        if tp1_hit and exit_reason != 'stopped_out':
            ret_tp1 = (entry_price - tp1) / entry_price
            ret_rest = (entry_price - exit_price) / entry_price
            raw_ret = 0.5 * ret_tp1 + 0.5 * ret_rest
        else:
            raw_ret = (entry_price - exit_price) / entry_price

    raw_ret *= size_mult
    tcost = TOTAL_ROUND_TRIP_COST * size_mult
    net_ret = raw_ret - tcost
    hold_days = (exit_date - entry_date).days if hasattr(exit_date, 'day') else 0

    return {
        'symbol': ticker,
        'direction': direction,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'entry_price': round(entry_price, 4),
        'exit_price': round(exit_price, 4),
        'confluence_score': score,
        'size_multiplier': size_mult,
        'pnl_gross': round(raw_ret, 6),
        'pnl_net': round(net_ret, 6),
        'exit_reason': exit_reason,
        'stopped_out': exit_reason == 'stopped_out',
        'hold_days': hold_days,
    }


def compute_metrics(trades_df):
    """Compute strategy-level metrics from trade log."""
    if trades_df.empty:
        print("No trades generated!"); return {}

    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['year'] = trades_df['exit_date'].dt.year

    n = len(trades_df)
    wins = trades_df[trades_df['pnl_net'] > 0]
    losses = trades_df[trades_df['pnl_net'] <= 0]

    win_rate = len(wins) / n if n > 0 else 0
    avg_win = wins['pnl_net'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_net'].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins['pnl_net'].sum() / losses['pnl_net'].sum()) if losses['pnl_net'].sum() != 0 else np.inf

    # Build equity curve (simple: compound returns)
    trades_sorted = trades_df.sort_values('exit_date')
    equity = [INITIAL_CAPITAL]
    for _, t in trades_sorted.iterrows():
        eq = equity[-1] * (1 + t['pnl_net'])
        equity.append(eq)
    equity = np.array(equity)

    total_return = (equity[-1] / equity[0]) - 1
    years = max((trades_sorted['exit_date'].max() - trades_sorted['entry_date'].min()).days / 365.25, 0.1)
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1

    # Daily returns proxy (by trade)
    rets = trades_sorted['pnl_net'].values
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    downside = rets[rets < 0]
    sortino = (rets.mean() * 252) / (downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    stopped = trades_df['stopped_out'].sum()
    stop_rate = stopped / n if n > 0 else 0

    # Yearly returns
    yearly = {}
    for yr, grp in trades_df.groupby('year'):
        yr_eq = INITIAL_CAPITAL
        for _, t in grp.sort_values('exit_date').iterrows():
            yr_eq *= (1 + t['pnl_net'])
        yearly[int(yr)] = round((yr_eq / INITIAL_CAPITAL) - 1, 4)

    # Long vs short breakdown
    longs = trades_df[trades_df['direction'] == 'long']
    shorts = trades_df[trades_df['direction'] == 'short']

    metrics = {
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'universe': 'NIFTY_750',
        'start_date': str(trades_sorted['entry_date'].min().date()),
        'end_date': str(trades_sorted['exit_date'].max().date()),
        'sweep_lookback': SWEEP_LOOKBACK,
        'vwap_period': VWAP_PERIOD,
        'min_confluence': MIN_CONFLUENCE_SCORE,
        'stop_loss_pct': STOP_LOSS_PCT,
        'tp1_pct': TAKE_PROFIT_1_PCT,
        'tp2_pct': TAKE_PROFIT_2_PCT,
        'tcost_per_side': TCOST_PER_SIDE,
        'slippage_per_side': SLIPPAGE_PER_SIDE,
        'total_return': round(total_return, 6),
        'cagr': round(cagr, 6),
        'volatility': round(vol, 6),
        'max_drawdown': round(max_dd, 6),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'calmar': round(calmar, 4),
        'trades': n,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'win_rate': round(win_rate, 4),
        'long_win_rate': round(len(longs[longs['pnl_net'] > 0]) / max(len(longs), 1), 4),
        'short_win_rate': round(len(shorts[shorts['pnl_net'] > 0]) / max(len(shorts), 1), 4),
        'avg_win': round(avg_win, 6),
        'avg_loss': round(avg_loss, 6),
        'profit_factor': round(profit_factor, 4),
        'avg_hold_days': round(trades_df['hold_days'].mean(), 2),
        'stopped_out': int(stopped),
        'stop_rate': round(stop_rate, 4),
        'best_trade': round(trades_df['pnl_net'].max(), 6),
        'worst_trade': round(trades_df['pnl_net'].min(), 6),
        'median_trade': round(trades_df['pnl_net'].median(), 6),
    }

    # Add yearly columns
    for yr in sorted(yearly.keys()):
        metrics[str(yr)] = yearly[yr]

    return metrics


def main():
    print("=" * 70)
    print("LSRS BACKTEST — Liquidity Sweep Reversal Scalper")
    print("=" * 70)

    stocks = load_all_stocks()
    print(f"\nRunning backtest on {len(stocks)} stocks up to {TRAIN_END_DATE}...")

    trades_df = run_backtest(stocks)
    print(f"\nTotal trades generated: {len(trades_df)}")

    if trades_df.empty:
        print("No trades — check parameters."); return

    # Compute metrics
    metrics = compute_metrics(trades_df)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # trades.csv
    trades_path = os.path.join(RESULTS_DIR, "trades.csv")
    trades_df = trades_df.sort_values('entry_date')
    trades_df.to_csv(trades_path, index=False)
    print(f"\nTrades saved to: {trades_path}")

    # metrics.csv
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    for k in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'sharpe',
              'sortino', 'calmar', 'trades', 'long_trades', 'short_trades',
              'win_rate', 'long_win_rate', 'short_win_rate', 'avg_win',
              'avg_loss', 'profit_factor', 'avg_hold_days', 'stopped_out',
              'stop_rate', 'best_trade', 'worst_trade']:
        print(f"  {k:20s}: {metrics.get(k, 'N/A')}")

    print("\nYearly Returns:")
    for k, v in metrics.items():
        if k.isdigit():
            pct = v * 100
            print(f"  {k}: {pct:+.2f}%")


if __name__ == '__main__':
    main()
