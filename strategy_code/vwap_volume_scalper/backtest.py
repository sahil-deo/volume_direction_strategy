"""
FVG Volume Momentum Scalper — Backtest Engine (v2)
Long-only. Proper portfolio simulation with position limits and fixed allocation.
"""

import pandas as pd
import numpy as np
import os, glob, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from config import *


def load_all_stocks():
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
    print(f"Loaded {len(stocks)} stocks ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    return stocks


def compute_indicators(df):
    df['vol_avg'] = df['volume'].rolling(VOL_AVG_PERIOD).mean()
    df['body_pct'] = (df['close'] - df['open']) / df['open']
    df['ma_trend'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    
    if USE_VWAP_FILTER:
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).rolling(VWAP_PERIOD).sum() / \
                     df['volume'].rolling(VWAP_PERIOD).sum()
    return df


def generate_all_signals(stocks):
    """Generate all signals across all stocks, then sort chronologically."""
    all_signals = []
    total = len(stocks)
    for count, (ticker, df) in enumerate(stocks.items(), 1):
        if count % 100 == 0 or count == total:
            print(f"  Scanning {count}/{total}...")
        df = compute_indicators(df)

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        volumes = df['volume'].values
        vol_avg = df['vol_avg'].values
        body_pct = df['body_pct'].values

        start_ts = pd.Timestamp(TRAIN_START_DATE)

        for i in range(1, len(df) - 2):
            # Only generate signals from start date onward
            if df['date'].iloc[i] < start_ts:
                continue
            # Bullish FVG: candle[i+1].low > candle[i-1].high
            if lows[i + 1] <= highs[i - 1]:
                continue
            gap_size = (lows[i + 1] - highs[i - 1]) / closes[i]
            if gap_size < FVG_MIN_GAP_PCT:
                continue

            # Volume spike + bullish body confirmation on candle i+1 (the breakout)
            # This ensures we aren't using the 'gap' candle itself for confirmation.
            # Check candle i+1 (the one that should close above the gap)
            check_idx = i + 1
            if check_idx >= len(df) - 1:
                continue
            if np.isnan(vol_avg[check_idx]) or vol_avg[check_idx] == 0:
                continue
            
            # Confirmation requires volume spike AND strong bullish close
            has_vol = volumes[check_idx] > VOL_SPIKE_MULTIPLIER * vol_avg[check_idx]
            has_body = body_pct[check_idx] > MIN_BULLISH_BODY_PCT
            
            if has_vol and has_body:
                # Trend filter: close must be above MA
                ma_val = df['ma_trend'].iloc[check_idx]
                if pd.notna(ma_val) and closes[check_idx] < ma_val:
                    continue
                
                # Setup trade targets
                entry_price = closes[check_idx] * (1 + SLIPPAGE_PER_SIDE)
                sl = entry_price * (1 - STOP_LOSS_PCT)
                tp = entry_price * (1 + TAKE_PROFIT_PCT)

                # Pre-simulate the trade outcome (Gap-aware)
                exit_price, exit_date, exit_reason, hold_bars = None, None, None, 0
                for j in range(check_idx + 1, min(check_idx + MAX_HOLD_DAYS + 1, len(df))):
                    bar = df.iloc[j]
                    b_open, b_low, b_high = bar['open'], bar['low'], bar['high']
                    
                    if b_open <= sl:
                        exit_price = b_open * (1 - SLIPPAGE_PER_SIDE)
                        exit_date = bar['date']
                        exit_reason = 'stopped_out_gap'
                        hold_bars = j - check_idx
                        break
                    if b_open >= tp:
                        exit_price = b_open * (1 - SLIPPAGE_PER_SIDE)
                        exit_date = bar['date']
                        exit_reason = 'take_profit_gap'
                        hold_bars = j - check_idx
                        break
                    if b_low <= sl:
                        exit_price = sl
                        exit_date = bar['date']
                        exit_reason = 'stopped_out'
                        hold_bars = j - check_idx
                        break
                    if b_high >= tp:
                        exit_price = tp
                        exit_date = bar['date']
                        exit_reason = 'take_profit'
                        hold_bars = j - check_idx
                        break
                        
                if exit_price is None:
                    last_idx = min(check_idx + MAX_HOLD_DAYS, len(df) - 1)
                    exit_price = df['close'].iloc[last_idx] * (1 - SLIPPAGE_PER_SIDE)
                    exit_date = df['date'].iloc[last_idx]
                    exit_reason = 'time_exit'
                    hold_bars = last_idx - check_idx

                raw_ret = (exit_price - entry_price) / entry_price
                net_ret = raw_ret - TOTAL_ROUND_TRIP_COST

                all_signals.append({
                    'symbol': ticker,
                    'entry_date': df['date'].iloc[check_idx],
                    'exit_date': exit_date,
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(exit_price, 4),
                    'gap_size_pct': round(gap_size * 100, 2),
                    'volume_ratio': round(volumes[check_idx] / vol_avg[check_idx], 2),
                    'body_pct': round(body_pct[check_idx] * 100, 2),
                    'pnl_gross': round(raw_ret, 6),
                    'pnl_net': round(net_ret, 6),
                    'exit_reason': exit_reason,
                    'stopped_out': exit_reason in {'stopped_out', 'stopped_out_gap'},
                    'hold_bars': hold_bars,
                })
                break  # only one entry per FVG setup

    df_signals = pd.DataFrame(all_signals)
    if not df_signals.empty:
        df_signals = df_signals.sort_values('entry_date').reset_index(drop=True)
    return df_signals


def simulate_portfolio(trades_df, stocks_dict):
    """
    Simulate portfolio with position limits and TRUE Daily MTM Equity Tracking.
    Uses freq='B' (Business Days) and looks up daily closes for accurate Sharpe.
    """
    trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
    # Create a Master Trading Calendar using the stock with the most history
    # This avoids 'noisy' dates from bad data/corporate actions
    master_ticker = max(stocks_dict, key=lambda k: len(stocks_dict[k]))
    all_dates = sorted(pd.to_datetime(stocks_dict[master_ticker]['date']))
    
    # Filter to requested backtest range
    start_dt = pd.to_datetime(trades_df['entry_date'].min())
    end_dt = pd.to_datetime(trades_df['exit_date'].max())
    all_dates = [d for d in all_dates if start_dt <= d <= end_dt]
    
    # Pre-index close prices for fast MTM lookup
    close_map = {}
    for ticker, df in stocks_dict.items():
        close_map[ticker] = dict(zip(pd.to_datetime(df['date']), df['close']))
    
    capital = INITIAL_CAPITAL
    open_positions = []  # list of {symbol, exit_date, allocated, pnl_net, entry_price}
    executed_trades = []
    daily_equity = []
    
    pos_size_frac = 1.0 / MAX_SIMULTANEOUS_POSITIONS

    # Group signals by date
    signals_by_date = {pd.to_datetime(d): group for d, group in trades_df.groupby('entry_date')}

    for current_date in all_dates:
        # 1. Close positions that exit on or before this date
        still_open = []
        for pos in open_positions:
            if pos['exit_date'] <= current_date:
                # Realize the P&L
                capital += pos['allocated'] * (1 + pos['pnl_net'])
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Open new positions
        if current_date in signals_by_date:
            for _, trade in signals_by_date[current_date].iterrows():
                if len(open_positions) < MAX_SIMULTANEOUS_POSITIONS:
                    alloc = capital * pos_size_frac
                    if alloc > 1000:
                        capital -= alloc
                        open_positions.append({
                            'symbol': trade['symbol'],
                            'exit_date': pd.to_datetime(trade['exit_date']),
                            'allocated': alloc,
                            'pnl_net': trade['pnl_net'],
                            'entry_price': trade['entry_price']
                        })
                        
                        executed_trade = trade.to_dict()
                        executed_trade['allocated_capital'] = round(alloc, 2)
                        executed_trade['trade_pnl_rupees'] = round(alloc * trade['pnl_net'], 2)
                        executed_trades.append(executed_trade)

        # 3. TRUE Mark-to-Market (MTM)
        unrealized_equity = 0
        for pos in open_positions:
            current_close = close_map[pos['symbol']].get(current_date, pos['entry_price'])
            unrealized_gain_pct = (current_close / pos['entry_price']) - 1
            unrealized_equity += pos['allocated'] * (1 + unrealized_gain_pct)

        total_equity = capital + unrealized_equity
        daily_equity.append({'date': current_date, 'equity': total_equity})

    # Consistency Fix: Close all remaining open positions at their simulated exit price
    # ensuring final_equity and the last point of daily_equity match perfectly.
    for pos in open_positions:
        capital += pos['allocated'] * (1 + pos['pnl_net'])
    
    if daily_equity:
        daily_equity[-1]['equity'] = capital
    
    final_equity = capital
    return pd.DataFrame(executed_trades), final_equity, pd.DataFrame(daily_equity)


def compute_metrics(trades_df, final_equity, daily_equity_df):
    if trades_df.empty:
        print("No trades!"); return {}

    # Daily Returns from Equity Curve
    daily_equity_df['returns'] = daily_equity_df['equity'].pct_change().fillna(0)
    
    n = len(trades_df)
    wins = trades_df[trades_df['pnl_net'] > 0]
    losses = trades_df[trades_df['pnl_net'] <= 0]

    win_rate = len(wins) / n
    avg_win = wins['pnl_net'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_net'].mean() if len(losses) > 0 else 0
    pf = abs(wins['pnl_net'].sum() / losses['pnl_net'].sum()) if losses['pnl_net'].sum() != 0 else np.inf

    total_return = final_equity / INITIAL_CAPITAL - 1
    
    # CAGR using actual calendar days
    days = (daily_equity_df['date'].max() - daily_equity_df['date'].min()).days
    years = max(days / 365.25, 0.1)
    cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1

    # Sharpe/Sortino using Daily Returns (Professional Standard)
    daily_rets = daily_equity_df['returns'].values
    vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / vol if vol > 0 else 0
    
    downside_rets = daily_rets[daily_rets < 0]
    sortino = (daily_rets.mean() * 252) / (downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 0 else 0

    # Max Drawdown from Daily Equity
    peak = daily_equity_df['equity'].cummax()
    dd = (daily_equity_df['equity'] - peak) / peak
    max_dd = dd.min()
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    # Yearly Returns (Corrected: based on Jan 1st Equity)
    daily_equity_df['year'] = daily_equity_df['date'].dt.year
    yearly = {}
    for yr, group in daily_equity_df.groupby('year'):
        y_start = group['equity'].iloc[0]
        y_end = group['equity'].iloc[-1]
        yearly[int(yr)] = round((y_end / y_start) - 1, 4)

    # Additional Metrics
    trades_sorted = trades_df.sort_values('exit_date')
    stopped = trades_df['stopped_out'].sum()
    metrics = {
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'universe': 'NIFTY_750',
        'start_date': str(trades_sorted['entry_date'].min().date()),
        'end_date': str(trades_sorted['exit_date'].max().date()),
        'strategy': 'FVG_Volume_Momentum_v2',
        'direction': 'long_only',
        'initial_capital': INITIAL_CAPITAL,
        'final_equity': round(final_equity, 2),
        'fvg_min_gap_pct': FVG_MIN_GAP_PCT,
        'vol_spike_mult': VOL_SPIKE_MULTIPLIER,
        'min_body_pct': MIN_BULLISH_BODY_PCT,
        'stop_loss_pct': STOP_LOSS_PCT,
        'take_profit_pct': TAKE_PROFIT_PCT,
        'max_hold_days': MAX_HOLD_DAYS,
        'max_positions': MAX_SIMULTANEOUS_POSITIONS,
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
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 6),
        'avg_loss': round(avg_loss, 6),
        'profit_factor': round(pf, 4),
        'avg_hold_bars': round(trades_df['hold_bars'].mean(), 2),
        'stopped_out': int(stopped),
        'stop_rate': round(stopped / n, 4),
        'best_trade': round(trades_df['pnl_net'].max(), 6),
        'worst_trade': round(trades_df['pnl_net'].min(), 6),
        'median_trade': round(trades_df['pnl_net'].median(), 6),
    }
    for yr in sorted(yearly.keys()):
        metrics[str(yr)] = yearly[yr]

    return metrics


def main():
    print("=" * 70)
    print("FVG VOLUME MOMENTUM SCALPER v2 — Proper Portfolio Simulation")
    print("=" * 70)

    stocks = load_all_stocks()
    print(f"\nGenerating signals across {len(stocks)} stocks...")
    all_trades = generate_all_signals(stocks)
    print(f"Raw signals found: {len(all_trades)}")

    if all_trades.empty:
        print("No signals."); return

    print(f"\nSimulating portfolio (max {MAX_SIMULTANEOUS_POSITIONS} positions)...")
    executed_trades, final_equity, daily_equity_df = simulate_portfolio(all_trades, stocks)
    print(f"Executed trades: {len(executed_trades)}")
    print(f"Final equity: ₹{final_equity:,.2f} (from ₹{INITIAL_CAPITAL:,.2f})")

    metrics = compute_metrics(executed_trades, final_equity, daily_equity_df)

    # ─── Saving Results ───────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trades_dir = os.path.join(RESULTS_DIR, "trades")
    os.makedirs(trades_dir, exist_ok=True)

    # Generate unique filename for trades based on config
    cfg_name = f"SL{int(STOP_LOSS_PCT*100)}_TP{int(TAKE_PROFIT_PCT*100)}_G{int(FVG_MIN_GAP_PCT*1000)}"
    timestamp = datetime.now().strftime("%H%M%S")
    trade_file = f"trades_{cfg_name}_{timestamp}.csv"
    
    trade_path = os.path.join(trades_dir, trade_file)
    executed_trades.sort_values('entry_date').to_csv(trade_path, index=False)

    # Append to metrics.csv
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved Trade Log: {trade_path}")
    print(f"Updated Metrics: {metrics_path}")
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for k in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'sharpe',
              'sortino', 'calmar', 'trades', 'win_rate', 'avg_win', 'avg_loss',
              'profit_factor', 'avg_hold_bars', 'stopped_out', 'stop_rate',
              'best_trade', 'worst_trade']:
        print(f"  {k:20s}: {metrics.get(k, 'N/A')}")

    print("\nYearly Returns (YoY):")
    for k, v in metrics.items():
        if k.isdigit():
            print(f"  {k}: {v * 100:+.2f}%")


if __name__ == '__main__':
    main()
