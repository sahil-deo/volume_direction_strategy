"""
RSI(2) Mean Reversion Scalping Strategy — Backtester
======================================================
Strategy Logic:
  Signal : Today's close > MA(200)  AND  today's RSI(2) < RSI_ENTRY_THRESHOLD
  Entry  : Next day's OPEN (no lookahead bias)
  Exit   : Signal on day i (RSI(2) > exit threshold OR close > MA(5)) → exit at day i+1 OPEN
  Stop   : If next open gaps down past stop level, stopped out at that open

Portfolio:
  - Up to PORTFOLIO_SLOTS simultaneous positions (equal-weight)
  - Daily portfolio return = sum of active trade returns / PORTFOLIO_SLOTS

Outputs (per run):
  - metrics.csv      : APPENDED — one row per run (params + all metrics + yearly cols)
                       All metrics are NET (after transaction costs + slippage).
  - trade_log.csv    : OVERWRITTEN — all trades sorted by exit_date
                       Contains both trade_return (gross) and trade_return_net (after costs).
  - equity_curve.png : OVERWRITTEN — charts (equity curve is net)
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ── edit everything here
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FOLDER        = '/Users/sahil/Developer/Quantitative Finance/data/YF Stock Data Nifty 500 2026'                  # folder containing *_daily.csv files
OUTPUT_FOLDER      = '/Users/sahil/Developer/Quantitative Finance/aakash/results/mean_resersion_rsi'             # folder for output files
EQUITY_CURVE_FILE  = "equity_curve.png"
TRADE_LOG_FILE     = "trade_log.csv"
METRICS_FILE       = "metrics.csv"           # appended each run (never overwritten)

# ── Universe label (stored in metrics row for identification) ──────────────────
UNIVERSE_LABEL     = "NIFTY_500_26"             # descriptive name for this run's universe

# ── Indicator Parameters ───────────────────────────────────────────────────────
RSI_PERIOD         = 2                       # RSI look-back period
MA_SHORT           = 5                       # short MA period (exit signal)
MA_LONG            = 200                     # long  MA period (trend filter)

# ── Entry / Exit Thresholds ───────────────────────────────────────────────────
RSI_ENTRY_THRESHOLD  = 10                    # buy  when RSI(2) < this
RSI_EXIT_THRESHOLD   = 90                    # sell when RSI(2) > this

# ── Risk Management ────────────────────────────────────────────────────────────
STOP_LOSS_PCT      = -0.02                   # -2% hard stop (clips trade PnL)

# ── Transaction Costs & Slippage ───────────────────────────────────────────────
# Applied per side (entry AND exit), as a fraction of trade price.
# e.g. 0.001 = 0.1% per side → 0.2% round-trip total cost
TRANSACTION_COST_PER_SIDE = 0.001           # brokerage + STT + exchange fees (per side)
SLIPPAGE_PER_SIDE         = 0.0005          # execution slippage assumption (per side)
# Combined round-trip cost = 2 * (TRANSACTION_COST_PER_SIDE + SLIPPAGE_PER_SIDE)

# ── Portfolio ──────────────────────────────────────────────────────────────────
PORTFOLIO_SLOTS    = 100                     # max simultaneous positions (equal-weight)

# ── Backtest Window ────────────────────────────────────────────────────────────
START_DATE         = "2026-01-01"            # inclusive; set None for full history
END_DATE           = None                    # inclusive; set None for full history

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_CSV      = None                    # path to benchmark CSV; None to skip

# ── Misc ───────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 100_000                 # for equity-curve scaling
RISK_FREE_RATE     = 0.06                    # annual, for Sharpe / Sortino
TRADING_DAYS_YEAR  = 252


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_and_prepare(path: str):
    """Load a single symbol CSV, compute indicators, return cleaned DataFrame."""
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None

    df = df.rename(columns={"date": "Date", "close": "Close",
                             "open": "Open", "high": "High",
                             "low": "Low", "volume": "Volume"})

    if not {"Date", "Close"}.issubset(df.columns):
        return None

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])

    # Compute indicators on FULL history first (no date filter yet).
    # This ensures MA and RSI are properly warmed up at the start of the
    # backtest window — avoids understated MA values from insufficient history.
    if len(df) < MA_LONG + RSI_PERIOD + 5:
        return None

    df["RSI"]      = compute_rsi(df["Close"], RSI_PERIOD)
    df["MA_SHORT"] = df["Close"].rolling(MA_SHORT).mean()
    df["MA_LONG"]  = df["Close"].rolling(MA_LONG).mean()

    df = df.dropna(subset=["RSI", "MA_SHORT", "MA_LONG"]).reset_index(drop=True)

    # Apply date window AFTER indicators are computed
    if START_DATE:
        df = df[df["Date"] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        df = df[df["Date"] <= pd.Timestamp(END_DATE)]

    if df.empty:
        return None

    return df.reset_index(drop=True)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate trades on a single symbol. Returns completed trade records.

    Execution model (no lookahead bias):
      - Signal generated using day i's close/indicators
      - Entry and exit executed at day i+1's OPEN
      - Stop-loss: if (open_i+1 - entry_price) / entry_price <= STOP_LOSS_PCT,
        the trade is stopped out at that same open (gap protection).
        Otherwise stop is checked on subsequent days against that day's open.
    """
    records     = []
    in_trade    = False
    entry_price = None
    entry_date  = None

    closes = df["Close"].values
    opens  = df["Open"].values
    rsi    = df["RSI"].values
    ma_s   = df["MA_SHORT"].values
    ma_l   = df["MA_LONG"].values
    dates  = df["Date"].values
    n      = len(df)

    for i in range(n - 1):   # stop at n-1 since execution is at i+1
        # ── values known at end of day i (signal day) ──
        c_i  = closes[i]
        r_i  = rsi[i]
        ms_i = ma_s[i]
        ml_i = ma_l[i]

        # ── execution day i+1 ──
        o_next  = opens[i + 1]
        dt_next = dates[i + 1]

        if in_trade:
            # Evaluate exit using day i's indicators (known before i+1 open)
            exit_signal = (r_i > RSI_EXIT_THRESHOLD) or (c_i > ms_i)

            # PnL based on next day's open (actual execution price)
            pnl_gross = (o_next - entry_price) / entry_price

            # Stop triggered if pnl <= stop threshold (includes gap-down opens)
            stopped = pnl_gross <= STOP_LOSS_PCT
            if stopped:
                pnl_gross = STOP_LOSS_PCT   # cap at stop; real fill may be worse

            if exit_signal or stopped:
                round_trip_cost = 2 * (TRANSACTION_COST_PER_SIDE + SLIPPAGE_PER_SIDE)
                pnl_net = pnl_gross - round_trip_cost
                records.append({
                    "entry_date"      : entry_date,
                    "exit_date"       : dt_next,
                    "entry_price"     : round(entry_price, 4),
                    "exit_price"      : round(o_next, 4),
                    "trade_return"    : round(pnl_gross, 6),
                    "trade_return_net": round(pnl_net, 6),
                    "stopped_out"     : stopped,
                    "hold_days"       : (pd.Timestamp(dt_next) - pd.Timestamp(entry_date)).days,
                })
                in_trade = False
                entry_price = entry_date = None

        # Entry: signal on day i → execute at day i+1 open
        # Only enter if not already in a trade (checked after potential exit above)
        if not in_trade and (c_i > ml_i) and (r_i < RSI_ENTRY_THRESHOLD):
            in_trade    = True
            entry_price = o_next
            entry_date  = dt_next

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio_returns(all_trades: pd.DataFrame,
                             trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Book full trade return on exit date; scale by 1/PORTFOLIO_SLOTS.
    Uses trade_return_net (after transaction costs + slippage) for all metrics."""
    if all_trades.empty:
        return pd.Series(0.0, index=trading_dates)
    daily = all_trades.groupby("exit_date")["trade_return_net"].sum() / PORTFOLIO_SLOTS
    return daily.reindex(trading_dates, fill_value=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def annual_returns(equity_curve: pd.Series) -> pd.Series:
    """Returns a Series indexed by integer year."""
    yearly      = equity_curve.resample("YE").last()
    yearly_prev = yearly.shift(1)
    yearly_prev.iloc[0] = equity_curve.iloc[0]
    ret = yearly / yearly_prev - 1
    ret.index = ret.index.year
    ret.index.name = "year"
    return ret


def compute_metrics(daily_returns: pd.Series,
                    equity_curve:  pd.Series,
                    all_trades:    pd.DataFrame) -> dict:
    rf_daily     = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_YEAR) - 1
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years      = len(daily_returns) / TRADING_DAYS_YEAR
    cagr         = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    excess   = daily_returns - rf_daily
    sharpe   = (excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_YEAR)
                if excess.std() > 0 else 0)

    downside = daily_returns[daily_returns < rf_daily] - rf_daily
    sortino  = ((daily_returns.mean() - rf_daily) / downside.std()
                * np.sqrt(TRADING_DAYS_YEAR)
                if len(downside) > 0 and downside.std() > 0 else 0)

    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd   = drawdown.min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0

    wins     = all_trades[all_trades["trade_return_net"] > 0]
    losses   = all_trades[all_trades["trade_return_net"] <= 0]
    stops    = all_trades[all_trades["stopped_out"]]
    loss_sum = losses["trade_return_net"].sum()
    win_sum  = wins["trade_return_net"].sum()
    n        = len(all_trades)

    # coverage: % of trading days where at least one trade exits (portfolio is active)
    coverage = round((daily_returns != 0).mean() * 100, 2)

    return dict(
        # run metadata
        run_timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        universe        = UNIVERSE_LABEL,
        start_date      = str(daily_returns.index[0].date()),
        end_date        = str(daily_returns.index[-1].date()),
        # strategy params
        rsi_period      = RSI_PERIOD,
        ma_short        = MA_SHORT,
        ma_long         = MA_LONG,
        rsi_entry       = RSI_ENTRY_THRESHOLD,
        rsi_exit        = RSI_EXIT_THRESHOLD,
        stop_loss_pct   = abs(STOP_LOSS_PCT) * 100,
        portfolio_slots = PORTFOLIO_SLOTS,
        tcost_per_side  = TRANSACTION_COST_PER_SIDE * 100,
        slippage_per_side = SLIPPAGE_PER_SIDE * 100,
        # return metrics (all NET)
        total_return    = round(total_return, 6),
        cagr            = round(cagr, 6),
        volatility      = round(daily_returns.std() * np.sqrt(TRADING_DAYS_YEAR), 6),
        best_day        = round(daily_returns.max(), 6),
        worst_day       = round(daily_returns.min(), 6),
        # risk metrics (all NET)
        max_drawdown    = round(max_dd, 6),
        sharpe          = round(sharpe, 4),
        sortino         = round(sortino, 4),
        calmar          = round(calmar, 4),
        # trade stats (all NET)
        trades          = n,
        win_rate        = round(len(wins) / n, 4)                          if n > 0 else 0,
        avg_win         = round(wins["trade_return_net"].mean(), 6)         if len(wins)   > 0 else 0,
        avg_loss        = round(losses["trade_return_net"].mean(), 6)       if len(losses) > 0 else 0,
        profit_factor   = round(win_sum / abs(loss_sum), 4)                 if loss_sum != 0 else float("inf"),
        avg_hold_days   = round(all_trades["hold_days"].mean(), 2)          if n > 0 else 0,
        stopped_out     = len(stops),
        stop_rate       = round(len(stops) / n, 4)                          if n > 0 else 0,
        best_trade      = round(all_trades["trade_return_net"].max(), 6)    if n > 0 else 0,
        worst_trade     = round(all_trades["trade_return_net"].min(), 6)    if n > 0 else 0,
        median_trade    = round(all_trades["trade_return_net"].median(), 6) if n > 0 else 0,
        coverage        = coverage,
    )


def append_metrics_row(metrics: dict, ann_ret: pd.Series, metrics_path: str) -> None:
    """
    Append one row to metrics.csv.
    Yearly return columns are dynamic (e.g. '2023', '2024').
    Missing year cols in older rows are left as NaN — schema stays consistent.
    """
    row = dict(metrics)
    for year, val in ann_ret.items():
        row[str(year)] = round(val, 4)

    new_df = pd.DataFrame([row])

    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    combined.to_csv(metrics_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(equity_curve:    pd.Series,
                 daily_returns:   pd.Series,
                 ann_ret:         pd.Series,
                 benchmark_curve,
                 output_path:     str) -> None:

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT = "#00d4ff"
    BENCH  = "#f7a440"
    GREEN  = "#26a69a"
    RED    = "#ef5350"
    GREY   = "#9e9e9e"
    BG     = "#1a1d27"
    TEXT   = "#e0e0e0"

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e2e3e")

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    ax1.plot(equity_curve.index, equity_curve.values, color=ACCENT, lw=1.5, label="Strategy")
    if benchmark_curve is not None:
        bc = benchmark_curve.reindex(equity_curve.index, method="ffill")
        bc = bc / bc.iloc[0] * INITIAL_CAPITAL
        ax1.plot(bc.index, bc.values, color=BENCH, lw=1.2, ls="--", label="Benchmark", alpha=0.8)
    ax1.set_title("Portfolio Equity Curve", fontsize=13, fontweight="bold")
    ax1.set_ylabel(f"Portfolio Value (₹, base {INITIAL_CAPITAL:,})")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.legend(facecolor=BG, edgecolor=GREY, labelcolor=TEXT)
    ax1.grid(axis="y", color="#2e2e3e", lw=0.5)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    roll_max = equity_curve.cummax()
    dd       = (equity_curve - roll_max) / roll_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.6)
    ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax2.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(axis="y", color="#2e2e3e", lw=0.5)

    # 3. Annual Returns Bar
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    years  = ann_ret.index.astype(str)
    rets   = ann_ret.values * 100
    colors = [GREEN if r >= 0 else RED for r in rets]
    bars   = ax3.bar(years, rets, color=colors, width=0.7, edgecolor="none")
    ax3.axhline(0, color=GREY, lw=0.8)
    ax3.set_title("Annual Returns (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Return %")
    ax3.tick_params(axis="x", rotation=45)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for bar, val in zip(bars, rets):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if val >= 0 else -1.5),
                 f"{val:.1f}%", ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=7, color=TEXT)
    ax3.grid(axis="y", color="#2e2e3e", lw=0.5)

    # 4. Daily Return Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4)
    dr_pct = daily_returns * 100
    ax4.hist(dr_pct[dr_pct != 0], bins=80, color=ACCENT, alpha=0.7, edgecolor="none")
    ax4.axvline(0, color=GREY, lw=1, ls="--")
    ax4.axvline(dr_pct.mean(), color=GREEN, lw=1.2, ls="--", label=f"Mean {dr_pct.mean():.3f}%")
    ax4.set_title("Daily Return Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Daily Return %")
    ax4.set_ylabel("Frequency")
    ax4.legend(facecolor=BG, edgecolor=GREY, labelcolor=TEXT, fontsize=8)
    ax4.grid(axis="y", color="#2e2e3e", lw=0.5)

    # 5. Rolling 1-Year Sharpe
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5)
    rf_d = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_YEAR) - 1
    roll_sharpe = (daily_returns - rf_d).rolling(TRADING_DAYS_YEAR).apply(
        lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS_YEAR) if x.std() > 0 else 0,
        raw=True)
    ax5.plot(roll_sharpe.index, roll_sharpe.values, color=ACCENT, lw=1.2)
    ax5.axhline(0, color=GREY, lw=0.8, ls="--")
    ax5.axhline(1, color=GREEN, lw=0.8, ls="--", alpha=0.6, label="Sharpe=1")
    ax5.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=roll_sharpe.values >= 0, color=GREEN, alpha=0.15)
    ax5.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=roll_sharpe.values < 0,  color=RED,   alpha=0.15)
    ax5.set_title("Rolling 1-Year Sharpe Ratio", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Sharpe")
    ax5.legend(facecolor=BG, edgecolor=GREY, labelcolor=TEXT, fontsize=8)
    ax5.grid(axis="y", color="#2e2e3e", lw=0.5)

    fig.suptitle("RSI(2) Mean Reversion Scalping Strategy — Backtest Results",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved        → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*_daily.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_daily.csv files found in '{DATA_FOLDER}/'")

    print(f"\nFound {len(csv_files)} symbol files. Processing …\n")

    all_trades_list  = []
    trading_date_set = set()

    for path in csv_files:
        symbol = os.path.basename(path).replace("_daily.csv", "")
        df     = load_and_prepare(path)
        if df is None or df.empty:
            continue
        trading_date_set.update(df["Date"].tolist())
        trades = generate_signals(df)
        if not trades.empty:
            trades["symbol"] = symbol
            all_trades_list.append(trades)

    if not all_trades_list:
        print("No trades generated. Check your data or thresholds.")
        return

    all_trades = pd.concat(all_trades_list, ignore_index=True)
    all_trades["exit_date"]  = pd.to_datetime(all_trades["exit_date"])
    all_trades["entry_date"] = pd.to_datetime(all_trades["entry_date"])

    # Sort by exit date → entry date → symbol
    all_trades = all_trades.sort_values(
        ["exit_date", "entry_date", "symbol"]
    ).reset_index(drop=True)

    trading_dates = pd.DatetimeIndex(sorted(trading_date_set))

    print(f"Total trades across all symbols : {len(all_trades):,}")
    print(f"Unique symbols with trades      : {all_trades['symbol'].nunique():,}")

    # ── Portfolio returns & equity curve ──────────────────────────────────────
    daily_returns = build_portfolio_returns(all_trades, trading_dates)
    equity_curve  = (1 + daily_returns).cumprod() * INITIAL_CAPITAL
    equity_curve.index  = pd.DatetimeIndex(equity_curve.index)
    daily_returns.index = pd.DatetimeIndex(daily_returns.index)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark_curve = None
    if BENCHMARK_CSV and os.path.exists(BENCHMARK_CSV):
        bdf = load_and_prepare(BENCHMARK_CSV)
        if bdf is not None:
            benchmark_curve = bdf.set_index("Date")["Close"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    ann_ret = annual_returns(equity_curve)
    metrics = compute_metrics(daily_returns, equity_curve, all_trades)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = "─" * 54
    W   = 12

    print(f"\n{'═'*54}")
    print(f"  RSI(2) MEAN REVERSION SCALPING — BACKTEST SUMMARY")
    print(f"{'═'*54}")
    print(f"  Universe      : {UNIVERSE_LABEL}")
    print(f"  Period        : {metrics['start_date']}  →  {metrics['end_date']}")
    print(f"  Symbols loaded: {len(csv_files)}")
    rt_cost = 2 * (TRANSACTION_COST_PER_SIDE + SLIPPAGE_PER_SIDE) * 100
    print(f"  Round-trip cost: {rt_cost:.3f}%  "
          f"(tcost {TRANSACTION_COST_PER_SIDE*100:.3f}% + slip {SLIPPAGE_PER_SIDE*100:.3f}% per side)")
    print(f"  Note: all metrics use NET returns (after costs). trade_log.csv has both gross and net.")
    print(sep)

    def row(label, val):
        print(f"  {label:<24} {val:>{W}}")

    print("  RETURN METRICS")
    row("Total Return",      f"{metrics['total_return']:.2%}")
    row("CAGR",              f"{metrics['cagr']:.2%}")
    row("Volatility (ann)",  f"{metrics['volatility']:.2%}")
    row("Best Day",          f"{metrics['best_day']:.2%}")
    row("Worst Day",         f"{metrics['worst_day']:.2%}")
    print(sep)
    print("  RISK METRICS")
    row("Max Drawdown",      f"{metrics['max_drawdown']:.2%}")
    row("Sharpe Ratio",      f"{metrics['sharpe']:.3f}")
    row("Sortino Ratio",     f"{metrics['sortino']:.3f}")
    row("Calmar Ratio",      f"{metrics['calmar']:.3f}")
    print(sep)
    print("  TRADE STATISTICS")
    row("Total Trades",      f"{metrics['trades']:,}")
    row("Win Rate",          f"{metrics['win_rate']:.2%}")
    row("Avg Win",           f"{metrics['avg_win']:.2%}")
    row("Avg Loss",          f"{metrics['avg_loss']:.2%}")
    row("Profit Factor",     f"{metrics['profit_factor']:.3f}")
    row("Avg Hold (days)",   f"{metrics['avg_hold_days']:.1f}")
    row("Stopped Out",       f"{metrics['stopped_out']:,}  ({metrics['stop_rate']:.1%})")
    row("Best Trade",        f"{metrics['best_trade']:.2%}")
    row("Worst Trade",       f"{metrics['worst_trade']:.2%}")
    row("Median Trade",      f"{metrics['median_trade']:.2%}")
    row("Coverage",          f"{metrics['coverage']:.1f}%")
    print(sep)
    print("  ANNUAL RETURNS")
    print(f"  {'Year':<8} {'Return':>{W}}")
    print(f"  {'─'*22}")
    for year, val in ann_ret.items():
        mark = "✓" if val >= 0 else "✗"
        print(f"  {year:<8} {val:>{W}.2%}  {mark}")
    print(f"{'═'*54}\n")

    # ── Save trade log (overwrite, sorted by exit date) ───────────────────────
    trade_path = os.path.join(OUTPUT_FOLDER, TRADE_LOG_FILE)
    all_trades.to_csv(trade_path, index=False)
    print(f"  Trade log saved    → {trade_path}  ({len(all_trades):,} rows, sorted by exit_date)")

    # ── Append one row to metrics.csv ─────────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_FOLDER, METRICS_FILE)
    append_metrics_row(metrics, ann_ret, metrics_path)
    print(f"  Metrics appended   → {metrics_path}")

    # ── Charts (overwrite) ────────────────────────────────────────────────────
    chart_path = os.path.join(OUTPUT_FOLDER, EQUITY_CURVE_FILE)
    plot_results(equity_curve, daily_returns, ann_ret, benchmark_curve, chart_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()