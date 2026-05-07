"""
Configuration for FVG Volume Momentum Scalper.
Long-only strategy on Nifty 750 daily data.

Strategy thesis:
  - Bullish FVGs show genuine institutional demand imbalance
  - Volume spike + strong bullish body confirms real momentum, not noise
  - Combining both: enter after a bullish FVG when confirmed by volume momentum
  - Long-only — Indian equities have structural upward bias
"""

# ─── Data Paths ───────────────────────────────────────────────────────
DATA_DIR = "/Users/sahil/Developer/Quantitative Finance/data/YF Stock Data Nifty 750"
RESULTS_DIR = "/Users/sahil/Developer/Quantitative Finance/aakash/results/vwap_volume_scalper"

# ─── Date Ranges ──────────────────────────────────────────────────────
TRAIN_START_DATE = "2010-01-01"    # backtest start date
TRAIN_END_DATE = "2025-12-31"

# ─── FVG Parameters ──────────────────────────────────────────────────
# Bullish FVG: candle[i+1].low > candle[i-1].high (gap up)
FVG_MIN_GAP_PCT = 0.005         # minimum gap size as % of price (0.5%)

# ─── Volume Parameters ───────────────────────────────────────────────
VOL_AVG_PERIOD = 20
VOL_SPIKE_MULTIPLIER = 1.5      # volume > 1.5x average
MIN_BULLISH_BODY_PCT = 0.015    # close-open > 1.5% of open (strong bullish candle)

# ─── Trend Filter ─────────────────────────────────────────────────────
TREND_MA_PERIOD = 50             # only enter when close > this MA (avoid downtrends)

# ─── VWAP Filter (optional confirmation) ─────────────────────────────
VWAP_PERIOD = 20
USE_VWAP_FILTER = False         # if True, only enter when close > VWAP (trend filter)

# ─── Risk Management ─────────────────────────────────────────────────
STOP_LOSS_PCT = 0.01            # 3% stop loss
TAKE_PROFIT_PCT = 0.05          # 5% take profit
MAX_HOLD_DAYS = 10              # exit after 10 bars if no SL/TP
RISK_PER_TRADE_PCT = 0.02       # 2% capital risk per trade
MAX_SIMULTANEOUS_POSITIONS = 5

# ─── Transaction Costs ───────────────────────────────────────────────
TCOST_PER_SIDE = 0.001          # 0.1% per side (Taxes + Brokerage)
SLIPPAGE_PER_SIDE = 0.0005      # 0.05% per side (Execution gap)
TOTAL_ROUND_TRIP_COST = 2 * TCOST_PER_SIDE  # Slippage is applied to prices directly

# ─── Portfolio ────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000

# ─── Filters ─────────────────────────────────────────────────────────
MIN_BARS_REQUIRED = 200
MIN_AVG_VOLUME = 100_000
