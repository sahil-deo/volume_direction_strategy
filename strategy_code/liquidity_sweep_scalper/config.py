"""
Configuration for Liquidity Sweep Reversal Scalper (LSRS) Strategy.
All tuneable parameters in one place.
"""

# ─── Data Paths ───────────────────────────────────────────────────────
DATA_DIR = "/Users/sahil/Developer/Quantitative Finance/data/YF Stock Data Nifty 750"
RESULTS_DIR = "/Users/sahil/Developer/Quantitative Finance/aakash/results/liquidity_sweep_scalper"

# ─── Date Ranges ──────────────────────────────────────────────────────
TRAIN_END_DATE = "2024-12-31"   # backtest on data up to this date
# Data after this date is reserved for blind/forward testing

# ─── Liquidity Sweep Parameters ───────────────────────────────────────
SWEEP_LOOKBACK = 20             # bars to look back for swing high/low

# ─── VWAP Parameters ─────────────────────────────────────────────────
VWAP_PERIOD = 20                # rolling VWAP window
VWAP_PROXIMITY_PCT = 0.015      # 1.5% — max distance from VWAP for confluence

# ─── Volume Profile Parameters ───────────────────────────────────────
VP_LOOKBACK = 20                # bars for volume profile computation
VP_NUM_BINS = 20                # price bins for volume distribution
POC_PROXIMITY_PCT = 0.02        # 2% — max distance from POC for confluence

# ─── Fair Value Gap Parameters ───────────────────────────────────────
FVG_LOOKBACK_BARS = 3           # how many bars back to search for FVG

# ─── Volume Spike Parameters ─────────────────────────────────────────
VOL_SPIKE_MULTIPLIER = 1.5      # volume must exceed this × 20-bar avg
VOL_AVG_PERIOD = 20             # period for average volume

# ─── Signal Quality Thresholds ───────────────────────────────────────
MIN_CONFLUENCE_SCORE = 3        # minimum confluences to enter (out of 5)
HALF_SIZE_THRESHOLD = 3         # score == this → half position
FULL_SIZE_THRESHOLD = 4         # score >= this → full position

# ─── Risk Management ─────────────────────────────────────────────────
STOP_LOSS_PCT = 0.015           # 1.5% stop loss
TAKE_PROFIT_1_PCT = 0.01        # 1.0% — first TP (exit 50% of position)
TAKE_PROFIT_2_PCT = 0.02        # 2.0% — second TP (exit remaining 50%)
RISK_PER_TRADE_PCT = 0.02       # 2% of capital risked per trade
MAX_SIMULTANEOUS_POSITIONS = 3  # max concurrent open trades

# ─── Transaction Costs ───────────────────────────────────────────────
TCOST_PER_SIDE = 0.001          # 0.1% per side (brokerage + STT + charges)
SLIPPAGE_PER_SIDE = 0.0005      # 0.05% slippage per side
TOTAL_ROUND_TRIP_COST = 2 * (TCOST_PER_SIDE + SLIPPAGE_PER_SIDE)  # 0.3%

# ─── Portfolio ────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000     # ₹10 Lakh starting capital

# ─── Filters ─────────────────────────────────────────────────────────
MIN_BARS_REQUIRED = 100         # skip stocks with fewer bars than this
MIN_AVG_VOLUME = 100_000        # skip illiquid stocks (avg daily vol < this)
