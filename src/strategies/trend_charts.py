"""
Trend Trading Analysis Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plots per ticker:  Candlestick chart · MA20 · MA50 · MA200
                   Stochastics (%K / %D)
                   RSI (14)
                   MACD (12/26/9)
                   Last candle highlighted as BULLISH / BEARISH

Requirements:  pip install yfinance matplotlib pandas numpy

Usage:
  python trend_analysis.py                    # uses default TICKERS list
  python trend_analysis.py AAPL TSLA BTC-USD  # pass tickers as CLI args
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D as MLine
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
PERIOD_DAYS     = 365          # calendar days of history to fetch
OUTPUT_DIR      = "."          # folder where PNGs are saved

# ── Color palette ──────────────────────────────────────────────────────────────
BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
BG_CARD  = "#1c2128"
BORDER   = "#30363d"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
CYAN     = "#76e3ea"
MA_COLORS = {"MA20": "#58a6ff", "MA50": "#e3b341", "MA200": "#bc8cff"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker: str, days: int = PERIOD_DAYS) -> pd.DataFrame:
    """Fetch via yfinance; fall back to synthetic GBM data if unavailable."""
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days + 60)
        df    = yf.download(ticker, start=start, end=end,
                            progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("empty response")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df.tail(days)
    except Exception as e:
        print(f"[yfinance unavailable: {e}] -> using synthetic data")
        return _synthetic(ticker, days)


def _synthetic(ticker: str, days: int) -> pd.DataFrame:
    """GBM price simulation as a demo fallback."""
    np.random.seed(abs(hash(ticker)) % (2**31))
    seeds = {
        "AAPL": (182.0, 0.0003, 0.015), "MSFT": (375.0, 0.0004, 0.014),
        "NVDA": (480.0, 0.0007, 0.028), "TSLA": (200.0, 0.0002, 0.032),
        "SPY":  (490.0, 0.0003, 0.009),
    }
    s0, mu, sigma = seeds.get(ticker, (100.0, 0.0003, 0.018))
    end   = datetime.today()
    dates = pd.bdate_range(end=end, periods=days)
    r     = np.exp((mu - 0.5*sigma**2) + sigma * np.random.randn(days))
    close = s0 * np.cumprod(r)
    noise = lambda s: s * (1 + np.random.uniform(-0.005, 0.005, days))
    high  = np.maximum(close, noise(close)) * (1 + np.abs(np.random.randn(days))*0.006)
    low   = np.minimum(close, noise(close)) * (1 - np.abs(np.random.randn(days))*0.006)
    opn   = noise(close)
    vol   = np.random.lognormal(17, 0.6, days).astype(int)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=dates
    )


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ma(s, n):
    return s.rolling(n).mean()

def _rsi(s, n=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd(s, fast=12, slow=26, sig=9):
    line   = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal

def _stoch(df, k=14, smooth=3, d=3):
    lo    = df["low"].rolling(k).min()
    hi    = df["high"].rolling(k).max()
    raw   = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    pct_k = raw.rolling(smooth).mean()
    return pct_k, pct_k.rolling(d).mean()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style(ax, ylabel=""):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8, labelpad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linewidth=0.45, linestyle="--", alpha=0.55)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def draw_candlesticks(ax, df: pd.DataFrame):
    """
    Draw OHLC candlesticks using integer x-positions for crisp pixel rendering.
    The last candle is outlined with a bright glow to make it stand out.
    Returns the integer index of the last candle (n-1).
    """
    n      = len(df)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    body_w = max(0.25, min(0.65, 100 / n))  # auto-scale width

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        bullish = c >= o
        is_last = (i == n - 1)
        color   = GREEN if bullish else RED

        body_h = max(abs(c - o), (h - l) * 0.012)   # min-height so doji shows
        body_y = min(o, c)

        if is_last:
            # Glow halos behind the last candle
            for gw, ga in [(body_w + 0.30, 0.12), (body_w + 0.15, 0.20)]:
                ax.add_patch(Rectangle(
                    (i - gw/2, body_y), gw, body_h,
                    linewidth=0, facecolor=color, alpha=ga, zorder=3
                ))
            wick_lw = 2.0
            edge_lw = 1.5
            edge_c  = "#ffffff" if bullish else "#ff8080"
            zord    = 6
        else:
            wick_lw = 0.75
            edge_lw = 0.5
            edge_c  = color
            zord    = 3

        # High-low wick
        ax.add_line(MLine([i, i], [l, h],
                          color=color, linewidth=wick_lw, alpha=0.9, zorder=zord))
        # Body
        ax.add_patch(Rectangle(
            (i - body_w/2, body_y), body_w, body_h,
            linewidth=edge_lw, edgecolor=edge_c,
            facecolor=color, alpha=0.9, zorder=zord + 1
        ))

    ax.set_xlim(-1, n)
    return n - 1   # index of last candle


def _candle_xaxis(ax, df, interval_months=2):
    """Replace integer ticks with month labels matching the date index."""
    ticks, labels = [], []
    last_label_month = None
    for i, d in enumerate(df.index):
        if last_label_month is None or (
            d.month != last_label_month and d.month % interval_months == 1
        ):
            ticks.append(i)
            labels.append(d.strftime("%b '%y"))
            last_label_month = d.month
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.5, color=TEXT_SEC, ha="center")


def _indicator_xaxis(ax, df, interval_months=2):
    """Same tick positions but expressed as dates (for sharex'd indicator panels)."""
    _candle_xaxis(ax, df, interval_months)


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL TICKER CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_ticker(ticker: str, df: pd.DataFrame, out_path: str):
    close = df["close"]
    n     = len(df)
    xs    = np.arange(n)   # integer x-axis shared by indicators

    ma20 = _ma(close, 20).values
    ma50 = _ma(close, 50).values
    ma200= _ma(close, 200).values
    r    = _rsi(close)
    ml, ms, mh = _macd(close)
    sk, sd     = _stoch(df)

    last_o = df["open"].iloc[-1]
    last_c = df["close"].iloc[-1]
    last_h = df["high"].iloc[-1]
    last_l = df["low"].iloc[-1]
    bullish_last = last_c >= last_o
    candle_label = "BULLISH" if bullish_last else "BEARISH"
    candle_color = GREEN     if bullish_last else RED

    cur = close.iloc[-1]
    chg = (cur - close.iloc[-2]) / close.iloc[-2] * 100

    pmin = df["low"].min()  * 0.975
    pmax = df["high"].max() * 1.025

    fig = plt.figure(figsize=(18, 14), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(4, 1, figure=fig,
                            height_ratios=[5, 2, 2, 2], hspace=0.07,
                            top=0.91, bottom=0.07, left=0.04, right=0.93)
    ax_p = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])   # NOT sharex; integer axis
    ax_r = fig.add_subplot(gs[2])
    ax_m = fig.add_subplot(gs[3])

    # ── Panel 1 · Candlesticks + MAs ──────────────────────────────────
    _style(ax_p, "Price (USD)")
    draw_candlesticks(ax_p, df)

    # MAs over integer x
    ax_p.plot(xs, ma20,  color=MA_COLORS["MA20"],  lw=1.2, zorder=2, label="MA 20")
    ax_p.plot(xs, ma50,  color=MA_COLORS["MA50"],  lw=1.2, zorder=2, label="MA 50")
    ax_p.plot(xs, ma200, color=MA_COLORS["MA200"], lw=1.4, zorder=2, label="MA 200", ls="--")

    ax_p.set_ylim(pmin, pmax)

    # Last-candle annotation box
    box_x = n - 1
    box_y = pmax - (pmax - pmin) * 0.04
    ax_p.annotate(
        f"  {candle_label}  ",
        xy=(box_x, last_c),
        xytext=(box_x - n * 0.08, box_y),
        fontsize=9, fontweight="bold", fontfamily="monospace",
        color=candle_color,
        arrowprops=dict(arrowstyle="->", color=candle_color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_CARD,
                  edgecolor=candle_color, linewidth=1.2),
        zorder=10
    )
    # OHLC readout for last candle
    ohlc_txt = (f"O {last_o:,.2f}   H {last_h:,.2f}\n"
                f"L {last_l:,.2f}   C {last_c:,.2f}")
    ax_p.text(0.01, 0.03, ohlc_txt, transform=ax_p.transAxes,
              color=TEXT_SEC, fontsize=7.5, fontfamily="monospace",
              va="bottom")

    legend_handles = [
        plt.Line2D([0],[0], color=MA_COLORS["MA20"],  lw=1.5, label="MA 20"),
        plt.Line2D([0],[0], color=MA_COLORS["MA50"],  lw=1.5, label="MA 50"),
        plt.Line2D([0],[0], color=MA_COLORS["MA200"], lw=1.5, label="MA 200", ls="--"),
        plt.Rectangle((0,0),1,1, color=GREEN, alpha=0.85, label="Bullish"),
        plt.Rectangle((0,0),1,1, color=RED,   alpha=0.85, label="Bearish"),
    ]
    ax_p.legend(handles=legend_handles, loc="upper left", fontsize=8,
                facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_PRI,
                framealpha=0.92, ncol=5)

    _candle_xaxis(ax_p, df)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # ── Panel 2 · Stochastics ──────────────────────────────────────────
    _style(ax_s, "Stoch %K/%D")
    ax_s.plot(xs, sk.values, color=CYAN,   lw=1.2, label="%K")
    ax_s.plot(xs, sd.values, color=ORANGE, lw=1.2, label="%D", ls="--")
    for lvl, col in [(80, RED), (20, GREEN)]:
        ax_s.axhline(lvl, color=col, lw=0.75, ls=":", alpha=0.8)
    ax_s.fill_between(xs, 80, 100, color=RED,   alpha=0.06)
    ax_s.fill_between(xs, 0,   20, color=GREEN, alpha=0.06)
    ax_s.set_xlim(-1, n)
    ax_s.set_ylim(0, 100); ax_s.set_yticks([20, 50, 80])
    ax_s.legend(loc="upper left", fontsize=8, facecolor=BG_CARD,
                edgecolor=BORDER, labelcolor=TEXT_PRI, framealpha=0.92, ncol=2)
    _candle_xaxis(ax_s, df)
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # ── Panel 3 · RSI ─────────────────────────────────────────────────
    _style(ax_r, "RSI (14)")
    ax_r.plot(xs, r.values, color=PURPLE, lw=1.3)
    for lvl, col in [(70, RED), (30, GREEN)]:
        ax_r.axhline(lvl, color=col, lw=0.75, ls=":", alpha=0.8)
    ax_r.axhline(50, color=BORDER, lw=0.6, alpha=0.5)
    ax_r.fill_between(xs, 70, 100, color=RED,   alpha=0.06)
    ax_r.fill_between(xs, 0,   30, color=GREEN, alpha=0.06)
    ax_r.fill_between(xs, 50, r.values, where=r.values >= 50,
                      color=GREEN, alpha=0.10, interpolate=True)
    ax_r.fill_between(xs, 50, r.values, where=r.values <  50,
                      color=RED,   alpha=0.10, interpolate=True)
    ax_r.set_xlim(-1, n)
    ax_r.set_ylim(0, 100); ax_r.set_yticks([30, 50, 70])
    cur_rsi   = r.dropna().iloc[-1]
    rsi_color = RED if cur_rsi > 70 else (GREEN if cur_rsi < 30 else PURPLE)
    ax_r.text(0.01, 0.86, f"RSI  {cur_rsi:.1f}", transform=ax_r.transAxes,
              color=rsi_color, fontsize=9, fontweight="bold", fontfamily="monospace")
    _candle_xaxis(ax_r, df)
    plt.setp(ax_r.get_xticklabels(), visible=False)

    # ── Panel 4 · MACD ────────────────────────────────────────────────
    _style(ax_m, "MACD (12/26/9)")
    ax_m.plot(xs, ml.values, color=ACCENT,  lw=1.3, label="MACD",     zorder=3)
    ax_m.plot(xs, ms.values, color=ORANGE,  lw=1.2, label="Signal",   zorder=3, ls="--")
    bar_colors = [GREEN if v >= 0 else RED for v in mh.values]
    ax_m.bar(xs, mh.values, color=bar_colors, alpha=0.5, width=0.8,
             label="Histogram", zorder=2)
    ax_m.axhline(0, color=BORDER, lw=0.8)
    ax_m.set_xlim(-1, n)
    ax_m.legend(loc="upper left", fontsize=8, facecolor=BG_CARD,
                edgecolor=BORDER, labelcolor=TEXT_PRI, framealpha=0.92, ncol=3)
    _candle_xaxis(ax_m, df)

    # ── Title bar ──────────────────────────────────────────────────────
    chg_col = GREEN if chg >= 0 else RED
    sym_chg = "▲" if chg >= 0 else "▼"
    fig.text(0.04, 0.955, ticker, fontsize=28, fontweight="bold",
             color=TEXT_PRI, fontfamily="monospace")
    fig.text(0.04, 0.930, f"${cur:,.2f}", fontsize=13,
             color=TEXT_PRI, fontfamily="monospace")
    fig.text(0.175, 0.930, f"{sym_chg} {abs(chg):.2f}%", fontsize=13,
             color=chg_col, fontfamily="monospace", fontweight="bold")
    fig.text(0.93, 0.955, df.index[-1].strftime("%d %b %Y"),
             fontsize=9, color=TEXT_SEC, ha="right", fontfamily="monospace")
    fig.text(0.93, 0.935, "Trend Analysis Dashboard  ·  Candlestick",
             fontsize=8, color=TEXT_SEC, ha="right")
    fig.add_artist(MLine([0.04, 0.93], [0.924, 0.924],
                         transform=fig.transFigure, color=BORDER, lw=0.8))

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {ticker} ({candle_label}) -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW CHART (all tickers, candlesticks + MAs)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(datasets: dict, out_path: str):
    n     = len(datasets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.5 * nrows),
                             facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (ticker, df) in enumerate(datasets.items()):
        ax    = flat[i]
        close = df["close"]
        nn    = len(df)
        xs    = np.arange(nn)
        cur   = close.iloc[-1]
        chg   = (cur - close.iloc[0]) / close.iloc[0] * 100
        last_bullish = df["close"].iloc[-1] >= df["open"].iloc[-1]
        last_col = GREEN if last_bullish else RED

        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(color=BORDER, lw=0.4, ls="--", alpha=0.45)

        draw_candlesticks(ax, df)
        ax.plot(xs, _ma(close, 20).values,  color=MA_COLORS["MA20"],  lw=0.9, alpha=0.85)
        ax.plot(xs, _ma(close, 50).values,  color=MA_COLORS["MA50"],  lw=0.9, alpha=0.85)
        ax.plot(xs, _ma(close, 200).values, color=MA_COLORS["MA200"], lw=1.0, alpha=0.85, ls="--")

        ax.tick_params(colors=TEXT_SEC, labelsize=7)
        _candle_xaxis(ax, df, interval_months=3)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelsize=7, colors=TEXT_SEC)
        ax.set_ylim(df["low"].min() * 0.975, df["high"].max() * 1.025)

        sym        = "▲" if chg >= 0 else "▼"
        trend_word = "BULLISH" if last_bullish else "BEARISH"
        ax.set_title(
            f"{ticker}   ${cur:,.2f}  {sym}{abs(chg):.1f}%  │  Last: {trend_word}",
            color=TEXT_PRI, fontsize=9.5, fontweight="bold",
            fontfamily="monospace", pad=7, loc="left"
        )
        # color the BULLISH/BEARISH part differently via a second text
        # (matplotlib title doesn't support multicolor; use text instead)
        ax.text(0.99, 1.02, trend_word, transform=ax.transAxes,
                color=last_col, fontsize=9, fontweight="bold",
                fontfamily="monospace", ha="right", va="bottom")

    for j in range(i + 1, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle("Trend Overview  ·  Candlestick  (MA 20 / 50 / 200)",
                 fontsize=13, color=TEXT_PRI, fontweight="bold",
                 fontfamily="monospace", y=1.012)
    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  Overview -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'━'*52}")
    print("  Trend Trading Analysis Dashboard  (Candlestick)")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  History : {PERIOD_DAYS} days")
    print(f"{'━'*52}\n")

    datasets = {}
    for ticker in tickers:
        print(f"  ↓ {ticker} ...", end=" ", flush=True)
        try:
            df = fetch_data(ticker)
            if len(df) < 60:
                print("not enough data, skipped.")
                continue
            datasets[ticker] = df
            print(f"{len(df)} trading days")
        except Exception as e:
            print(f"FAILED ({e})")

    if not datasets:
        print("\n  No data. Exiting.\n")
        sys.exit(1)

    print()
    output_files = []
    for ticker, df in datasets.items():
        path = os.path.join(OUTPUT_DIR, f"{ticker}_trend.png")
        plot_ticker(ticker, df, path)
        output_files.append(path)

    if len(datasets) > 1:
        path = os.path.join(OUTPUT_DIR, "trend_overview.png")
        plot_overview(datasets, path)
        output_files.append(path)

    print(f"\n{'━'*52}")
    print(f"  {len(output_files)} chart(s) saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'━'*52}\n")


if __name__ == "__main__":
    main()