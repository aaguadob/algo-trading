import numpy as np

def compute_performance(equity_curve):
    returns = equity_curve["equity"].pct_change().dropna()

    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cumulative_return = equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1

    rolling_max = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / rolling_max - 1
    max_dd = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }