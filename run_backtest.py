import pandas as pd
import matplotlib.pyplot as plt

from src.core.engine import EventDrivenEngine
from src.data.downloader import DataDownloader
from src.data.csv_data_handler import CSVDataHandler
from src.strategies.moving_average import MovingAverageStrategy
from src.portfolio.portfolio import Portfolio
from src.execution.simulated_execution import SimulatedExecutionHandler

# ------------------------------
# 1. Download / Load Data
# ------------------------------
symbol = "SPY"
start_date = "2022-01-01"
end_date = "2022-12-31"

downloader = DataDownloader()
df = downloader.get(symbol, start=start_date, end=end_date, refresh=True)

data_handler = CSVDataHandler(df, symbol)

# ------------------------------
# 2. Initialize Components
# ------------------------------
short_window = 20
long_window = 50

strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
portfolio = Portfolio(initial_capital=100000)
execution = SimulatedExecutionHandler(commission=0.001, slippage=0.001)

# Inject same event queue for engine
event_queue = []
strategy.event_queue = portfolio.event_queue = execution.event_queue = event_queue

# ------------------------------
# 3. Initialize Engine
# ------------------------------
engine = EventDrivenEngine(
    data_handler=data_handler,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution
)

# ------------------------------
# 4. Run Backtest
# ------------------------------
print("Running backtest...")
engine.run()
print("Backtest completed.")

# ------------------------------
# 5. Equity Curve
# ------------------------------
equity_curve = portfolio.get_equity_curve()
print(equity_curve.tail())

plt.figure(figsize=(12, 6))
plt.plot(equity_curve.index, equity_curve['equity'], label='Equity Curve')
plt.plot(equity_curve.index, [portfolio.initial_capital]*len(equity_curve), '--', label='Initial Capital')
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.title(f"{symbol} Moving Average Backtest ({short_window}/{long_window})")
plt.legend()
plt.grid(True)
plt.show()