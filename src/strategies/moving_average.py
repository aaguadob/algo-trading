import pandas as pd
from collections import deque
from src.core.events import SignalEvent
from src.core.base import Strategy

class MovingAverageStrategy(Strategy):
    """
    Simple moving average crossover strategy.
    Generates signals only on position change.
    """

    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window
        self.event_queue = None  # injected by engine
        self.prices = deque(maxlen=long_window)
        self.current_position = 0
        self.signals = []

    def on_market_event(self, event):
        if self.event_queue is None:
            raise ValueError("event_queue not set")

        self.prices.append(event.price)
        if len(self.prices) < self.long_window:
            return

        prices_series = pd.Series(self.prices)
        print(prices_series[0])
        short_ma = prices_series.rolling(self.short_window).mean().iloc[-1]
        long_ma = prices_series.mean()

        direction = 1 if short_ma > long_ma else -1

        if direction != self.current_position:
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                direction=direction
            )
            self.event_queue.append(signal)
            self.current_position = direction
            self.signals.append((event.timestamp, direction))