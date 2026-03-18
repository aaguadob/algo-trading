from collections import deque
from src.core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent

class EventDrivenEngine:
    """
    Unified event-driven trading engine.
    Works for backtest, paper, and live.
    """

    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution_handler
        self.event_queue = deque()
        self._inject_event_queue()

    def _inject_event_queue(self):
        """
        Ensures all components share the same event queue.
        """
        self.strategy.event_queue = self.event_queue
        self.portfolio.event_queue = self.event_queue
        self.execution.event_queue = self.event_queue

    def run(self):
        """
        Main event loop.
        """
        while True:
            market_event = self.data_handler.get_next_bar()

            if market_event is None:
                break

            self.event_queue.append(market_event)

            while self.event_queue:
                event = self.event_queue.popleft()
                self._process_event(event)

    def _process_event(self, event):

        if isinstance(event, MarketEvent):
            self.strategy.on_market_event(event)
            self.portfolio.update_market_value(
                event.timestamp,
                event.price
            )

        elif isinstance(event, SignalEvent):
            self.portfolio.on_signal_event(event)

        elif isinstance(event, OrderEvent):
            self.execution.execute_order(event, event.price)

        elif isinstance(event, FillEvent):
            self.portfolio.on_fill_event(event)