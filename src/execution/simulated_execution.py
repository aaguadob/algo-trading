from src.core.base import ExecutionHandler
from src.core.events import FillEvent


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates order execution with slippage and commission.
    Used for both backtest and paper trading.
    """

    def __init__(self, commission=0.0005, slippage=0.0005):
        self.event_queue = None  # injected by engine
        self.commission = commission
        self.slippage = slippage

    def execute_order(self, order_event, market_price):

        if order_event.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        direction = 1 if order_event.side == "BUY" else -1

        # Apply slippage
        fill_price = market_price * (1 + direction * self.slippage)

        # Commission in currency terms
        commission_cost = abs(order_event.quantity * fill_price) * self.commission

        fill_event = FillEvent(
            timestamp=order_event.timestamp,
            symbol=order_event.symbol,
            quantity=direction * order_event.quantity,
            fill_price=fill_price,
            commission=commission_cost
        )

        self.event_queue.append(fill_event)