import pandas as pd
from src.core.events import OrderEvent, FillEvent


class Portfolio:
    """
    Tracks positions, cash, and equity over time.
    """

    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # number of shares (can be negative)
        self.avg_entry_price = 0.0

        self.current_price = None
        self.equity_curve = []

        self.event_queue = None  # injected by engine

    # --------------------------------------------------
    # SIGNAL → ORDER
    # --------------------------------------------------
    def on_signal_event(self, signal_event):

        target_direction = signal_event.direction  # -1, 0, 1

        if target_direction == 0:
            target_quantity = 0
        else:
            target_quantity = target_direction * 1  # fixed size 1 share

        quantity_diff = target_quantity - self.position

        if quantity_diff == 0:
            return

        side = "BUY" if quantity_diff > 0 else "SELL"

        order = OrderEvent(
            timestamp=signal_event.timestamp,
            symbol=signal_event.symbol,
            quantity=abs(quantity_diff),
            side=side
        )

        self.event_queue.append(order)

    # --------------------------------------------------
    # ORDER FILLED → UPDATE STATE
    # --------------------------------------------------
    def on_fill_event(self, fill_event: FillEvent):

        fill_qty = fill_event.quantity  # signed
        fill_price = fill_event.fill_price
        commission = fill_event.commission

        trade_value = fill_qty * fill_price

        # Update cash
        self.cash -= trade_value
        self.cash -= commission

        # Update position
        new_position = self.position + fill_qty

        # Update average entry price (only if same direction)
        if self.position == 0 or (self.position * fill_qty > 0):
            # Increasing position in same direction
            total_cost = (
                self.avg_entry_price * abs(self.position)
                + fill_price * abs(fill_qty)
            )
            total_qty = abs(self.position) + abs(fill_qty)

            self.avg_entry_price = total_cost / total_qty
        else:
            # Reducing or flipping
            if abs(fill_qty) >= abs(self.position):
                self.avg_entry_price = 0.0

        self.position = new_position

    # --------------------------------------------------
    # MARK-TO-MARKET
    # --------------------------------------------------
    def update_market_value(self, timestamp, price):

        self.current_price = price

        market_value = self.position * price
        total_equity = self.cash + market_value

        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": self.cash,
            "position": self.position
        })

    # --------------------------------------------------
    def get_equity_curve(self):
        return pd.DataFrame(self.equity_curve).set_index("timestamp")