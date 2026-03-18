from dataclasses import dataclass
from datetime import datetime

class Event:
    pass

@dataclass
class MarketEvent(Event):
    timestamp: datetime
    symbol: str
    price: float

@dataclass
class SignalEvent(Event):
    timestamp: datetime
    symbol: str
    direction: int  # -1 short, 0 flat, 1 long

@dataclass
class OrderEvent(Event):
    timestamp: datetime
    symbol: str
    quantity: float
    side: str  # "BUY" or "SELL"

# @dataclass
# class FillEvent(Event):
#     """ Represents reality, the order that got executed """
#     timestamp: datetime
#     symbol: str
#     quantity: float
#     fill_price: float

class FillEvent:
    def __init__(self, timestamp, symbol, quantity, fill_price, commission=0.0):
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission