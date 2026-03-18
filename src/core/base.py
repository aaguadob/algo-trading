from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def on_market_event(self, event):
        pass

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, order_event, market_price): # at some point move execution price logic ti te¡
        pass