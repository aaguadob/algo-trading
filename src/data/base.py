from abc import ABC, abstractmethod

class DataHandler(ABC):
    @abstractmethod
    def get_next_bar(self):
        """Return the next MarketEvent"""
        pass

    @abstractmethod
    def rewind(self):
        """Reset to beginning of dataset"""
        pass