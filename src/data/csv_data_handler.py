import pandas as pd
from src.core.events import MarketEvent
from .base import DataHandler

class CSVDataHandler(DataHandler):
    """
    Streams CSV data row-by-row for backtest or paper trading.
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.symbol = symbol
        self.df = self._prepare_dataframe(df)
        self.index = 0
        self.max_index = len(self.df)

    def _prepare_dataframe(self, df):
        # Validate required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Ensure no duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]

        return df

    def get_next_bar(self):
        if self.index >= self.max_index:
            return None

        row = self.df.iloc[self.index]
        timestamp = row.name

        print(f"Data: {row["Close"]}")

        event = MarketEvent(
            timestamp=timestamp,
            symbol=self.symbol,
            price=row["Close"]
        )

        self.index += 1
        return event

    def rewind(self):
        self.index = 0