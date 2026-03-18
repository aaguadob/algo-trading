import yfinance as yf
import pandas as pd
from pathlib import Path

class DataDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download(self, symbol, start, end):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def save(self, df, symbol):
        df.to_csv(self.data_dir / f"{symbol}.csv")

    def load(self, symbol):
        return pd.read_csv(
            self.data_dir / f"{symbol}.csv",
            parse_dates=True,
            index_col=0
        )

    def get(self, symbol, start, end, refresh=False):
        file_path = self.data_dir / f"{symbol}.csv"

        if refresh or not file_path.exists():
            df = self.download(symbol, start, end)
            self.save(df, symbol)
        else:
            df = self.load(symbol)

        return df.loc[start:end]