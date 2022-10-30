import os
import datetime
import pandas as pd
from common.modules.logger import logger
from common.paths import Paths
from common.refdata import date_formats


class BitmexReader:
    dir_quotes = os.path.join(Paths.bitmex_raw, 'quote')
    dir_trades = os.path.join(Paths.bitmex_raw, 'trade')

    @classmethod
    def load(cls, sym: str, start: datetime.datetime, end: datetime.datetime, directory: str):
        df_lst = []
        for root, dirs, filenames in os.walk(directory):
            for file in filenames:
                fn_date = datetime.datetime.strptime(file[:8], date_formats.Ymd)
                if start <= fn_date <= end + datetime.timedelta(days=1):
                    df = pd.read_csv(os.path.join(directory, file), compression='gzip')
                    df = df[df['symbol'] == sym.upper()]
                    df_lst.append(df)
            break
        logger.info(f'Concatenating {len(df_lst)} dataframes ...')
        df = pd.concat(df_lst).reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dD%H:%M:%S.%f', utc=True)
        return df

    @classmethod
    def load_trades(cls, sym: str, start: datetime.datetime, end: datetime.datetime):
        df = cls.load(sym, start, end, cls.dir_trades)
        df['side'] = df['side'].map({'Sell': -1, 'Buy': 1})
        return df

    @classmethod
    def load_quotes(cls, sym: str, start: datetime.datetime, end: datetime.datetime):
        return cls.load(sym, start, end, cls.dir_quotes)
