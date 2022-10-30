import os
import datetime
import pandas as pd
from common.modules.logger import logger
from common.paths import Paths
from common.refdata import date_formats


class BitfinexReader:
    dir_tick = Paths.bitfinex_tick
    schema_trade = ['timestamp', 'price', 'size', 'side']
    schema_quote = ['timestamp', 'price', 'size', 'count']

    @classmethod
    def load(cls, start: datetime.date, end: datetime.date, directory: str, fn_key: str):
        df_lst = []
        for root, dirs, filenames in os.walk(directory):
            for fn in filenames:
                if fn_key not in fn:
                    continue
                fn_date = datetime.datetime.strptime(fn[:8], date_formats.Ymd)
                if start <= fn_date < end + datetime.timedelta(days=1):
                    try:
                        df = pd.read_csv(os.path.join(directory, fn), compression='zip', header=None)
                        # bad value
                        def is_float(val):
                            try:
                                float(val)
                                return True
                            except:
                                return False
                        ix_drop = df.index[(~df[0].apply(is_float))]
                        if len(ix_drop) > 0:
                            logger.info(f'Dropping {len(ix_drop)}. Time not float. File capture was interrupted on server likely.')
                            df = df.drop(ix_drop).reset_index(drop=True)

                        df[0] = pd.to_datetime(df[0].astype(float) * 1000 ** 3, origin=fn_date).dt.round('ms')  # float issue giving imprecise ns when converting to dt
                    except pd.errors.EmptyDataError:
                        logger.info(f'No Data for {fn_date}')
                        continue
                    df_lst.append(df)
            break
        # logger.info(f'Concatenating {len(df_lst)} dataframes ...')
        return pd.concat(df_lst).reset_index(drop=True) if df_lst else None

    @staticmethod
    def remove_merged_rows(df):
        b4 = len(df)
        df = df[~(df[0].astype(str).apply(len) > 9)]
        if len(df) < b4:
            print(f'Removed {b4 - len(df)} rows')
        return df

    @classmethod
    def load_trades(cls, sym: str, start: datetime.datetime, end: datetime.datetime):
        df = cls.load(start, end, os.path.join(cls.dir_tick, sym.lower()), fn_key='trade')
        df.columns = cls.schema_trade
        df['side'] = df['side'].map({'Sell': -1, 'Buy': 1})
        df['size'] = df['size'].abs()
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        return df

    @classmethod
    def load_quotes(cls, sym: str, start: datetime.datetime, end: datetime.datetime):
        df = cls.load(start, end, os.path.join(cls.dir_tick, sym.lower()), fn_key='quote')
        if df is None or df.empty:
            return None
        df.columns = cls.schema_quote
        # amount > 0: Bid < 0 Ask. count 0: deleted
        df['side'] = (df['size'] > 0).map({True: 1, False: -1})
        # df['side'].loc[df.index[df['count'] == 0]] = 0
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        return df


if __name__ == '__main__':
    print(BitfinexReader.load_quotes(sym='ethusd', start=datetime.datetime(2022, 2, 8), end=datetime.datetime(2022, 2, 8)))
