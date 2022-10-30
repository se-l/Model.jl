import pandas as pd
from typing import List
from julia import Main as Jl

Jl.eval(f'''import TsDb: TsDb, Client''')


def upsert(meta: dict, data):
    if isinstance(data, pd.Series):
        Jl.Client.py_upsert(meta, data.index.values, data.values)
    elif isinstance(data, pd.DataFrame):
        for col, data in data.iteritems():
            Jl.Client.py_upsert({**meta, **{'col': col}}, data.index.values, data.values)
    else:
        raise ValueError("Type of df not clear")


def query(meta: dict, start="", stop="9") -> pd.DataFrame:
    cols, mat = Jl.Client.py_query(meta, start=str(start), stop=str(stop))
    df = pd.DataFrame(mat, columns=cols)
    if 'ts' in cols:
        df = df.set_index("ts")
    print('Query Done.')
    return df


def matching_metas(meta: dict) -> List[dict]:
    return Jl.Client.matching_metas(meta)


# if __name__ == '__main__':
#     meta = {
#                "measurement_name": "trade bars",
#                "exchange": "bitfinex",
#                "asset": "ethusd",
#                "information": "volume"
#            }
#     start = "2022-02-07"
#     stop = "2022-02-13"
#     import datetime
#     import pandas as pd
#     upsert({'a': 3}, pd.Series([1], index=[datetime.datetime(2022, 1, 1)]))
#     print(query(
#     ))
    # Jl.Client.drop({
    #         "measurement_name": "trade bars",
    #         "exchange": 'bitfinex',
    #         "asset": 'ethusd',
    #         "information": 'price',
    #         "unit": "ethusd",
    #         "col": "price",
    #         "unit_size": 1,
    #     })
