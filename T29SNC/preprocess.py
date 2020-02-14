import os
import numpy as np
import pandas as pd
import datetime as dt

## configs
DATA_DIR = 'T29SNC/sampled_data_features'
MONTH = 2
END_YEAR = 2019
SAVE_DIR = 'T29SNC/'

# get date range
start_date = dt.datetime(month=MONTH+1, year=END_YEAR-1, day=1)\
    if MONTH!=12 else dt.datetime(month=1, year=END_YEAR, day=1)

date_range = pd.date_range(start_date, periods=12, freq='M')
date_range = date_range[date_range.month.argsort()]
date_range = [
    f'{yr}_{mth}' if mth>=10 else f'{yr}_0{mth}'
    for yr, mth in zip(date_range.year, date_range.month)
]

files = np.array(os.listdir(DATA_DIR))
get_filename = lambda yrmth: files[[x.startswith(yrmth) for x in files]][0]


# read csv files and concatenate them into one dataframe
df = pd.DataFrame()
for year_month in date_range:
    print(year_month)
    try:
        filepath = os.path.join(DATA_DIR, get_filename(year_month))
    except IndexError:
        raise IOError(f'Data for period {year_month} not found.')
    _df = pd.read_csv(filepath)
    cols = _df.columns.difference(df.columns)
    df = df.join(_df[cols], how='outer')

del _df, cols, filepath, year_month, start_date


# drop band values for pixels containing clouds
prefixes = [c.replace('_SCL', '') for c in  df.columns[df.columns.str.endswith('_SCL')]]
for prefix in prefixes:
    df.loc[df[f'{prefix}_SCL'].isin([3,8,9,10]), df.columns.str.startswith(prefix)] = np.nan

# interpolate missing values
bands = set([
    c.split('_')[-1]
    for c in df.columns
    if c.split('_')[-1].startswith('B') or c.split('_')[-1].startswith('ND')
])

df = df.sort_index(1)
for band in bands:
    cols_to_interpolate = df.columns.str.endswith(band)
    df.loc[:,cols_to_interpolate] = df.loc[:,cols_to_interpolate]\
        .interpolate(method='linear', axis=1, limit=2, limit_direction='both', limit_area=None)

# drop cloud mask columns
df = df[df.columns[~df.columns.str.endswith('SCL')]]

if MONTH >= 10:
    df.to_csv(SAVE_DIR+f'{END_YEAR}_{MONTH}', index=False)
else:
    df.to_csv(SAVE_DIR+f'{END_YEAR}_0{MONTH}', index=False)
