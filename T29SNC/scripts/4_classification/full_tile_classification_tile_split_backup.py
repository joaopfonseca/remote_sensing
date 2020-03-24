# imports
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import datetime as dt

from affine import Affine
from rasterio import features
from src.preprocess.readers import SentinelProductReader

# configs
BANDS_PATH = 'T29SNC/data/organized_images/'
MODELS_PATH = 'T29SNC/models/'
RESULTS_PATH = 'T29SNC/results/'
TILES_SHAPEFILE = 'T29SNC/data/shapefiles/s2tilesPT/sentinel2_tiles.shp'
LABELS_SHAPEFILE15 = 'T29SNC/data/COS2015/cos15_merged_29SNC_UTMZ29N.shp'
LABELS_SHAPEFILE18 = 'T29SNC/data/COS2018/COS2018v1.shp'
FEATURE_RANK_PATH = RESULTS_PATH+'feature_rankings_standardized.csv'
SAVE_PATH = '/run/media/green/Elements/full_tile_saves/'
n_tile_splits = 5
MONTH = 1
END_YEAR = 2020

# load classifier
clf = pickle.load(open(MODELS_PATH+'final_RFC.pkl', 'rb'))

# util functions
def make_data_cubes(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 4, 3, int(X.shape[-1]/(12)))
    )
    return X_reshaped

def make_data_arrays(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 12, int(X.shape[-1]/(12)))
    )
    return X_reshaped


# retrieve tile boundaries and crs
gdf_tiles = gpd.read_file(TILES_SHAPEFILE)
tile_shape = gdf_tiles[gdf_tiles['Name']=='29SNC']
bounds = tile_shape.bounds.iloc[0]
crs = tile_shape.crs
del gdf_tiles #, tile_shape

# get shapefiles
gdf_cos15 = gpd.read_file(LABELS_SHAPEFILE15)
gdf_cos18 = gpd.read_file(LABELS_SHAPEFILE18).to_crs(gdf_cos15.crs)
del gdf_cos15

# rasterize COS
labels = {v:i for i, v in enumerate(gdf_cos18['COS2018_n1'].unique())}
shapes = [
    (geom, value)
    for geom, value
    in zip(gdf_cos18['geometry'], gdf_cos18['COS2018_n1'].map(labels))
]

out_shape = (
    int(np.ceil((bounds.maxy-bounds.miny)/10)),
    int(np.ceil((bounds.maxx-bounds.minx)/10))
)
transf = Affine(10.0, 0.0, bounds.minx,
         0.0, -10.0, bounds.maxy)

raster = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    fill=-1,
    transform=transf,
    all_touched=False,
    default_value=1,
    dtype='int32',
)

# create masks
step = tuple((int(i/n_tile_splits) for i in raster.shape))
masks = []
for i in range(n_tile_splits**2):
    mask = np.zeros(raster.shape)*np.nan
    row = i//n_tile_splits
    column = i%n_tile_splits
    if row!=(n_tile_splits-1) and column!=(n_tile_splits-1):
        mask[row*step[0]:(row+1)*step[0], column*step[1]:(column+1)*step[1]] = 1
    elif row==(n_tile_splits-1) and column!=(n_tile_splits-1):
        mask[row*step[0]:, column*step[1]:(column+1)*step[1]] = 1
    elif row!=(n_tile_splits-1) and column==(n_tile_splits-1):
        mask[row*step[0]:(row+1)*step[0], column*step[1]:] = 1
    elif row==(n_tile_splits-1) and column!=(n_tile_splits-1):
        mask[row*step[0]:, column*step[1]:] = 1
    masks.append(mask)


# get date range
start_date = dt.datetime(month=MONTH+1, year=END_YEAR-1, day=1)\
    if MONTH!=12 else dt.datetime(month=1, year=END_YEAR, day=1)

date_range = pd.date_range(start_date, periods=12, freq='M')
date_range = date_range[date_range.month.argsort()]
date_range = [
    (str(yr), str(mth)) if mth>=10 else (str(yr), '0'+str(mth))
    for yr, mth in zip(date_range.year, date_range.month)
]

feats = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
bands_all_months = feats[feats.apply(lambda x: x.startswith('B'))]\
    .apply(lambda x: x.split('_')[0]).to_list()
bands_all_months += ['SCL']
bands_all_months = list(set(bands_all_months))

for i, mask in enumerate(masks):
    print(f'Tile {i+1}/25')
    # get data
    df = pd.DataFrame()
    for year, month in date_range:
        date = [path for path in os.listdir(BANDS_PATH) if path.startswith(f'{year}_{month}')][0]
        feat_subset = feats[feats.apply(lambda x: len(x.split('_'))<=2)]
        feat_subset = feats[feats.str.startswith(month)]\
            .apply(lambda x: x.split('_')[-1]).to_list()
        if 'NDVI' in feat_subset: feat_subset += ['B08', 'B04']
        if 'NDBI' in feat_subset: feat_subset += ['B11', 'B08']
        if 'NDMI' in feat_subset: feat_subset += ['B08', 'B11']
        if 'NDWI' in feat_subset: feat_subset += ['B08', 'B12']
        feat_subset = list(set(feat_subset + bands_all_months))

        if os.path.isdir(os.path.join(BANDS_PATH, date)):
            print(f'\t({year}_{month}) Reading image')
            img = SentinelProductReader(
                bands_path = BANDS_PATH+f'{date}/',
                labels_shapefile = LABELS_SHAPEFILE18,
                label_col = 'COS2018_n1',
                bands = feat_subset
            )
            img.get_X_array()
            img.get_y_array()

            print(f'\t({year}_{month}) Applying mask')
            img.y_array = img.y_array*mask
            _df = img.to_pandas().dropna(subset=['COS2018_n1'])
            del img
            rename_mapper = {c:f"{month}_{c.split('_')[-2]}"
                for c in _df.drop(columns=['x','y','COS2018_n1']).columns
            }
            _df = _df.rename(columns=rename_mapper)
            index_maker = lambda base, var: (base-var)/(base+var)
            print(f'\t({year}_{month}) Extracting Indices')
            if 'NDVI' in feat_subset:
                _df[f'{month}_NDVI'] = index_maker(base = _df[f'{month}_B08'], var= _df[f'{month}_B04'])
            if 'NDBI' in feat_subset:
                _df[f'{month}_NDBI'] = index_maker(base = _df[f'{month}_B11'], var= _df[f'{month}_B08'])
            if 'NDMI' in feat_subset:
                _df[f'{month}_NDMI'] = index_maker(base = _df[f'{month}_B08'], var= _df[f'{month}_B11'])
            if 'NDWI' in feat_subset:
                _df[f'{month}_NDWI'] = index_maker(base = _df[f'{month}_B08'], var= _df[f'{month}_B12'])

            # join dataframes
            print(f'\t({year}_{month}) Joining Dataframes')
            cols = _df.columns.difference(df.columns)
            df = df.join(_df[cols], how='outer')
            del _df


    # mask clouds
    print(f'Tile {i+1}/25: Cloud Masking')
    prefixes = [c.replace('_SCL', '') for c in  df.columns[df.columns.str.endswith('_SCL')]]
    for prefix in prefixes:
        df.loc[df[f'{prefix}_SCL'].isin([3,8,9,10]), df.columns.str.startswith(prefix)] = np.nan

    df = df.drop(columns=df.columns[df.columns.str.endswith('_SCL')])

    # fill missing values
    print(f'Tile {i+1}/25: Interpolating missing values')
    bands = set([
        c.split('_')[-1]
        for c in df.columns
        if c.split('_')[-1].startswith('B') or c.split('_')[-1].startswith('ND')
    ])

    df = df.sort_index(1)
    for band in bands:
        band_cols = df.columns[df.columns.str.endswith(band)]
        df.loc[:,band_cols] = df.loc[:,band_cols]\
            .interpolate(method='linear', axis=1, limit=2, limit_direction='both', limit_area=None)

    # extract features
    print(f'Tile {i+1}/25: Extracting year-long features')
    to_extract = feats[feats.apply(lambda x: x.startswith('B'))].to_list()
    for quart_range in to_extract:
        if len(quart_range.split('_'))==2:
            band, quart = quart_range.split('_')
            band_cols = df.columns[df.columns.str.endswith(band)]
            if quart == 'var':
                df[f'{band}_var'] = df.loc[:,band_cols].var(1, skipna=True)
            else:
                df[f'{band}_{quart}'] = df.loc[:,band_cols].quantile(int(quart.replace('q',''))/100, axis=1)

        else:
            band, quart1, quart2 = quart_range.split('_')
            band_cols = df.columns[df.columns.str.endswith(band)]
            df[f'{band}_{quart1}_{quart2}'] = (
                df.loc[:,band_cols].quantile(int(quart1.replace('q',''))/100, axis=1) \
                - \
                df.loc[:,band_cols].quantile(int(quart2)/100, axis=1)
            )

    df = df.dropna()
    y = df['COS2018_n1'].values.astype(int)
    X = df.rename(columns=cols_mapper)[feats.to_list()].values

    df_res = df[['x','y']]
    del df
    df_res['y_true'] = y
    df_res['y_pred'] = clf.predict(X)
    df_res.to_csv(SAVE_PATH+f'endmonth_{MONTH}_endyear_{END_YEAR}_part_{i+1}_of_{n_tile_splits**2}.csv')
    del df_res
