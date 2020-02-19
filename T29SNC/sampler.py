import os
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features
from affine import Affine

from src.preprocess.readers import SentinelProductReader
#from sklearn.model_selection import train_test_split

BANDS_PATH = 'T29SNC/data/organized_images/'
TILES_SHAPEFILE = 'T29SNC/data/shapefiles/s2tilesPT/sentinel2_tiles.shp'
LABELS_SHAPEFILE15 = 'T29SNC/data/COS2015/cos15_merged_29SNC_UTMZ29N.shp'
LABELS_SHAPEFILE18 = 'T29SNC/data/COS2018/COS2018v1.shp'
CSV_SAVE_PATH = 'T29SNC/data/sampled_data_features/'
random_state = 0

## retrieve tile boundaries and crs
gdf_tiles = gpd.read_file(TILES_SHAPEFILE)
tile_shape = gdf_tiles[gdf_tiles['Name']=='29SNC']
bounds = tile_shape.bounds.iloc[0]
crs = tile_shape.crs
del gdf_tiles #, tile_shape

# get shapefiles
gdf_cos15 = gpd.read_file(LABELS_SHAPEFILE15)
gdf_cos18 = gpd.read_file(LABELS_SHAPEFILE18)

# rasterize COS
labels = {v:i for i, v in enumerate(gdf_cos15['Megaclasse'].unique())}
shapes = [
    (geom, value)
    for geom, value
    in zip(gdf_cos15['geometry'], gdf_cos15['Megaclasse'].map(labels))
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

# apply negative buffer and repeat rasterization process
freq = pd.Series(Counter(raster.flatten())).drop(index=-1)
freq_perc = freq/freq.sum()

gdf_cos15.geometry = gdf_cos15.buffer(-5)
gdf_cos15 = gdf_cos15[~gdf_cos15.is_empty]

shapes = [
    (geom, value)
    for geom, value
    in zip(gdf_cos15['geometry'], gdf_cos15['Megaclasse'].map(labels))
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

# sample points
columns = ['y', 'x']+['class']
shp  = raster.shape
coords = np.indices(shp)
data = np.concatenate([
    coords,
    np.expand_dims(raster, 0)],
    axis=0
).reshape((len(columns), shp[0]*shp[1]))
df_coords = pd.DataFrame(data=data.T, columns=columns)

df_coords = df_coords[df_coords['class']!=-1]
sample_size = (freq_perc*1_000_000).astype(int)
df_coords['class1'] = df_coords['class']
df_coords_sampled = df_coords.groupby('class')\
    .apply(lambda x: x.sample(sample_size[x['class1'].iloc[0]], random_state=random_state))\
    .drop(columns='class1')\
    .reset_index(drop=True)


# create mask
mask = np.zeros(raster.shape)*np.nan
np.apply_along_axis(lambda idx: mask.itemset(tuple(idx),1), 1, df_coords_sampled[['y','x']].values)

for date in os.listdir(BANDS_PATH):
    if os.path.isdir(os.path.join(BANDS_PATH, date)):
        print(date)
        img = SentinelProductReader(
            bands_path = BANDS_PATH+f'{date}/',
            labels_shapefile = LABELS_SHAPEFILE15,
            label_col = 'Megaclasse'
        )
        img.get_X_array()
        img.get_y_array()

        img.y_array = img.y_array*mask
        df = img.to_pandas().dropna(subset=['Megaclasse'])
        del img
        rename_mapper = {c:f"{date}_{c.split('_')[-2]}"
            for c in df.drop(columns=['x','y','Megaclasse']).columns
        }
        df = df.rename(columns=rename_mapper)
        index_maker = lambda base, var: (base-var)/(base+var)
        df[f'{date}_NDVI'] = index_maker(base = df[f'{date}_B08'], var= df[f'{date}_B04'])
        df[f'{date}_NDBI'] = index_maker(base = df[f'{date}_B11'], var= df[f'{date}_B08'])
        df[f'{date}_NDMI'] = index_maker(base = df[f'{date}_B08'], var= df[f'{date}_B11'])
        df[f'{date}_NDWI'] = index_maker(base = df[f'{date}_B08'], var= df[f'{date}_B12'])
        df.sort_index(1).to_csv(os.path.join(CSV_SAVE_PATH,f'{date}.csv'), index=False)
