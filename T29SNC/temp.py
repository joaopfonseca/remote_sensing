import numpy as np
import rasterio
import rasterio.plot as rplt
import geopandas as gpd

cos15 = gpd.read_file('shapefiles/cropped/shapefile/cos15_merged_29SNC_UTMZ29N.shp')

cos18 = gpd.read_file('COS2018/COS2018v1.shp')

rasterio.open('shapefiles/cropped/shapefile/cos15_merged_29SNC_UTMZ29N.shp')
