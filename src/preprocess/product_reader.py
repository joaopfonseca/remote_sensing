
"""
TODO:
    - add custom indices
    - Assertions
    - Method for x and y min and max
    - documentation
"""

import os
import pickle
import numpy as np
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd


class SentinelProductReader:
    """
    Reads Sentinel 2 product and converts it into a matrix with band values.
    """

    def __init__(self, bands_path=None, bands=None, labels_shapefile=None, label_col=None):

        # # assertions:
        # bands_path must be None or point to a directory with .jp2 files having
        # names recognized as sentinel 2 images
        # Also check if bands_path exists
        # #
        # bands must be either None or type list
        # #
        # labels_shapefile: only .shp or .shx file types are supported

        self.bands_path = os.path.abspath(bands_path) if bands_path else None
        self.meta       = []
        self.X          = []
        self.X_labels   = []

        if bands_path and type(bands)==list:
            self.bands = bands
        elif bands_path and not bands:
            self.bands = [self._get_band_name(x) for x in os.listdir(bands_path)]

        if bands_path:
            _b_prefix = [
                x.replace(x.split('_')[-1],'')
                for x in os.listdir(bands_path) if x.endswith('.jp2')
            ]

            # assertion:
            # check if self.b_prefix length >= than bands length

            b_prefix = _b_prefix[0]
            for band in self.bands:
                band_path = os.path.join(bands_path, b_prefix+band+'.jp2')
                self.add_band(band_path)

        if labels_shapefile and label_col:
            self.add_labels(labels_shapefile, label_col)
        elif labels_shapefile and not label_col:
            raise TypeError('missing required argument \'label_col\'')

    def plot(self, bands=['B04', 'B03', 'B02'], show_legend=True, show_y=True, alpha=1, cmap='Pastel1', figsize=(20, 20), dpi=80, *args):
        # assertion:
        # check if X_array exists
        plt.figure(
            figsize=figsize,
            dpi=dpi
        )
        if type(bands)==list:
            map_arr = self.X_array[:,:,[self.bands.index(x) for x in bands]]
            pre_map_final = np.clip(map_arr, 0, 3000)/3000
        else:
            map_final = self.X_array[:,:,self.bands.index(bands)]

        plt.imshow(
            map_final,
            *args
        )

        if hasattr(self, 'y_array') and show_y:
            im = plt.imshow(
                np.ma.masked_where(self.y_array == self.y_fill, self.y_array),
                alpha=alpha,
                cmap=cmap
            )

        plt.axis('off')

        if show_legend and hasattr(self, 'y_array') and show_y:
            values = list(self.y_labels.values())
            labels = list(self.y_labels.keys())

            colors = [ im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[i], label=f'{labels[i]}' ) for i in values]
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.01, 1),
                loc=2,
                borderaxespad=0.,
                fontsize='large'
            )

    def add_labels(self, labels_shapefile, label_col):
        # assertion:
        # check in self.meta is not empty
        self.label_col = label_col
        gdf = gpd.read_file(labels_shapefile)\
            .to_crs(dict(self.meta[0]['crs']))\
            [[label_col, 'geometry']]

        keys = dict(pd.Series(gdf[label_col].unique()))

        self.y_labels = {v: k for k, v in keys.items()}
        gdf[label_col] = gdf[label_col].map(self.y_labels)
        self.y = gdf

        return self

    def add_band(self, band_path):
        band_name = self._get_band_name(band_path)
        if band_name not in self.X_labels:
            band = rasterio.open(band_path, driver='JP2OpenJPEG')
            self.X.append(band.read(1))
            self.X_labels.append(band_name)
            self.meta.append(band.meta)
        else:
            raise ValueError(f'Band \'{band_name}\' already in X at index {self.X_labels.index(band_name)}.')
        return self

    def get_X_array(self, high_resolution=True):
        hw = np.array([arr.shape for arr in self.X])
        idx = hw.argmax(axis=0)[0]
        self.height, self.width = hw[idx]
        X_final = []
        for arr in self.X:
            w, h = arr.shape
            X_resoluted = np.repeat(arr, repeats=self.height/h, axis=0)
            X_resoluted = np.repeat(X_resoluted, repeats=self.width/w, axis=1)
            X_final.append(X_resoluted)
        self.X_array = np.moveaxis(np.array(X_final), 0, -1)
        return self.X_array

    def get_y_array(self, fill=-1):
        self.y_fill = fill
        shapes = ((geom, value) for geom, value in zip(self.y['geometry'], self.y[self.label_col]))
        hw = np.array([arr.shape for arr in self.X])
        idx = hw.argmax(axis=0)[0]

        _meta = self.meta[idx]
        out_shape = tuple(hw[idx])
        transf = _meta['transform']

        burned = features.rasterize(
            shapes=shapes,
            out_shape=out_shape,
            fill=fill,
            transform=transf,
            all_touched=False,
            default_value=1,
            dtype='int32',
        )
        self.y_array = burned
        return self.y_array

    def _get_band_name(self, band_path):
        return band_path.split('_')[-1].split('.')[0]

    def to_pandas(self):

        # assertion:
        # check if X_array exists, else run get_X_array
        # check if y_array exists, else don't add label column
        columns = self.bands+[self.label_col]
        shp  = self.X_array.shape
        X_reshaped = np.reshape(self.X_array, (shp[0], shp[1]*shp[2]))
        y_reshaped = np.reshape(self.y_array, (1, shp[1]*shp[2]))
        data = np.concatenate([X_reshaped, y_reshaped])
        return pd.DataFrame(data=data.T, columns=columns)

    def add_indices(self, indices=['NDVI', 'NDBI', 'NDMI', 'NDWI']):
        assert type(indices)==list, '\'indices\' must be of type list'

        formula = lambda base, var: (base-var)/(base+var)
        indices_bands = {
            'NDVI': {'base': 'B08', 'var': 'B04'},
            'NDBI': {'base': 'B11', 'var': 'B08'},
            'NDMI': {'base': 'B08', 'var': 'B11'},
            'NDWI': {'base': 'B08', 'var': 'B12'},
        }
        for index in indices:
            if index not in self.X_labels:
                base_band, var_band = indices_bands[index].values()
                base = self.X[self.bands.index(base_band)]
                var = self.X[self.bands.index(var_band)]

                hw = np.array([arr.shape for arr in [base, var]])
                idx = hw.argmax(axis=0)[0]
                height, width = hw[idx]
                base_var = []
                for arr in [base, var]:
                    w, h = arr.shape
                    bv_resoluted = np.repeat(arr, repeats=height/h, axis=0)
                    bv_resoluted = np.repeat(bv_resoluted, repeats=width/w, axis=1)
                    base_var.append(bv_resoluted)

                feature = formula(base_var[0], base_var[1])

                self.X.append(feature)
                self.X_labels.append(index)
                self.meta.append(None)
        return self

    def dump(self, fname):
        pickle.dump(self, open(fname, 'wb'))
