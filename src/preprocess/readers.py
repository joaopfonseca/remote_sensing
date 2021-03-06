
"""
TODO:
    - There's a bug in the plot function
    - add custom indices
    - Assertions
    - Method for x and y min and max
    - documentation
    - Sort bands (!!!)
"""

import os
import itertools
import pickle
import numpy as np
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd
from .utils import get_2Dcoordinates_matrix


class ProductReader:
    """
    Base class for all product readers. Should not be used on its own.
    """
    def __init__(self):
        pass

    def plot(self, bands=['B04', 'B03', 'B02'], show_legend=True, show_y=True, alpha=1, cmap='Pastel1', figsize=(20, 20), dpi=80, *args):
        # assertion:
        # check if X_array exists
        plt.figure(
            figsize=figsize,
            dpi=dpi
        )
        if type(bands)==list:
            map_arr = self.X_array[:,:,[self.bands.index(x) for x in bands]]
            map_final = np.clip(map_arr, 0, 3000)/3000
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
        out_shape = tuple(hw[idx][:2])
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

    def add_labels(self, labels_shapefile, label_col):
        # assertion:
        # check in self.meta is not empty
        self.label_col = label_col
        gdf = gpd.read_file(labels_shapefile)\
            .to_crs(dict(self.meta[0]['crs']))\
            .dropna()\
            [[label_col, 'geometry']]

        self.y_labels = dict(pd.Series(gdf[label_col].unique()))

        y_labels_inv = {v: k for k, v in self.y_labels.items()}
        gdf[label_col] = gdf[label_col].map(y_labels_inv)
        self.y = gdf
        #self.get_y_array()
        return self

    def _get_band_name(self, band_path):
        return 'B'+band_path.split('B')[-1].split('_')[0]

    def to_pandas(self):

        # assertion:
        # check if X_array exists, else run get_X_array
        # check if y_array exists, else don't add label column
        columns = ['y', 'x']+self.X_labels+[self.label_col]
        shp  = self.X_array.shape
        coords = np.indices(shp[:-1])
        data = np.concatenate([
            coords,
            np.moveaxis(self.X_array,-1,0),
            np.expand_dims(self.y_array,0)]
        ).reshape((len(columns), shp[0]*shp[1]))
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

    def add_custom_index(self, band1, band2, name='Custom index'):
        formula = lambda base, var: (base-var)/(base+var)
        if index not in self.X_labels:
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

    def dump(self, fname):
        del self.X_array, self.y_array
        pickle.dump(self, open(fname, 'wb'))
        self.get_X_array()
        self.get_y_array()


class SentinelProductReader(ProductReader):
    """
    Reads Sentinel 2 product (jpeg2000 format) and converts it to a matrix with band values.
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
            self.bands = [band for band in os.listdir(bands_path) if band.split('_')[-2] in bands]
        elif bands_path and not bands:
            self.bands = os.listdir(bands_path)

        if bands_path:

            # assertion:
            # check if self.b_prefix length >= than bands length

            for band in self.bands:
                band_path = os.path.join(self.bands_path, band)
                self.add_band(band_path)
            #self.get_X_array()

        if labels_shapefile and label_col:
            self.add_labels(labels_shapefile, label_col)
        elif labels_shapefile and not label_col:
            raise TypeError('missing required argument \'label_col\'')

    def add_band(self, band_path):
        band_name = band_path.split('/')[-1]
        if band_name.endswith('.jp2'):
            if band_name not in self.X_labels:
                band = rasterio.open(band_path, driver='JP2OpenJPEG')
                self.X.append(band.read(1))
                self.X_labels.append(band_name)
                self.meta.append(band.meta)
            else:
                raise ValueError(f'Band \'{band_name}\' already in X at index {self.X_labels.index(band_name)}.')
        return self


class TIFFProductReader(ProductReader):
    """
    Reads a .tiff product and converts it to a matrix with band values.
    """
    def __init__(self, TIFF_path, band_names=None, labels_shapefile=None, label_col=None):
        # TODO: check if len band names is the same as the number of bands
        # check condition missing required argument \'label_col\''

        self.TIFF_path = os.path.abspath(TIFF_path) if TIFF_path else None
        self.meta       = []
        self.X          = []
        self.X_labels   = []

        if type(band_names)==list:
            self.bands = band_names
            self.read_tiff(TIFF_path, band_names=self.bands)
        else:
            self.tiff_files = [i for i in os.listdir(TIFF_path) if i.endswith('.tif')]
            num_bands = [self._get_num_bands(os.path.join(TIFF_path, x)) for x in self.tiff_files]
            self.bands = [
                f'{i.replace(".tif","")}_{n}'
                for i,n
                in zip(
                    np.repeat(self.tiff_files, num_bands),
                    list(itertools.chain(*[list(range(m)) for m in num_bands]))
                    )
                ]
            self.read_tiff(self.TIFF_path, band_names=self.bands)



        if labels_shapefile and label_col:
            self.add_labels(labels_shapefile, label_col)
        elif labels_shapefile and not label_col:
            raise TypeError('missing required argument \'label_col\'')

    def read_tiff(self, filepath, band_names=None):
        for file in self.tiff_files:
            raster = rasterio.open(os.path.join(filepath, file))
            self.X.append(np.moveaxis(raster.read(), 0, -1))
            self.meta.append(raster.meta)
            self.X_labels.extend(band_names)
        return self

    def get_X_array(self, high_resolution=True):
        hw = np.array([arr.shape for arr in self.X])
        idx = hw.argmax(axis=0)[0]
        self.height, self.width, _ = hw[idx]
        X_final = []
        for arr in self.X:
            w, h, _ = arr.shape
            X_resoluted = np.repeat(arr, repeats=self.height/h, axis=0)
            X_resoluted = np.repeat(X_resoluted, repeats=self.width/w, axis=1)
            X_final.append(X_resoluted)
        self.X_array = np.moveaxis(np.array(X_final), 0, -1)
        return self.X_array


    def _get_num_bands(self, tiff_path):
        return rasterio.open(tiff_path).count
