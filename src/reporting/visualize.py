"""
TODO:
    - visualize entire map (same as in self.plot from product_reader.py)
    - visualize ground truth or predicted product
    - visualize rgb product
    - multiplot rgb product, ground truth, predicted product and
"""
import matplotlib.pyplot as plt
import numpy as np

def _plot_image(X, figsize=(20, 20), dpi=80, *args):
    if X.ndim==3 and (X>1.0).any():
        X = np.clip(X, 0, 3000)/3000
    plt.imshow(
        X,
        *args
    )
    plt.axis('off')

def plot_image(arrays, num_rows=1, figsize=(40, 20), dpi=80, *args):
    assert type(arrays) in [np.ndarray, list], '\'arrays\' must be either a list of arrays, or a single 2-dimensional array'

    if type(arrays)==np.ndarray:
        plt.figure(
            figsize=figsize,
            dpi=dpi
        )
        _plot_image(arrays.astype(int), figsize=figsize, dpi=dpi, *args)
    else:
        num_arrays = len(arrays)
        plt.figure(
            figsize=figsize,
            dpi=dpi
        )
        for i in range(num_arrays):
            plt.subplot(num_rows, int(np.ceil(num_arrays/num_rows)), i+1)
            _plot_image(arrays[i].astype(float))
