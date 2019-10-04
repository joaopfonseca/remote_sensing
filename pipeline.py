
import pickle
from get_data_matrix import product_reader
from cnn_test import (
    createImageCubes,
    split_data,
    applyPCA,
    HybridSpectralNet,
    HybridSpectralNetMine
)

PRODUCT_PATH = 'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = 'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
test_ratio = 0.7
window_size = 25
read_pickle = False
random_state = 0
K=10

if not read_pickle:
    coimbrinhas = product_reader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='COS2015_Le' # Much more complex labels, originally was 'Megaclasse'
    )
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()
    coimbrinhas.dump('coimbra.pkl')
else:
    coimbrinhas = pickle.load(open('coimbra.pkl', 'rb'))



X = coimbrinhas.X_array
y = coimbrinhas.y_array

## standardization

i = 5000
X_sample, zscore = ZScoreNormalization(X[i:i+400, i:i+400], axes=(0,1))
y_sample = y[i:i+400, i:i+400]

X_sample, pca = applyPCA(X_sample, numComponents=K)

X_sample, y_sample = createImageCubes(X_sample, y_sample, window_size=window_size)

X_train, X_test, y_train, y_test = split_data(
    X_sample,
    y_sample,
    test_size=test_ratio,
    random_state=random_state,
    stratify=y_sample
)

ConvNet = HybridSpectralNetMine(input_shape=(window_size, window_size, K), output_units=40)
ConvNet.fit(X_train, y_train, batch_size=256, epochs=100, filepath='best_model.hdf5')
ConvNet.classification_report(X_test, y_test.reshape((y_test.shape[0],1)))
