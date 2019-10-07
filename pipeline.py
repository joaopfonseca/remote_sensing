
import pickle
from src.preprocess.product_reader import SentinelProductReader
from src.models.HybridSpectralNet import HybridSpectralNet
from src.preprocess.utils import (
    ZScoreNormalization,
    createImageCubes,
    split_data,
    applyPCA
)
from src.reporting.reports import reports # make this structure more proper

DATA_PATH = 'data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = 'data/sentinel_coimbra/interim/'

test_ratio = 0.7
window_size = 25
read_pickle = True
random_state = 0
K=10

if not read_pickle:
    coimbrinhas = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='COS2015_Le' # Much more complex labels, originally was 'Megaclasse'
    )
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.dump(INTERIM_PATH+'coimbra.pkl')
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()
    coimbrinhas.plot(alpha=0.5)
else:
    coimbrinhas = pickle.load(open(INTERIM_PATH+'coimbra.pkl', 'rb'))
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()


labels = coimbrinhas.y_labels
X = coimbrinhas.X_array
y = coimbrinhas.y_array
del coimbrinhas

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

ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=40)
ConvNet.fit(X_train, y_train, batch_size=256, epochs=100, filepath='best_model.hdf5')

y_true = y_test[:1000]
y_pred = ConvNet.predict(X_test[:1000], filepath='best_model.hdf5')
reports(y_true, y_pred, labels)

plot_image([X_test, y_true, y_pred])
