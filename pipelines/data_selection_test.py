
import pandas as pd
from src.preprocess.readers import SentinelProductReader
from src.preprocess.utils import get_2Dcoordinates_matrix
from src.reporting.visualize import plot_image


## configs
DATA_PATH = 'data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = DATA_PATH+'../interim/'
PROCESSED_PATH = DATA_PATH+'../processed/'
CSV_PATH = PROCESSED_PATH+'picture_data.csv'

read_csv = True

## read data
if read_csv:
    df = pd.read_csv(CSV_PATH, nrows=30000000)
else:
    sentdat = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='ID'
    )
    df = sentdat.to_pandas()
    sentdat.add_labels(COS_PATH, 'Megaclasse')
    classtype = sentdat.to_pandas()['Megaclasse']
    df['Megaclasse'] = classtype
    df.to_csv(CSV_PATH, index=False)

#{0: 'Territórios artificializados',
# 1: 'Agricultura',
# 2: 'Pastagens',
# 3: 'Sistemas agro-florestais',
# 4: 'Florestas',
# 5: 'Matos',
# 6: 'Espaços descobertos ou com vegetação esparsa',
# 7: 'Zonas húmidas',
# 8: 'Corpos de água'}


## Preprocess

# drop unlabelled pixels
df = df[df['Megaclasse']!=-1]

# get specific example
polygon = df[df['ID']==33764].reset_index() #30636, 15440, 6946, 36403, 33764
polygon2 = polygon[['B01', 'B03', 'B02', 'B06', 'B12', 'B07', 'B11', 'B05', 'B04',
       'B10', 'TCI', 'B09', 'B08', 'B8A']]


## Unsupervised Training Sets Identification: Altered version from Paris et al. 2019

# Polygon clustering (SOM)
som_architectures = get_2Dcoordinates_matrix((4,4)).reshape((2,-1))
som_architectures = som_architectures[:,np.apply_along_axis(lambda x: (x!=0).all() and (x!=1).any(), 0, som_architectures)]
labels = find_optimal_architecture_and_cluster(polygon2.values, som_architectures.T, 0)

# Take dominant cluster
polygon['label'] = pd.Series(labels)
labels_premapper = polygon.groupby(['label']).size().sort_values(ascending=False).to_frame()
labels_premapper['labels_choice'] = ['keep']+['discard' for i in range(len(labels_premapper)-1)]
mapper = labels_premapper[['labels_choice']].to_dict()['labels_choice']
polygon['status'] = polygon['label'].map(mapper)

plt_res = polygon[['x', 'y']] - polygon[['x', 'y']].min()
plt_res[['B02', 'B03', 'B04']] = polygon[['B02', 'B03', 'B04']].clip(0,3000)/3000
plt_res['status'] = polygon['status']



# plot results on specific polygon
img = np.array([plt_res.pivot('x', 'y', band).values for band in ['B04', 'B03', 'B02']]).T
_accepted = (plt_res.pivot('x','y','status').values=='keep').T.astype(float)
accepted = img*np.array([_accepted for i in range(3)]).T.swapaxes(0,1)
_rejected = (plt_res.pivot('x','y','status').values!='keep').T.astype(float).T.swapaxes(0,1)
rejected = img*np.array([_rejected for i in range(3)]).T.swapaxes(0,1)
plot_image([img, rejected, accepted], num_rows=1, figsize=(40, 20), dpi=20)




# Polygon Consistency Analysis
# Bhattacharyya distance for each cluster to determine its distance from the the whole set of clusters

# Select only the clusters within the 65th percentile of the cluster distances

# (alternative method to test): rerun Polygon Clustering process in the set of dominant clusters to identify data points to keep


# Stratified Random Sampling: Generate balanced training sets proportionate to the original prior probabilities of the land-cover classes
