
from src.preprocess.readers import SentinelProductReader

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
    df = pd.read_csv(CSV_PATH)
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
