import sys
import os
#PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
#sys.path.append(PROJ_PATH)
#print(PROJ_PATH)
PROJ_PATH = '.'
## data manipulation and transformation
import numpy as np
import pandas as pd

################################################################################
# CONFIGS
################################################################################
## data path configs
DATA_PATH = PROJ_PATH+'/data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = PROJ_PATH+'/data/sentinel_coimbra/interim/'
PROCESSED_PATH = DATA_PATH+'../processed/'
CSV_PATH = PROCESSED_PATH+'classification_results_resnet.csv'

################################################################################
# DATA READING AND PREPROCESSING
################################################################################

df = pd.read_csv(CSV_PATH).drop(columns=['Unnamed: 0'])
noisy_lookup = df.pivot('x','y','y_pred').fillna(-1).values
clean_lookup = df.pivot('y','x','Megaclasse').fillna(-1).values
