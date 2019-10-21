import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder

from .utils import get_2Dcoordinates_matrix


class pixel_selection:
    def __init__(self, df, polygon_id_col, class_col=None):
        """df must have only band values, polygon_id_col and class_col"""
        assert type(df)==pd.DataFrame, 'df needs to be of type `pd.DataFrame`.'
        assert type(polygon_id_col)==str and type(class_col) in [str, type(None)], 'Both polygon_id_col and class_col need to be of type `str`.'
        assert polygon_id_col in df.columns, f'{polygon_id_col} not in dataframe.'
        self.methods = ['som', 'bhattacharyya', 'kmeans', 'hierarchical']
        if not hasattr(self, '_previous_cluster_col'): self._previous_cluster_col = False
        self._polygon_id = polygon_id_col
        self.class_col = class_col
        self.df = df.sort_values(by=self._polygon_id)
        self.k = 12
        self.som_architectures = get_2Dcoordinates_matrix((5,5)).reshape((2,-1))
        self.som_architectures = self.som_architectures[:,np.apply_along_axis(lambda x: (x!=0).all() and (x!=1).any(), 0, self.som_architectures)].T

        if self.df[self._polygon_id].dtype == np.dtype('O'):
            self.is_string_identifier = True
            self.label_encoder = LabelEncoder().fit(self.df['Label'])
            self.df[self._polygon_id] = self.label_encoder.transform(self.df[self._polygon_id])
        else:
            self.is_string_identifier = False

        if class_col: drop_cols = [self._polygon_id, self.class_col]
        else: drop_cols = [self._polygon_id]

        polygon_list = np.split(self.df.drop(columns=drop_cols), np.where(np.diff(self.df[self._polygon_id]))[0]+1)
        # drop polygons with too few pixels to be relevant for classification
        self._polygon_list = [x for x in polygon_list if len(x)>=10]

    def get_clusters(self, method='SOM', cluster_col='clusters', identify_dominant_cluster=False, random_state=None):
        """stage 1"""
        assert method in self.methods, f'Method {method} not implemented. Possible options are {methods}'
        assert self._previous_cluster_col or method!='bhattacharyya', f'bhattacharyya method should only be used for consistency analysis.'

        if method == 'som':
            # Polygon clustering (SOM)
            labels = []
            indices = []
            total = len(self._polygon_list) # testing
            i=1
            for polygon in self._polygon_list:
                print(f'Clustering process: {i}/{total}'); i+=1
                indices.append(polygon.index)
                _labels = SOM_find_optimal_architecture_and_cluster(polygon.values, self.som_architectures, random_state)
                # generally you will want to use get_dominant_pixels only if the majority cluster is being passed for consistency analysis
                if identify_dominant_cluster:
                    labels.append(get_dominant_pixels(_labels))
                else:
                    labels.append(_labels)

        elif method == 'bhattacharyya':
            labels = []
            indices = []

            for polygon in self._polygon_list:
                a = self._df[self._df[self._previous_cluster_col].isin(polygon.index)]\
                    .drop(columns=[self._polygon_id, self._previous_polygon_id, self._previous_cluster_col])\
                    .values
                clusters_per_label = list(polygon.index)
                for clust in clusters_per_label:
                    b = self._df[self._df[self._previous_cluster_col]==clust]\
                        .drop(columns=[self._polygon_id, self._previous_polygon_id, self._previous_cluster_col])\
                        .values
                    distance = Bhattacharyya(a, b)
                    indices.append([clust])
                    labels.append([distance])
            indices_labels = np.array([indices,labels]).squeeze().T
            indices_labels = indices_labels[indices_labels[:,1].astype(float).argsort()]
            percentile_65 = int(indices_labels.shape[0]*.65)
            indices_labels[:percentile_65,1] = True
            indices_labels[percentile_65:,1] = False
            indices = np.expand_dims(indices_labels[:,0], 1).astype(str).tolist()
            labels  = np.expand_dims(indices_labels[:,1], 1).tolist()

        elif method in ['kmeans', 'hierarchical']:
            labels = []
            indices = []
            total = len(self._polygon_list)  # testing
            i = 1
            for polygon in self._polygon_list:
                print(f'Clustering process: {i}/{total}')
                i += 1
                indices.append(polygon.index)
                _labels = find_optimal_k_and_cluster(X=polygon.values, k_max=self.k, method=method, random_state=random_state)
                # generally you will want to use get_dominant_pixels only if the majority cluster is being passed for consistency analysis
                if identify_dominant_cluster:
                    labels.append(get_dominant_pixels(_labels))
                else:
                    labels.append(_labels)

        else:
            raise ValueError('method not yet implemented')

        self.cluster_col = cluster_col
        clusters = pd.Series(data=np.concatenate(labels), index=np.concatenate(indices), name=self.cluster_col)
        self.df = self.df.join(clusters)
        self.df[self.cluster_col] = self.df[self._polygon_id].astype(str)+'_'+self.df[self.cluster_col].astype(str)
        return self.df

    def get_consistency_analysis(self, consistency_col, method='SOM', class_col=None, cluster_col=None, random_state=None):
        """
        stage 2
        - SOM: Runs clustering based on Kohonen self-organizing maps
        - Bhattacharyya: Distance based selection (keeps 65% of the clusters closest to the "centroid of centroids")
        """
        if class_col: self.class_col = class_col
        if cluster_col: self.cluster_col = cluster_col
        assert self.cluster_col in self.df.columns, f'No columns with cluster id detected ({self.cluster_col}). Run self.get_clusters or manually add column with cluster values (pass column name on `cluster_col`)'
        assert type(self.cluster_col)!=type(None), '`cluster_col` is not defined.'
        assert method in self.methods, f'Method {method} not implemented. Possible options are {self.methods}'
        assert self.class_col in self.df.columns, f'{self.class_col} not in dataframe.'

        self._previous_polygon_id  = deepcopy(self._polygon_id)
        self._previous_cluster_col = deepcopy(self.cluster_col)
        self._df = deepcopy(self.df)

        pre_class_wide_clusters = self.df[[self.cluster_col, self.class_col]].drop_duplicates().set_index(self.cluster_col)
        class_wide_clusters = self.df.drop(columns=[self._polygon_id, self.class_col]).groupby([self.cluster_col]).mean()
        class_wide_clusters = class_wide_clusters.join(pre_class_wide_clusters)

        self.__init__(class_wide_clusters, self.class_col)

        self.k = 2
        cluster_info = self.get_clusters(method=method, cluster_col=consistency_col, identify_dominant_cluster=True, random_state=random_state)
        mapper = cluster_info[consistency_col].apply(lambda x: x.split('_')[-1]=='True')\
            .astype(int).to_dict()
        self._df[consistency_col] = self._df[self._previous_cluster_col].map(mapper)
        return self._df, cluster_info


def SOM_clustering(X, grid_shape, random_state=None):
    # setup SOM
    som = MiniSom(grid_shape[0],grid_shape[1],X.shape[1],sigma=0.8,learning_rate=0.6,random_seed=random_state)
    # fit SOM
    som.train_random(X, 2000)
    # assign labels to node
    labels = np.apply_along_axis(som.winner, 1, X).astype(str)
    return np.char.add(labels[:,0], labels[:,1])

def SOM_find_optimal_architecture_and_cluster(X, nodes, random_state=None):
    label_list = []
    CH_score = []
    for architecture in nodes:
        if X.shape[0]>=architecture[0]*architecture[1]:
            labels = SOM_clustering(X,architecture,random_state=random_state)
            # Paris et al. 2019 uses the Calinski Harabasz index to identify the number of clusters to use
            score = calinski_harabasz_score(X, labels)
            label_list.append(labels)
            CH_score.append(score)
    return label_list[np.argmax(CH_score)]


def find_optimal_k_and_cluster(X, k_max=12, method='kmeans', random_state=None):
    label_list = []
    CH_score = []
    for k in range(2,k_max+1):
        if X.shape[0] > k:
            if method == 'kmeans':
                labels = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state, n_jobs=None).fit_predict(X)
            elif method == 'hierarchical':
                labels = AgglomerativeClustering(n_clusters=k, linkage='single').fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            label_list.append(labels)
            CH_score.append(score)
    return label_list[np.argmax(CH_score)]

def get_dominant_pixels(labels):
    l = pd.Series(labels)
    labels_premapper = l\
            .groupby(labels)\
            .size()\
            .sort_values(ascending=False)\
            .to_frame()
    labels_premapper['labels_choice'] = [True]+[False for i in range(len(labels_premapper)-1)]
    mapper = labels_premapper[['labels_choice']].to_dict()['labels_choice']
    return l.map(mapper)


def Bhattacharyya(a, b):
    a_mean = np.expand_dims(a.mean(axis=0), 1)
    a_cov  = np.cov(a.T)
    b_mean = np.expand_dims(b.mean(axis=0), 1)
    b_cov  = np.cov(b.T)

    sigma = (a_cov + b_cov)/2
    sigma_inv = np.linalg.inv(sigma)

    term_1 = (1/8)*np.dot(np.dot((a_mean-b_mean).T,sigma_inv),(a_mean-b_mean))
    term_2 = (1/2)*np.log(np.linalg.det(sigma)/np.sqrt(np.linalg.det(a_cov)*np.linalg.det(b_cov)))
    return float(np.squeeze(term_1+term_2))
