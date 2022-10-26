from mapper_clusterer import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from geography_helper import *
from sklearn.metrics import pairwise_distances
from geography_helper import places_to_geom
import geopandas as gpd

class AnomalyAnalyzer:
    @staticmethod
    def contains_na(dataframe, row_idx):
        return dataframe.iloc[row_idx,:].isna().any()
    
    fivemi = 8.04672
    
    def __init__(self,dcb, features, pca_basis = None, pca_latlong = False):
        """
        Creates anomaly analysis object. This object has methods for identifying and visualizing anomalies based on the result of a clustering algorithm, as well as for comparing them to similar or nearby clusters.
        
        A DataClusterBundle is required to initialize the analyzer. The DataClusterBundle contains the DataFrame a clustering model was trained on. Since this data may be transformed from the original features (by imputation, rescaling, dimension reduction, ...), a 'features' DataFrame must also be passed.
        
        To begin anomaly analysis, first run the get_all_anomalies method to set the anomalies. Re-running the method will overwrite any saved anomalies, so that analysis can be performed with different settings without having to initialize a new object.
        
        Parameters
        ----------
        dcb: a DataClusterBundle containing the data a cluster model was trained on together with the labels 
        
        features: DataFrame representing original features. 
        The row indices must match those of the data stored in the DataClusterBundle, and matching indices should correspond to the same observation.
        Must have 'latitude' and 'longitude' columns for creating the maps.
        Must have 'price' column to use any of the methods extracting price.
        
        pca_basis: numpy array of PCA 'loadings'.
        The number of columns must match that of the data stored in the DataClusterBundle, and the rows must match the columns of the 'features' DataFrame (without 'latitude' and 'longitude', unless specified with pca_latlong).
        This matrix should be the basis of the ambient PCA space containing the data in the DataClusterBundle, expressed in the coordinates of the (possibly transformed) original feature DataFrame. In other words, the columns of the matrix should be the images of the coordinate basis vectors (1,0,...), (0,1,0,...) under the inverse PCA transformation. 
        
        
        Attributes
        ----------
        dcb: access the DataClusterBundle passed to the constructor
        data: the data stored in the DataClusterBundle. This is the representation which was passed to the clustering algorithm
        anomalies: DataFrame storing the anomalies using the representation stored in the data attribute
        labeled_data: the data together with the lat/long and cluster labels
        cluster_sizes: DataFrame of the cluster labels and their sizes. Ordered from largest to smallest cluster.
        features: the original features DataFrame passed to the constructor. This is the  representation which was transformed into the one passed to the clustering algorithm.
        coords: DataFrame just of the latitude and longitude of the observations
        haversine_distances: matrix of geographic distances between all observations
        distances: matrix of euclidean distance between observations using the representation stored in the data attribute
        feature_list: list of feature names (without lat/long, unless specified), in the order they occur in the 'features' DataFrame. Used to align PCA-transformed data with the original features with the pca_basis, if provided.
        
        """
        #set data, labeled_data, and cluster_sizes attributes from the DataClusterBundle
        self.dcb = dcb
        self.data = dcb.data
        self.labeled_data = dcb.data_with_labels()
        self.cluster_sizes = self.dcb.get_cluster_sizes()
        
        #Set features, coordinates, and feature_list for the pca loadings from the features DataFrame
        self.features = features 
        self.coords = self.features[['latitude', 'longitude']]
        if pca_latlong: #only keep lat/long columns for loadings if user requests
            self.feature_list = list(features.columns)
        else:
            self.feature_list = list(set(features.columns).difference(set(['latitude','longitude'])))
        
        #load pca basis
        self.pca_basis = pca_basis
        
        #Compute distances
        self.distances = pairwise_distances(self.data) #distances in pca feature space
        self.haversine_distances = distances_from_dfs(self.coords,self.coords) #haversine distances
        
        #no anomalies at initialization
        self.anomalies = None
        
        
    
    def get_all_anomalies(self, lower_threshold, upper_threshold, drop_imputed=False):
        """
        Defines the anomalies as observations in clusters whose size is between the lower and upper threshold. Rows with imputed values may optionally be excluded from being classified as anomalies with the drop_imputed keyword.
        
        When the method is run, those observations get stored in the 'anomalies' attribute.
        """
        cluster_sizes = self.cluster_sizes
        spare_clusters = list(cluster_sizes[(cluster_sizes.n_members <= upper_threshold) 
                                            & (cluster_sizes.n_members >= lower_threshold)]['cluster_label'].values)
        labeled_data = self.labeled_data
        anomalies = labeled_data[labeled_data.cluster.isin(spare_clusters)].reset_index().rename({'index':'data_idx'}, axis = 1)  
        if not drop_imputed:
            self.anomalies =  anomalies
        else:
            anomaly_indices = list(anomalies['data_idx'])
            self.anomalies = anomalies[pd.Series([*map(lambda x: ~self.contains_na(self.features, x), anomaly_indices)])]
            
#General information extraction methods

    def house_price(self, row_idx):
        """
        Returns the price of the house
        """
        return self.features.loc[row_idx, 'price']
            
    def cluster_price(self, cluster_label):
        """
        Returns the array of the prices of all houses in a cluster.

        args:
            cluster_label: label from the 'cluster' column of the labeled_data
        """
        in_cluster = list(self.labeled_data[self.labeled_data['cluster'] 
                                            == cluster_label].index)
        return self.features.loc[in_cluster,'price'].values
            


#Methods for comparing a given house to SIMILAR houses and the clusters that contain them
#As opposed to geographically nearby

    def get_similar_points(self, row_idx, n = 1):
        """
        Returns the indices of the n most similar observations, in terms of euclidean distance between the points stored in the data attribute (e.g. in PCA representation).
        
        args:
            row_idx: index of the observation
            n: the number of similar observations to be found
        returns:
            indices of the n most similar observations
        """
        return self.distances[row_idx,:].argsort()[1:n+1]
    
    def similar_point_clusters(self, row_idx, n=1):
        """
        Given the index of an observation, returns all observations in the same cluster as one of the n observations most similar to the given one.
        
        args:
            row_idx: index of the observation
            n: the number of similar observations to be found
        returns:
            Subset of the labeled_data DataFrame of observations in the same cluster as one of the simlar observations.
        """
        close_points = self.get_similar_points(row_idx, n=n)
        close_clusters = []
        for point in close_points:
            close_cluster_label = self.labeled_data.loc[point, 'cluster']
            close_cluster = self.labeled_data[self.labeled_data['cluster'] == close_cluster_label]
            close_clusters.append(close_cluster)
            
        return pd.concat(close_clusters, axis = 0).drop_duplicates()
    
    def dif_from_clusters(self, row_idx, n = 1):
        """
        For each cluster containing one of the n most similar observations, computes the difference between the given observation and the mean (centroid) of each cluster, using the representation stored in the data attribute.
        
        args:
            row_idx: index of the observation
            n: number of similar observations to find
        returns:
            DataFrame of differences observation - cluster mean for each cluster containing a similar observation.
        """
        in_cluster = self.labeled_data.loc[row_idx,'cluster']
        print(f'Given point is in cluster {in_cluster}')
        cluster_means = self.similar_point_clusters(row_idx, n=n).groupby('cluster').mean()
        try:
            cluster_means.drop(['latitude', 'longitude'], axis = 1, inplace = True)
        except:
            pass
        cluster_difs = pd.DataFrame(np.array(self.data.loc[row_idx,:]).reshape(-1,1) - 
                                    np.array(cluster_means.T) ,
                           columns = cluster_means.index)
        cluster_difs.index = ['PC'+str(x) for x in np.arange(cluster_means.shape[1])+1]
        return cluster_difs
    
    def similar_cluster_prices(self, row_idx, n = 1):
        """
        Returns the average price for each cluster which contains one of the n observations most similar to the given observation.
        
        args:
            row_idx: index of the observation
            n: number of similar observations to find
        returns:
            DataFrame of clusters containing one of the n observations most similar to the given observation, together with the average price of each cluster.
        """
        similar_clusters = self.similar_point_clusters(row_idx, n = n)
        return pd.concat([self.features.loc[similar_clusters.index,'price'], 
           similar_clusters['cluster']], axis = 1).groupby('cluster').mean().reset_index() 
    
    def similar_cluster_prices_median(self, row_idx, n = 1):
        """
        Returns the average price for each cluster which contains one of the n observations most similar to the given observation.
        
        args:
            row_idx: index of the observation
            n: number of similar observations to find
        returns:
            DataFrame of clusters containing one of the n observations most similar to the given observation, together with the median price of each cluster.
        """
        similar_clusters = self.similar_point_clusters(row_idx, n = n)
        return pd.concat([self.features.loc[similar_clusters.index,'price'], 
           similar_clusters['cluster']], axis = 1).groupby('cluster').median().reset_index() 
    
    def significantly_cheaper_than_similar_houses(self, n =1, quant = 0.025):
        """
        Out of all the anomalies, returns those which are at least as cheap the house at the 2.5%ile amongst those in clusters containing one similar to it. Ideally, finds anomalies which are significantly cheaper than houses in the clusters it is most similar to.
        
        args:
            n: number of similar houses to look for for each anomaly
            quant: the price percentile that an anomaly must be at among similar houses to be considered "cheap"
        returns:
            list of indices of anomalies which are significantly cheaper than those in clusters with similar houses
        """
        similar_prices = lambda indx: np.concatenate([self.cluster_price(x) for x in
                            self.similar_point_clusters(indx, n = n)['cluster'].unique()])
        return [row_idx for row_idx in list(self.anomalies['data_idx'])
                if self.house_price(row_idx) <= np.quantile(similar_prices(row_idx), quant)]
    
    def compare_price_to_similar(self, row_idx, n =1):
        """
        Returns the DataFrame showing the (price of the house at row_idx) - (average price of similar cluster) for each similar cluster, where a similar cluster is any cluster containing a house among the n closest houses in the space containing the DataFrame stored in the data attribute (n most similar houses).
        
        args:
            row_idx: index of house
            n: number of closest houses in the data space to find
        returns:
            DataFrame of price difference between the house at row_idx and the average price of each cluster containing one of the most similar houses houses (in the feature space)
        """
        return self.house_price(row_idx) - np.concatenate([self.cluster_price(x) for x in
                 self.similar_point_clusters(row_idx, n = n)['cluster'].unique()]).mean()
    
#Methods for comparing house to clusters containing nearby houses
#When the clusters are geographically localized and represent similar houses,
#clusters containing a nearby house represent all houses similar to a nearby house in a nearby neighborhood
    
    def within_radius(self, row_idx, radius = fivemi):
        """
        Returns the row indices of all houses within the radius of the house specified by row_idx.
        
        args:
            row_idx: index of house
            radius: distance in km
        returns:
            NumPy array of indices of houses within the radius of the house at row_idx.
        """
        distances = self.haversine_distances[row_idx,:]
        return np.where(distances <= radius)[0]
    
    def clusters_within_radius(self, row_idx, radius = fivemi):
        """
        Returns all houses in any cluster containing a house within the radius of the house specified by row_idx. 
        
        args:
            row_idx: index of house
            radius: distance in km
        returns:
            Subset of the labeled_data DataFrame of all houses in any cluster containing a house within the radius of the house specified by row_idx.
        
        """
        close_points = self.within_radius(row_idx, radius = radius)
        close_clusters = []
        for point in close_points:
            close_cluster_label = self.labeled_data.loc[point, 'cluster']
            close_cluster = self.labeled_data[self.labeled_data['cluster'] == close_cluster_label]
            close_clusters.append(close_cluster)
        return pd.concat(close_clusters, axis = 0).drop_duplicates()
    
    def dif_from_clusters_in_radius(self, row_idx, radius = fivemi):
        """
        Shows how point differs from mean of clusters, out of
        those clusters containing the points within radius of the given point
        """
        in_cluster = self.labeled_data.loc[row_idx,'cluster']
        print(f'Given point is in cluster {in_cluster}')
        cluster_means = self.clusters_within_radius(row_idx, radius=radius).groupby('cluster').mean()
        try:
            cluster_means.drop(['latitude', 'longitude'], axis = 1, inplace = True)
        except:
            pass
        cluster_difs = pd.DataFrame(np.array(self.data.loc[row_idx,:]).reshape(-1,1) - np.array(cluster_means.T),
                           columns = cluster_means.index).reset_index(drop=True)
        cluster_difs.index = ['PC'+str(x) for x in np.arange(cluster_means.shape[1])+1]
        return cluster_difs
    
    def nearby_cluster_prices(self, row_idx, radius =fivemi):
        """
        Find the average price of each cluster containing a house within the radius of the house at row_idx.
        
        args:
            row_idx: index of house
            radius: distance in km
        returns:
            DataFrame of average price for each cluster containing a house within the radius. One column contains the cluster label, and the other the average price.
        """
        nearby_clusters = self.clusters_within_radius(row_idx, radius = radius)
        return pd.concat([self.features.loc[nearby_clusters.index,'price'], 
           nearby_clusters['cluster']], axis = 1).groupby('cluster').mean().reset_index()
    
    
    def nearby_cluster_prices_median(self, row_idx, radius =fivemi):
        """
        Find the median price of each cluster containing a house within the radius of the house at row_idx.
        
        args:
            row_idx: index of house
            radius: distance in km
        returns:
            DataFrame of median price for each cluster containing a house within the radius. One column contains the cluster label, and the other the median price.
        """
        nearby_clusters = self.clusters_within_radius(row_idx, radius = radius)
        return pd.concat([self.features.loc[nearby_clusters.index,'price'], 
           nearby_clusters['cluster']], axis = 1).groupby('cluster').median().reset_index()
    

    def significantly_cheaper_than_nearby_houses(self, radius =fivemi, quant = 0.025):
        """
        Find the row indexes of all anomalies which are in the bottom 'quant' quantile of price among all houses in a nearby cluster, where a nearby cluster is any cluster containing a house within the radius of the anomaly.
        
        args:
            radius: distance in km
            quant: float representing quantile
        returns:
            list of row indices of anomalies which are within the lower quantile of prices among all houses in the nearby clusters.
        
        """
        nearby_prices = lambda indx: np.concatenate([self.cluster_price(x) for x in
                            self.clusters_within_radius(indx, radius = radius)['cluster'].unique()])
        
        return [row_idx for row_idx in list(self.anomalies['data_idx'])\
                if self.house_price(row_idx) <= np.quantile(nearby_prices(row_idx), quant)]
    
    def compare_to_nearby_price(self, row_idx, radius =fivemi):
        """
        Returns the DataFrame showing the (price of the house at row_idx) - (average price of nearby cluster) for each nearby cluster, where a nearby cluster is any cluster containing a house within the radius of the house at the row_idx.
        
        args:
            row_idx: index of house
            radius: distance in km
        returns:
            DataFrame of price difference between the house at row_idx and the average price of each cluster containing a house within the radius.
        """
        return self.house_price(row_idx) - np.concatenate([self.cluster_price(x) for x in
             self.clusters_within_radius(row_idx, radius = radius)['cluster'].unique()]).mean()
    

# Plotting methods

    def map_anomalies(self, map_shape, title = None, fig = None, ax = None, **kwargs):
        """
        Plots the anomalies on a map. 
        
        args:
            map_shape = shape which can be used as geopandas geometry, given in lat/long
            title = string for title
            fig, ax = optionally pass matplotlib figure and axes objects to plot on a given axis
        
        Plots the image using geopandas.
        """
        
        #Create figure if fig, ax not passed
        if (not fig) or (not ax):
            fig,ax = plt.subplots(figsize = (10,10))
        
        #Plot background map shape
        boundary = gpd.GeoDataFrame(data = pd.DataFrame({'shape':['shape1']}), 
                                    geometry = [map_shape])
        boundary.plot(ax =ax, alpha = .5)

        places_to_geom(self.anomalies).plot(ax = ax, **kwargs)
        
        if title:
            plt.title(title)
        plt.axis('off')
        
    
    def map_single_and_clusters(self, single, clusters, map_shape, 
                                single_color = 'blue', single_alpha = .5, 
                                cluster_color = 'orange', cluster_alpha = .5, 
                                fig = None, ax = None, title = None):
        
        """Plot a single data point on the map together with a given cluster.
        
        args:
            single: row index of the point in data
            cluster: cluster label
            map_shape = shape which can be used as geopandas geometry, given in lat/long
        
        Plots the image using geopandas.
        """
        
        #Create figure if fig, ax not passed
        if (not fig) or (not ax):
            fig,ax = plt.subplots(figsize = (10,10))
        
        #Plot background map shape
        boundary = gpd.GeoDataFrame(data = pd.DataFrame({'shape':['shape1']}), 
                                    geometry = [map_shape])
        boundary.plot(ax =ax, alpha = .5)
        
        #plot cluster
        places_to_geom(self.labeled_data.loc[self.labeled_data['cluster'].isin(clusters)])\
            .plot(ax = ax, color = cluster_color, alpha = cluster_alpha)
        
        #plot single point
        places_to_geom(self.labeled_data[self.labeled_data.index == single])\
            .plot(ax = ax, color = single_color, alpha = single_alpha)
        
        if title:
            plt.title(title)
        plt.axis('off')
        
    def map_single(self, single, map_shape, color = 'blue',
                   alpha = .5, fig = None, ax = None):
        """Plot a single data point on the map.
        
        args:
        single: row index of the point in data
        map_shape = shape which can be used as geopandas geometry, given in lat/long
        
        Plots the image using geopandas.
        """
        if (not fig) or (not ax): #create fig,ax if not provided
            fig,ax = plt.subplots(figsize = (10,10))
            
        #Plot boundary
        boundary = gpd.GeoDataFrame(data = pd.DataFrame({'shape':['shape1']}), 
                                    geometry = [map_shape])
        boundary.plot(ax =ax, alpha = .5)
        
        #plot single point
        places_to_geom(self.labeled_data[self.labeled_data.index == single])\
            .plot(ax = ax, color = color, alpha = alpha)
        plt.axis('off')

        
#Methods for PCA loadings
        
    def get_loadings(self, pca_series):
        """
        Given a series representing a vector in PCA space, returns series representing its inverse image in the original feature space (possibly after transformations/imputation/etc.).
        
        args:
            pca_series: series representing PCA vector
        returns:
            series representing inverse image in original feature space
        """
        return pd.Series(np.dot(self.pca_basis, pca_series.values), index = self.feature_list)
    
    
    def get_all_loadings(self, dataframe):
        """
        Applies get_loadings to every column of input DataFrame. The columns of the DataFrame should each correspond to a PCA vector.
        
        returns:
            DataFrame of PCA loadings.
        """
        loadings = []
        for col in dataframe:
            loadings.append(self.get_loadings(dataframe[col]))
        new_df =  pd.concat(loadings,axis =1)
        new_df.columns = dataframe.columns
        return new_df