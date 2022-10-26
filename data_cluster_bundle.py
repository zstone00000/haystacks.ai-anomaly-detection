import pandas as pd
import numpy as np

class DataClusterBundle:
    '''
    Creates bundle with data, labels, and cluster results. 
    Endows the bundle with the methods needed to feed it to an AnomalyAnalyzer,
    but only requires data, coordinates, and labels.
    
    pca_data is be a dataframe with n_samples rows
    coords is a numpy array of shape (n_samples, 2) containing the coordinates 
    for each data point
    labels is a numpy array of shape (n_samples, ) containing the cluster labels
    for each data point
    '''
    def __init__(self, pca_data, coords, labels):
        self.data = pca_data
        self.coords = pd.DataFrame({'latitude' : coords[:,0], 'longitude' : coords[:,1]}, index = self.data.index)
        self.labels = labels
        
    def data_with_labels(self, clustername = 'cluster'):
        return pd.concat([self.coords, self.data, pd.Series(self.labels, name = clustername, index = self.data.index)], axis = 1)
        
    def get_cluster_sizes(self):
        """
        Returns dataframe showing the number of datapoints in each cluster.
        """
        return pd.Series(self.labels).value_counts().reset_index().rename({'index':'cluster_label', 0: 'n_members'}, axis = 1)