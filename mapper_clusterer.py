import kmapper as km
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from geography_helper import import_GA_boundary_file, places_to_geom, distances_from_dfs, haversine_distance
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph

def get_clusters_containing(ind, clustergraph):
    clusters = dict(clustergraph['nodes'])
    nodes = list(dict(clustergraph['nodes']).keys())
    edges = dict(clustergraph['links'])
    return [key for key in list(clusters.keys()) if ind in clusters[key]]

def branches_from_datapoint(ind, clustergraph):
    #A dictionary with cluster names as keys and lists of indices of datapoints contained in that cluster as values
    clusters = dict(clustergraph['nodes'])
    
    #The list of cluster names
    nodes = list(dict(clustergraph['nodes']).keys())
    
    #Dictionary with cluster names as keys and lists of cluster names it branches to as values
    edges = dict(clustergraph['links'])
    
    #Get the clusters containing the data index ind
    in_clusters = get_clusters_containing(ind,clustergraph)
    
    #Create the list of all clusters branching from any cluster containing the data index
    connected_to = in_clusters.copy() #it is connected to any cluster containing it
    for cluster in in_clusters:
        try:
            connected_to += edges[cluster] #and any cluster branching from one of those clusters
        except:
            continue
    
    #Create the list of all data indices belonging to any of the clusters connected to a cluster containing ind
    connected_to_data = []
    for cluster in connected_to:
        connected_to_data += clusters[cluster]
        
    #this is the collection of all data indices contained in a cluster branching from a cluster containing the data index ind
    #These can be thought of as the data points reachable from the datapoint ind
    return set(connected_to_data)

def graph_to_adjacency(data, clustergraph):
    """
    Takes in a mapper graph whose clusters contain indices corresponding to the rows of data. Returns a sparse matrix of size (n_samples, n_samples), where n_samples is the number of samples in the data. It represents the adjacency matrix of the graph whose nodes are the datapoints, and there is an edge between datapoints a and b if a and b are in the same cluster, or a is contained in a cluster which branches to a cluster containing b.
    args:
        data: dataframe
        clustergraph: a mapper graph such that each node contains indices of the rows of data
    returns: sparse adjacency matrix of graph on the data
    """

    #Create lists for coordinates where adjacency matrix nonzero
    samples = data.shape[0]
    row = []
    col = []
    
    for i in tqdm.tqdm(range(samples), total = samples):
        branches = list(branches_from_datapoint(i, clustergraph))
        for j in branches:
            row.append(i)
            col.append(j)
    
    vals = np.ones(len(col))
    return csr_matrix((vals, (row,col)), dtype = np.int8)

class ClusterOverCoords:
    """
    Clusters using the specified clusterer using the mapper algorithm over the coordinates. 'data' and 'coords' must have the same length. Each row of the data will be associated with the coordinates at the same row of coords by the mapper. 
    
    Initializing a ClusterOverCoords instance requires data, coords, and clustering algorithm with a .fit_predict method. The data must be a dataframe or numpy array, and coords must be a numpy array of the same length with 2 columns representing coordinates. A default cover for the mapper of 20 cubes and 30% overlap is assumed if none is provided. A precomputed distance matrix can also be passed as the data if precomputed is set to True, and the clustering algorithm also accepts a distance matrix.
    
    To immediately run all methods needed to generate cluster labels, run the method generate_clusters(). Then, the adjacency matrix on the data, mapper graph, and cluster labels can be accessed. If the object has a mapper graph as its graph attribute, the export_graph(filepath) method can be used to export a visualization of the local cluster graph.
    
    attributes:
        data: the data to be clustered
        coords: the (x,y) or (lat,long) coordinates associated with each row of data
        clusterer: clustering algorithm with .fit and .predict methods
        cover: the keplermapper cover being used
        graph: the mapper graph of the data. 
        A: sparse adjacency matrix of mapper graph extended to the data of size (n_samples, n_samples)
        components: numpy array of labels of connected components of the mapper graph on the data. These can be thought of as cluster global cluster labels. Has size (n_samples,)
        precomputed: boolean parameter specifying if the data is a precomputed distance matrix. When set to true, make sure that clustering algorithm also accepts distance matrices, and set its precomputed parameter to true as well, if needed.
        
    methods:
        generate_clusters(): runs all methods needed to generate mapper graph, adjacency matrix on the data, and identify the components
        generate_adjacency(): generates sparse adjacency matrix on the data. Will run method to construct the graph if none available
        make_graph(): makes mapper graph without running any other methods. Useful to check if graph is reasonable before generating adjacency matrix and component list.
        export_graph(filepath): exports mapper vizualization to filepath. If graph attribute is empty, runs make_graph() first.
         
    """
    def __init__(self, data, coords: np.array, clusterer, cover = km.Cover(n_cubes = 20, perc_overlap = 0.3), precomputed = False):
        self.data = data
        self.coords = coords
        self.clusterer = clusterer
        self.cover = cover
        self.graph = None
        self.A = None
        self.components = None
        self.precomputed = precomputed
        self.n_clusters = None
        self._mapper = None
        
        try:
            assert isinstance(self.data, (pd.DataFrame, np.ndarray))
        except:
            raise TypeError('Data must be a pandas DataFrame or numpy array.')           
        try:
            assert self.data.shape[0] == self.coords.shape[0]
        except:
            raise ValueError('data and coords must be the same length')        
        try:
            assert self.coords.shape[1] == 2
        except:
            raise ValueError('coords must be a numpy array of shape (n_samples, 2)')
        
    def make_graph(self):
        self._mapper = km.KeplerMapper()

        self.graph = self._mapper.map(
        self.coords,
        X= self.data.copy(),
        clusterer = self.clusterer,
        cover=self.cover,
        precomputed = self.precomputed
    )
    
    def export_graph(self, filepath = 'output.html'):
        if self.graph is None:
            self.make_graph()
            
        _ = self._mapper.visualize(self.graph, path_html=filepath)
      
        
    def generate_adjacency(self):
        if self.graph is None:
            self.make_graph()
            
        self.A = graph_to_adjacency(self.data,self.graph)
        
    def generate_clusters(self):
        if self.A is None:
            self.generate_adjacency()
        graph = csgraph.connected_components(self.A, directed = False)
            
        self.components = graph[1]
        self.n_clusters = graph[0]
    
    def plot_cluster_map(self, figsize = (10,10), fig = None, ax = None, **kwargs):
        #XXX Check that this works. Want to be able to pass fig, ax.
        #If it doesnt work the original just didn't have those arguments
        #and fig, ax were generated at the beginning unconditionally
        if (not ax) or (not fig):
            fig, ax = plt.subplots(figsize = figsize)
        gdf = places_to_geom(pd.DataFrame({'latitude' : self.coords[:,0], 'longitude' : self.coords[:,1]}))
        
        gdf.plot(c = self.components, ax = ax, **kwargs)
        
    def get_cluster_sizes(self):
        """
        Returns dataframe showing the number of datapoints in each cluster.
        """
        return pd.Series(self.components).value_counts().reset_index().rename({'index':'cluster_label', 0: 'n_members'}, axis = 1)
    
    def data_in_clusters(self, cluster_list):
        return np.hstack([np.where(self.components == x)[0] for x in cluster_list])
    
    def data_with_labels(self, clustername = 'cluster', include_coords = True):
        if include_coords:
            coords_df = pd.DataFrame({'latitude' : self.coords[:,0], 'longitude' : self.coords[:,1]})
            return pd.concat([coords_df, self.data, pd.Series(self.components, name = clustername)], axis = 1)
        else:
            return pd.concat([self.data, pd.Series(self.components, name = clustername)], axis = 1)
    
        
