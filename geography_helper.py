import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc = 'progress')
from shapely.geometry import Point
from matplotlib import pyplot as plt
from shapely.wkt import dumps, loads
from sklearn.metrics.pairwise import haversine_distances

#Converts df with latitude and longitude columns to GeoDataFrame. Needed for many geometric/geographic computations.

def places_to_geom(places, plot = False):
    """
    Takes a DataFrame with columns named 'latitude' and 'longitude' and returns a GeoDataFrame with geometry corresponding to the lat/long points.
    Plots the points to check that conversion happened correctly. This can be suppressed by setting plot = False.
    
    arg: a df with 'longitude' and 'latitude' columns
    returns: GeoDataFrame with geometry of points"""
    places_copy = places.copy()
    places_locs = [Point(xy) for xy in zip(places_copy['longitude'], places_copy['latitude'])]
    places_gdf = gpd.GeoDataFrame(places_copy, geometry = places_locs)
    if plot:
        places_gdf.plot(markersize = .2)
    return places_gdf

#Functions for identifying tracts/regions containing a given point 
#and attaching them to the dataframe

def check_in_boundary(pointgdf, polygon):
    """
    Checks if each point in a GeoDataFrame is contained in a given shapely polygon.
    
    arg: 
        pointgdf: a GeoDataFrame of points
        polygon: a shapely polygon
    returns:
        Boolean pandas series. Each coordinate is the result of testing if the given point is in the polygon.
    """
    
    return pointgdf.geometry.progress_apply(lambda x: polygon.contains(x))


def tract_containing(point, censusgdf, id_col = 'GEOID'):
    """
    Finds the GEOID of the census tract containing the given point. 
    
    args:
        point: a shapely point given in longitude and latitude.
        censusdf: a GeoDataFrame of census tract shapes with a column named 'GEOID'
    returns:
        GEOD of census tract the point belongs to
    """
    censusgdf = censusgdf.reset_index()
    tracts = censusgdf.geometry
    results = np.ones(len(censusgdf))
    for idx, tract in enumerate(tracts):
        results[idx] = tract.contains(point)

    return censusgdf.loc[results.argmax(),id_col]

def add_census_tracts(places, censusgdf, id_col = 'GEOID'):
    """
    Converts dataframe with 'latitude' and 'longitude' columns to a GeoDataFrame and adds census tract data. Essentially just 'places_to_geom' composed with coordinatewise application of 'tract_containing', but adds a tqdm processing bar since this can take a few minutes on a large dataframe.
    
    args: 
        listings: DataFrame containing 'latitude' and 'longitude' columns
        censusgdf: GeoDataFrame of census tracts
    returns: GeoDataFrame of listings, together with point geometry for each listing, and a column showing which census tract each listing belongs to. Useful for merging census track-tagged data.
    """
    places_gdf = places_to_geom(places)
    places_gdf['tract_containing'] = places_gdf['geometry'].progress_apply(lambda x: tract_containing(x,censusgdf, id_col = id_col))
    return places_gdf
    
#GA shapely object given in lat/long needed for mapping.
#There are 'output' (creation) and 'input' (load saved boundary file) functions.
def create_GA_boundary_file(filename, censusgdf):
    """
    Creates a dump of a shapely file representing Georgia. Takes in a GeoDataFrame of census tracts to create the shape.
    """
    georgia = censusgdf.loc[censusgdf['STATEFP'] == 13]
    GAboundaries = censusgdf.dissolve(by = 'STATEFP').geometry[0]
    
    with open(filename, 'w') as f:
        f.write(dumps(GAboundaries))
        

def import_GA_boundary_file(filename):
    """
    Returns the GA boundary stored in the dump at the given file path.
    
    args:
        filename: string giving path to GA shape file dump
        returns: shapely polygon of GA
    """
    with open(filename) as f:
        GAboundary = loads(f.read())
    return GAboundary


#Distance related functions
def haversine_distance(x, y):
    """
    Haversine distance between two points given as [lat,long] in radians.
    args:
        x, y: arrays of shape (2)
        returns: distance in km
    """
    return 2* np.arcsin(np.sqrt(np.sin((x[0] - y[0])/2) ** 2 + np.cos(x[0])* np.cos(y[0]) * np.sin((x[1] - y[1])/2) ** 2)) * 6371000/1000

def lat_long_rad(dataframe):
    """
    arg: dataframe with 'latitude' and 'longitude' coordinates. 
    return: (n_samples, 2) numpy array with latitude/longitude coordinates converted to radians
    """
    return np.radians(dataframe[['latitude', 'longitude']].values)

def distances_from_dfs(df1, df2):
    """
    args: two dataframes with 'latitude' and 'longitude' coordinates
    returns: matrix of pairwise approximate distances in km between points in df1 and points in df2
    """
    return haversine_distances(lat_long_rad(df1), lat_long_rad(df2)) * 6371000/1000

def get_n_closest(df1, df2, n, limit = None):
    """
    Gets the row indices of the n closest points in df2 to each point in df1, within a limit, if desired.
    args: 
        df1 and df2: dataframes with 'latitude' and 'longitude' columns
        n: the number of nearest points in df2 to find
        limit: an optional limit in km
    return:
        A list the length of df1. The ith element in is a list of row indices of df2, identifying the n closest points of df2 to this element, within the limit distance. 
    """
    distances = distances_from_dfs(df1,df2)
    n_closest = distances.argsort()[:,:n]
    
    if limit:
        closest =[]
        for k in range(n_closest.shape[0]):
            closest.append([x for x in n_closest[k] if distances[k,x] <= limit])
        return closest
    else:
        return n_closest

#Includes functions for:
#filtering a set of POIs to a boundary,
#computing distance to POIs,
#and aggregating information about nearby points of interest
def get_aggregate(POIdf, subset, aggfunc):
    """
    Applies an aggregating function to a subset of a dataframe (e.g. of points of interest).
    
    args:
        POIdf: a dataframe
        subset: an iterable of indices of the POIdf
        aggfunc: an aggregating function
    returns:
        The value of the aggfunc applied to the subset of the POIdf.
    """
    return aggfunc(POIdf.iloc[subset,:])

def apply_local_aggfunc(df1, df2, aggfunc, n, limit = None, name = 'agg_value'):
    """
    For each location in df1, finds n closest points of df2 within the limit. Then, applies the aggfunc to df2 subbsetted to these rows, and returns the results in a series the same length as df1.
    
    args:
        df1 and df2: dataframes with 'latitude' and 'longitude' columns
        aggfunc: an aggregating function which can take in subsets of df2 and return a single value
        n: number of closest points of df2 to consider for each point of df1
        limit: distance in km specifying largest radius of neighborhood around each point of df1
        name: string for the aggregate value column name
    return:
        Pandas series of aggregate values, one for each point in df1.
    """
    neighborhood = get_n_closest(df1, df2, n, limit = limit)
    return pd.Series([get_aggregate(df2, x, aggfunc) for x in neighborhood], name = name)

def min_distance(df1, df2, name = 'closest_dist'):
    """
    df1 and df2 must be dataframes containing columns 'latitude' and 'longitude'. Uses haversine distance to find the distance from each point in df1 to the closest point in df2.
    """
    distances = distances_from_dfs(df1, df2)
    return pd.Series(distances.min(axis = 1), name = name)

def filter_by_boundary(df, boundary):
    """
    Takes in dataframe with 'latitude' and 'longitude' columns, and filters out rows which are not contained in the given boundary. The boundary should be a shapely polygon described in latitude and longitude.
    args:
        df: pandas dataframe with 'latitude' and 'longitude' columns
        boundary: shapely polygone given in lat/long
    returns:
        dataframe filtered to those inside the boundary
    """
    gdf = places_to_geom(df)
    GA_filter = check_in_boundary(gdf,boundary)
    print(f'Dropped {(~GA_filter).sum()} rows which were outside the boundary')
    print(f'{GA_filter.sum()} rows are remaining')
    return pd.DataFrame(gdf[GA_filter]).drop('geometry', axis = 1)
    