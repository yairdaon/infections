import numpy as np
from linecache import getline as gl
import pandas as pd
import pdb

from geography import augment

def process(n, filename):
    line = gl(filename, n).replace('\n','').split(' ')
    return line[0], line[-1]


def load_density():
    filename = 'data/gpw_v4_population_density_rev11_2020_2pt5_min.asc'
    data = np.loadtxt(filename, skiprows=6)
    data[(data==-9999)] = 0

    f = lambda n: process(n, filename)
    specs = {k:float(v) for k,v in map(f,range(1,7))}
    specs['data'] = data
    return specs

def load_airports(specs, outbreak_estimation=None):
    airports = pd.read_csv('data/AirportInfo.csv')
    g = lambda lat, lon: augment(lon=lon, lat=lat, specs=specs)
    airports = [{**row, **g(lat=row['Lat'], lon=row['Lon'])} for _, row in airports.iterrows()]
    airports = pd.DataFrame(airports)
    airports.index = airports.NodeName.astype(str)
    wuhan_factor =  4 / airports.loc['WUH', 'density'] # R_0 / density for Wuhan
    airports['R0'] = airports.density * wuhan_factor
    airports['p_outbreak'] = 1 - 1 / airports.R0
    airports.loc[airports.p_outbreak < 0, 'p_outbreak'] = 0
    return airports


def load_travel(airports):
    filename = 'data/Prediction_Monthly.csv'
    travel = pd.read_csv(filename)
    nodes = airports.NodeName.values
    travel = travel.query('Origin in @nodes and Dest in @nodes')
    
    annual = travel.groupby(['Origin', 'Dest']).mean().reset_index(drop=False).drop('Month',axis=1)
    datas = {month: data for month, data in travel.groupby('Month')}
    datas['annual'] = annual
    return datas
