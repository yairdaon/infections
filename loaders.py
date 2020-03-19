import numpy as np
from linecache import getline as gl
import pandas as pd
import pdb
from scipy.optimize import fsolve
from functools import partial

from geography import augment

f = lambda p, kappa, R: (1-p) * np.power(1 + p * R / kappa, kappa) - 1
def p_outbreak(kappa, n=1):
    return lambda R: 1 - (1 - (fsolve(lambda p: f(p, kappa, R), x0=np.array(0.99))[0]))**n
    
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


def load_airports(specs, n=1, kappa=1, wuhan_R0=4):
    airports = pd.read_csv('data/AirportInfo.csv')
    g = lambda lat, lon: augment(lon=lon, lat=lat, specs=specs)
    airports = [{**row, **g(lat=row['Lat'], lon=row['Lon'])} for _, row in airports.iterrows()]
    airports = pd.DataFrame(airports)
    airports.index = airports.NodeName.astype(str)
    wuhan_factor =  wuhan_R0 / airports.loc['WUH', 'density'] # R_0 / density for Wuhan
    airports['R0'] = airports.density * wuhan_factor
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


def augment_travel(travel, airports):
    airport_p_outbreak = dict(zip(airports.index, airports.p_outbreak))
    travel = travel.assign(origin_p_outbreak=travel.Origin.map(airport_p_outbreak),
                           dest_p_outbreak=travel.Dest.map(airport_p_outbreak))
    travel.dropna()
    travel['outgoing_total'] = travel.groupby('Origin').Prediction.transform('sum')
    travel = travel.query('outgoing_total > 0')

    travel['P_ij'] = travel.Prediction / travel.outgoing_total
    travel['risk_ij'] = travel.P_ij - travel.P_ij * travel.dest_p_outbreak
    travel['risk_i'] = travel.groupby('Origin').risk_ij.transform('sum').clip(lower=0, upper=1)
    travel['origin_lon'] = travel.Origin.replace(airports.Lon)
    travel['origin_lat'] = travel.Origin.replace(airports.Lat)
    travel['dest_lon'] = travel.Dest.replace(airports.Lon)
    travel['dest_lat'] = travel.Dest.replace(airports.Lat)
    # travel['red'] = travel.risk_i
    # travel['blue'] = (1 - travel.risk_i)
    # travel['green'] = 0
    assert np.max(np.abs(travel.groupby('Origin').P_ij.sum()-1)) < 1e-9
    return travel


def calculate_outbreaks(airports, kappa, n):
    f = p_outbreak(kappa, n)
    return airports.assign(p_outbreak=airports.R0.apply(f))


if __name__ == '__main__':
    kappas = [1, 1, 1, 1, 1, 2]
    Rs     = [3, 4, 5, 2, 3, 2]
    ns     = [1, 1, 1, 2, 2, 1]

    for kappa, R, n in zip(kappas, Rs, ns):
        p = p_outbreak(kappa, n=n)(R)
        print(f'P(outbreak|R={R}, n={n}, kappa={kappa}) = {p:2.3f}') 



