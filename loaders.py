import numpy as np
from linecache import getline as gl
import pandas as pd
import pdb
from scipy.optimize import fsolve
from functools import partial
import os
import pickle


from geography import augment

P_BASAL = 1e-5 ## Assumed frequency of Corona in entire population

f = lambda p, kappa, R: (1-p) * np.power(1 + p * R / kappa, kappa) - 1
def p_no_outbreak(kappa):
    if kappa > 1:
        ##               1 - P(major outbreak introducing one infected individual   )
        return lambda R: 1 - (fsolve(lambda p: f(p, kappa, R), x0=np.array(0.99))[0])
    else:
        return lambda R: np.minimum(1/R, 1)

    
def process(n, filename):
    line = gl(filename, n).replace('\n','').split(' ')
    return line[0], line[-1]


def load_density():
    filename = './data/gpw_v4_population_density_rev11_2020_2pt5_min.asc'
    pick_name = filename.replace('asc','pickle')
    if os.path.exists(pick_name):
        with open(pick_name,'rb') as f:
            data = pickle.load(f)
    else:
        data = np.loadtxt(filename, skiprows=6)
        data[(data==-9999)] = 0
        with open(pick_name, 'wb') as f:
            pickle.dump(data, f)

    f = lambda n: process(n, filename)
    specs = {k:float(v) for k,v in map(f,range(1,7))}
    specs['data'] = data
    return specs


def load_airports(specs):
    filename = './data/AirportInfo.csv'
    pick_name = filename.replace('csv', 'pickle')
    if os.path.exists(pick_name):
        with open(pick_name,'rb') as f:
            return pickle.load(f)
        
    airports = pd.read_csv(filename)
    g = lambda lat, lon: augment(lon=lon, lat=lat, specs=specs)
    airports = [{**row, **g(lat=row['Lat'], lon=row['Lon'])} for _, row in airports.iterrows()]
    airports = pd.DataFrame(airports)
    airports.index = airports.NodeName.astype(str)
    with open(pick_name,'wb') as f:
        return pickle.dump(airports, f)
    
    return airports


def load_travel(airports):
    filename = './data/Prediction_Monthly.csv'
    travel = pd.read_csv(filename)
    nodes = airports.NodeName.values
    travel = travel.query('Origin in @nodes and Dest in @nodes')    
    travel = travel.assign(origin_lon=travel.Origin.replace(airports.Lon),
                           origin_lat=travel.Origin.replace(airports.Lat),
                           dest_lon=travel.Dest.replace(airports.Lon),
                           dest_lat=travel.Dest.replace(airports.Lat))
    annual = travel.groupby(['Origin', 'Dest']).mean().reset_index(drop=False).drop('Month',axis=1)
    annual['Prediction'] = (annual.Prediction * 12).astype(int)
    annual['upper'] = (annual.upper * 12).astype(int)
    annual['lower'] = (annual.lower * 12).astype(int)
    datas = {month: data for month, data in travel.groupby('Month')}
    datas['annual'] = annual
    return datas


def augment_travel(travel, airports, destinations=None, p_basal=P_BASAL):
    airport_p_no_outbreak = dict(zip(airports.index, airports.p_no_outbreak_from_one))
    if destinations is not None:
        travel = travel.query("Dest in @destinations")
    p_no_outbreak_from_one = travel.Dest.replace(airport_p_no_outbreak)
    p_no_outbreak = np.power(p_no_outbreak_from_one, travel.Prediction * p_basal) ## Per origin and destination
    assert np.all(p_no_outbreak.values >= 0) and np.all(p_no_outbreak.values <= 1) 
    travel = travel.assign(p_no_outbreak=p_no_outbreak)
    travel['risk_ij'] = 1 - p_no_outbreak
    travel['risk_i'] = 1 - travel.groupby('Origin').p_no_outbreak.transform('prod')
    assert np.all(travel.risk_i.values >= 0) and np.all(travel.risk_i.values <= 1) 
    return travel
    


def calculate_p_no_outbreaks(airports, kappa):
    f = p_no_outbreak(kappa)
    if kappa > 1:
        return airports.R0.apply(f)
    else:
        return f(airports.R0)

if __name__ == '__main__':
    kappas = [1, 1, 1, 1, 1, 2]
    Rs     = [1, 2, 3, 4, 5, 2]

    ## Sanity check
    for R in Rs:
        p = p_no_outbreak(1)(R/3)
        p_eps = p_no_outbreak(1+1e-10)(R/3)
        assert abs(p-p_eps) < 1e-5, str(p) + '   ' + str(p_eps) 
    for kappa, R in zip(kappas, Rs):
        p = p_no_outbreak(kappa)(R)
        print(f'P(no outbreak|R={R}, kappa={int(kappa)}) = {p:2.3f}') 



