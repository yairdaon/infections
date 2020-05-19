import numpy as np
from linecache import getline as gl
import pandas as pd
import pdb
from scipy.optimize import fsolve
from functools import partial
import os
import pickle


from geography import augment

P_BASAL = 1e-6 ## Assumed frequency of Corona in entire population

f = lambda p, kappa, R: (1-p) * np.power(1 + p * R / kappa, kappa) - 1
def p_no_outbreak(kappa):
    if kappa == 'superspreaders':
        return p_no_outbreak_superspreaders
    elif kappa == 1:
        return lambda R: np.minimum(1/R, 1)
    elif kappa > 1:
        ##               1 - P(major outbreak introducing one infected individual   )
        return lambda R: 1 - (fsolve(lambda p: f(p, kappa, R), x0=np.array(0.99))[0])
    else:
        raise ValueError('Invalid kappa.')

    
def p_no_outbreak_superspreaders(R,
                                 alpha=0.01,
                                 factor=5,
                                 gamma=1/7):
    '''alpha is the fraction of population which is a super spreader
    factor is the increase in infectivity for a super spreader 
    gamma is the infection period (???)  
    beta is the number of days an individual is infected until removal from population (???)
    overall, R0 = beta / gamma
    '''
    beta = R * gamma
    def f(x, alpha, factor, gamma, beta):
        beta1 = factor * beta
        beta2 = beta
        y = np.zeros(2)

        rat1 = beta1 / (beta1 + gamma)
        res1 = gamma / (beta1 + gamma)
        y[0] = rat1 * (alpha * x[0]**2 + (1-alpha) * x[0]*x[1]) + res1 - x[0]

        rat2 = beta2 / (beta2 + gamma)
        res2 = gamma / (beta2 + gamma)
        y[1] = rat2 * (alpha * x[0]*x[1] + (1-alpha) * x[1]**2) + res2 - x[1]
        return y

    np.random.seed(19) ## For reproducibility
    x0s = np.random.uniform(size=(10,2))
    solutions = [fsolve(f, x0=x0, args=(alpha, factor, gamma, beta)) for x0 in x0s]
    solutions = [alpha * x[0] + (1-alpha) * x[1] for x in solutions]
    return np.min(solutions)

    
def get_destinations_dict(valid):
    '''get data from https://ourairports.com/data/'''
    destinations = pd.read_csv('./data/airports.csv')
    dest_dict = {}
    dest_dict['africa'] = np.intersect1d(valid, destinations.query('continent == "AF"').iata_code.dropna().unique())
    dest_dict['india']  = np.intersect1d(valid, destinations.query('iso_country == "IN"').iata_code.dropna().unique())
    dest_dict['global'] = valid
    return dest_dict


def desc_from_iata_code(data, col, prefix=''):
    '''get data from https://ourairports.com/data/'''
    df = pd.read_csv('./data/airports.csv', keep_default_na=False)
    df = df.loc[~pd.isnull(df.iata_code)]
    muni = dict(zip(df.iata_code, df.municipality))
    name = dict(zip(df.iata_code, df.name))
    region = dict(zip(df.iata_code, df.iso_region))
    continent = dict(zip(df.iata_code, df.continent.fillna('NA')))

    new_cols = {prefix + 'municipality': data[col].map(muni),
                prefix + 'name': data[col].map(name),
                prefix + 'region': data[col].map(region),
                prefix + 'continent': data[col].map(continent)}
    return data.assign(**new_cols)
    
def process(n, filename):
    line = gl(filename, n).replace('\n','').split(' ')
    return line[0], line[-1]


def load_density():
    '''Get data from Nasa's gridded population of the world 
    https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11
    '''

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
    '''Data in supplementary of: Modeling monthly flows of global air
    travel passengers: An open-accessdata resource
    '''

    filename = './data/AirportInfo.csv'
    pick_name = filename.replace('csv', 'pickle')
    if os.path.exists(pick_name) and False:
        with open(pick_name,'rb') as f:
            airports = pickle.load(f)
    else:
        airports = pd.read_csv(filename)
        g = lambda lat, lon: augment(lon=lon, lat=lat, specs=specs)
        airports = [{**row, **g(lat=row['Lat'], lon=row['Lon'])} for _, row in airports.iterrows()]
        airports = pd.DataFrame(airports)
        airports.index = airports.NodeName.astype(str)
        with open(pick_name,'wb') as f:
            pickle.dump(airports, f)

        ## Save csv
        airports[['OAGName', 'Name', 'Lon', 'Lat', 'density']].to_csv(filename.replace('data', 'tables'))

    return airports


def load_travel(airports):
    '''Data in supplementary of: Modeling monthly flows of global air
    travel passengers: An open-accessdata resource
    '''

    filename = './data/Prediction_Monthly.csv'    
    travel = pd.read_csv(filename)
    nodes = airports.NodeName.values
    travel = travel.query('Origin in @nodes and Dest in @nodes')    
    travel = travel.assign(origin_lon=travel.Origin.replace(airports.Lon),
                           origin_lat=travel.Origin.replace(airports.Lat),
                           dest_lon=travel.Dest.replace(airports.Lon),
                           dest_lat=travel.Dest.replace(airports.Lat))
    # annual = travel.groupby(['Origin', 'Dest']).mean().reset_index(drop=False).drop('Month',axis=1)
    # annual['Prediction'] = annual.Prediction.astype(int) 
    # annual['upper'] = annual.upper.astype(int)
    # annual['lower'] = annual.lower.astype(int)
    # datas = {month: data for month, data in travel.groupby('Month')}
    # datas['annual'] = annual
    # return datas
    return travel

def augment_travel(travel, airports, destinations=None, p_basal=P_BASAL):
    airport_p_no_outbreak = dict(zip(airports.index, airports.p_no_outbreak_from_one))
    if destinations is not None:
        travel = travel.query("Dest in @destinations")
    dest_p_no_outbreak_from_one = travel.Dest.replace(airport_p_no_outbreak)
    dest_p_no_outbreak = np.power(1 - p_basal * (1-dest_p_no_outbreak_from_one), travel.Prediction) ## Per origin and destination 
    travel = travel.assign(dest_p_no_outbreak=dest_p_no_outbreak.clip(0,1))
    travel['risk_ij'] = 1 - dest_p_no_outbreak
    travel['risk_i'] = 1 - travel.groupby('Origin').dest_p_no_outbreak.transform('prod')
    assert np.all(travel.risk_i.values >= 0) and np.all(travel.risk_i.values <= 1) 
    return travel
    

def calculate_p_no_outbreaks(airports, kappa):
    f = p_no_outbreak(kappa)
    if kappa == 1:
        return f(airports.R0)
    else:
        return airports.R0.apply(f)



if __name__ == '__main__':
    kappas = [1, 1, 1, 1, 1, 2]
    Rs     = [1, 2, 3, 4, 5, 2]

    print('Sanity check:')
    for R in Rs:
        p = p_no_outbreak(1)(R/3)
        p_eps = p_no_outbreak(1+1e-10)(R/3)
        assert abs(p-p_eps) < 1e-5, str(p) + '   ' + str(p_eps) 
    for kappa, R in zip(kappas, Rs):
        p = p_no_outbreak(kappa)(R)
        print(f'P(no outbreak|R0={R}, kappa={int(kappa)}) = {p:2.3f}') 

    alpha = .01
    factor = 5
    beta = 3/7
    gamma = 1/7
    for _ in range(15):
        p = p_superspreaders(alpha=alpha, factor=5)
        print(f'P(no outbreak|alpha={alpha}, factor={factor:2.3f}, beta={beta:2.3f}, gamma={gamma:2.3f}) = {p:2.3f}') 
