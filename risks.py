import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from linecache import getline as gl
import pandas as pd
from scipy.signal import convolve2d
from itertools import product
import os
import pickle
import math
import pdb

## Coordinate format: Latitude N , Longitude E.
## Longitude / East / x-axis / columns / (-180,180)
## latitude / North / y-axis / rows / (-90,90).


RADIUS = 6371
airport_list = ['JFK', 'EWR', 'LGA',
                'PUQ', 
                'FRA',
                'TLV', 'SDV', 
                'WUH',
                'DEL', 
                'CDG', 'ORY',
                'LTN', 'LHR', 'LGW',
                'TXL', 'SXF',
                'SFO', 'SJC', 'OAK',
                'MIA', 'FLL'
               ]

def deg_dist_at_lat(lat): ## Latitude in degrees!
    assert lat >= -90 and lat <= 90
    return math.cos(lat * 2 * math.pi / 360) * RADIUS * 2 * math.pi / 360


# Compare https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
# _ = [print(f'1 degree at latitutde {int(lat)} = {deg_dist_at_lat(lat)} km') for lat in(range(-90,105,15))]


def process(n, filename):
    line = gl(filename, n).replace('\n','').split(' ')
    return line[0], line[-1]


def get_coordinate_functions(specs):
    cellsize = specs['cellsize']
    xll = specs['xllcorner']
    yll = specs['yllcorner']
    nrows = specs['nrows']
    ncols = specs['ncols']

    def longitude2idx(long):
        long = np.array(long)
        #assert np.all(long >= xll)
        return ((long - xll)/cellsize).astype(int)
    def latitude2idx(lat):
        lat = np.array(lat)
        #assert np.all(lat >= yll)
        return (nrows - (lat - yll)/cellsize).astype(int)
    specs['long_f'] = longitude2idx
    specs['lat_f'] =  latitude2idx
    return specs

def row2lat(row, specs):
    nrows = specs['nrows']
    yll = specs['yllcorner']
    cellsize = specs['cellsize']
    return (nrows - row) * cellsize + yll


def augment(long, lat, specs, scan=30, avg=15): #resolution in degrees
    
    data = specs['data']
    long_f = specs['long_f']
    lat_f = specs['lat_f']
    cellsize = specs['cellsize']
    n_rows = specs['nrows']
    n_cols = specs['ncols']
    
    col = long_f(long)
    row = lat_f(lat)
    
    horz_pixel = deg_dist_at_lat(lat) * cellsize
    vert_pixel = deg_dist_at_lat(0) * cellsize
    horz_window = int(scan / horz_pixel)
    vert_window = int(scan / vert_pixel)
    row_ran = np.mod(np.arange(row-vert_window,row+vert_window+1), n_rows).astype(int)
    col_ran = np.mod(np.arange(col-horz_window,col+horz_window+1), n_cols).astype(int)
    assert len(row_ran) == 2*vert_window + 1
    assert len(col_ran) == 2*horz_window + 1
    tmp = data[row_ran, :][:, col_ran]

    shape = tmp.shape
    assert tmp.shape == (2*vert_window+1, 2*horz_window+1), f'({2*vert_window+1}, {2*horz_window+1} != {shape}'
    idx = np.unravel_index(np.argmax(tmp), tmp.shape)
    assert tmp[idx[0], idx[1]] == np.max(tmp)
    row_max = row - vert_window + idx[0]
    col_max = col - horz_window + idx[1]
    assert row_max in row_ran, f'{row_max} not in {row_ran}'
    assert col_max in col_ran, f'{col_max} not in {col_ran}'
    
    
    horz_pixel = deg_dist_at_lat(row2lat(row, specs)) * cellsize
    vert_pixel = deg_dist_at_lat(0) * cellsize
    horz_window_avg = int(avg / horz_pixel)
    vert_window_avg = int(avg / vert_pixel)
    row_ran = np.mod(np.arange(row_max-vert_window_avg,row_max+vert_window_avg+1), n_rows).astype(int)
    col_ran = np.mod(np.arange(col_max-horz_window_avg,col_max+horz_window_avg+1), n_cols).astype(int)
    tmp = data[row_ran, :][:, col_ran]
    tmp = tmp[tmp > 0]
    if len(tmp) == 0:
        density = 0
    else:
        density = np.mean(tmp)
   
    
    ret = {
        'vert_window': vert_window,
        'horz_window': horz_window, 
        'vert_window_avg': vert_window_avg,
        'horz_window_avg': horz_window_avg, 
        'row': row,
        'col': col,
        'row_max': row_max,
        'col_max': col_max,
        'row_top': row + vert_window,
        'row_bottom': row - vert_window,
        'col_left': col - horz_window,
        'col_right': col + horz_window,
        'horz_pixel': horz_pixel,
        'vert_pixel': vert_pixel,
        'density': density, 
        'idx': idx,
        'shape': shape,
    }
    pt = lambda x: [print(k, v) for k, v in x.items()]
    assert ret['row_max'] <= ret['row_top'] and ret['row_max'] >= ret['row_bottom'], pt(ret)
    assert ret['col_max'] <= ret['col_right'] and ret['col_max'] >= ret['col_left'], pt(ret)
    
    return ret

def main():
    filename = 'data/gpw_v4_population_density_rev11_2020_2pt5_min.asc'
    data = np.loadtxt(filename, skiprows=6)
    data[(data==-9999)] = 0
    specs = {k:float(v) for k,v in [process(n, filename) for n in range(1,7)]}
    specs['data'] = data


    specs = get_coordinate_functions(specs)

    airports = pd.read_csv('data/AirportInfo.csv')
    g = lambda lat, long: augment(long=long, lat=lat, specs=specs)
    airports = [{**row, **g(lat=row['Lat'], long=row['Lon'])} for _, row in airports.iterrows()]
    airports = pd.DataFrame(airports)
    airports.index = airports.NodeName.astype(str)
    vis = airports.loc[airport_list]

    plt.close('all')
    fig, ax= plt.subplots(1,1, figsize=(30,20))
    dd = np.log(1+data)
    im = ax.imshow(dd, cmap=cm.gray)
    #fig.colorbar(im, ax=ax)
    ax.scatter(vis.col_left, vis.row, label='Window', color='b')
    ax.scatter(vis.col_right, vis.row, color='b')
    ax.scatter(vis.col, vis.row_top, color='b')
    ax.scatter(vis.col, vis.row_bottom, color='b')

    ax.scatter(vis.col, vis.row, label='Airports', color='r')
    ax.scatter(vis.col_max, vis.row_max, label='Maximizers', color='g')

    for _, row in vis.iterrows():
        ax.annotate(row['NodeName'] + ' airport', (row['col']+1, row['row']), color='r', fontsize=10)
        ax.annotate('Density= ' + str(int(row['density'])) + ' ' + row['NodeName'], (row['col_max']+1, row['row_max']), color='g', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show()

    filename = 'data/Prediction_Monthly.csv'
    travel = pd.read_csv(filename)
    airport2density = dict(zip(airports.index, airports.density))
    travel = travel.assign(origin_density=travel.Origin.map(airport2density),
                           dest_density=travel.Dest.map(airport2density))
    travel = travel.query('origin_density > 0 and dest_density > 0')
    travel.dropna(inplace=True)
    outgoing = {origin + f'_{month}': data.Prediction.sum() for (origin, month), data in travel.groupby(['Origin', 'Month'])}
    travel = travel.assign(outgoing_total=(travel.Origin + '_' + travel.Month.astype(str)).replace(outgoing))
    travel = travel.assign(fraction=travel.Prediction / travel.outgoing_total)
    travel = travel.assign(risk=travel.fraction * (1 - 1/travel.dest_density))

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        pass
    except:
        import pdb, sys, traceback
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
