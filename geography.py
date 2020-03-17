import numpy as np
import math
import pdb



RADIUS = 6371

def deg_dist_at_lat(lat): ## Latitude in degrees!
    assert lat >= -90 and lat <= 90
    return math.cos(lat * 2 * math.pi / 360) * RADIUS * 2 * math.pi / 360


## Coordinate format: Latitude N , Longitude E.
## Longitude / East / x-axis / columns / (-180,180)
## latitude / North / y-axis / rows / (-90,90).

def get_coordinate_functions(specs):
    cellsize = specs['cellsize']
    xll = specs['xllcorner']
    yll = specs['yllcorner']
    nrows = specs['nrows']
    ncols = specs['ncols']

    def longitude2idx(lon):
        lon = np.array(lon)
        #assert np.all(lon >= xll)
        return ((lon - xll)/cellsize).astype(int)
    def latitude2idx(lat):
        lat = np.array(lat)
        #assert np.all(lat >= yll)
        return (nrows - (lat - yll)/cellsize).astype(int)
    specs['lon_f'] = longitude2idx
    specs['lat_f'] =  latitude2idx
    return specs


def row2lat(row, specs):
    nrows = specs['nrows']
    yll = specs['yllcorner']
    cellsize = specs['cellsize']
    return (nrows - row) * cellsize + yll


def augment(lon, lat, specs, scan=30, avg=15): #resolution in degrees
    data = specs['data']
    lon_f = specs['lon_f']
    lat_f = specs['lat_f']
    cellsize = specs['cellsize']
    n_rows = specs['nrows']
    n_cols = specs['ncols']
    
    col = lon_f(lon)
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


def augment_travel(travel, airports):
    airport_p_outbreak = dict(zip(airports.index, airports.p_outbreak))
    travel = travel.assign(origin_p_outbreak=travel.Origin.map(airport_p_outbreak),
                           dest_p_outbreak=travel.Dest.map(airport_p_outbreak))
    travel.dropna()
    travel['outgoing_total'] = travel.groupby('Origin').Prediction.transform('sum')
    travel = travel.query('outgoing_total > 0')

    travel['P_ij'] = travel.Prediction / travel.outgoing_total
    travel['risk_ij'] = travel.P_ij - travel.P_ij * travel.dest_p_outbreak
    travel['risk_i'] = travel.groupby('Origin').risk_ij.transform('sum')
    travel['origin_lon'] = travel.Origin.replace(airports.Lon)
    travel['origin_lat'] = travel.Origin.replace(airports.Lat)
    travel['dest_lon'] = travel.Dest.replace(airports.Lon)
    travel['dest_lat'] = travel.Dest.replace(airports.Lat)
    travel['red'] = travel.risk_i.clip(lower=0, upper=1)
    travel['blue'] = (1 - travel.risk_i).clip(lower=0,upper=1)
    travel['green'] = 0
    assert np.max(np.abs(travel.groupby('Origin').P_ij.sum()-1)) < 1e-9
    return travel

