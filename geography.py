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
AFRICA_MAX_LAT = (37.277778, 9.863889)[0] ## Bizerte, Tunisia
AFRICA_WEST_LONG = (14.692778, -17.446667)[1] ##Dakar, Senegal
AFRICA_EAST_LONG = (10.416667, 51.266667)[1] ##Hafun, Somalia	

def get_africa_airports():
    df = pd.read_csv('./data/airports.csv')

def get_coordinate_functions(specs):
    cellsize = specs['cellsize']
    xll = specs['xllcorner']
    yll = specs['yllcorner']
    nrows = specs['nrows']
    ncols = specs['ncols']

    def longitude2idx(lon):
        lon = np.array(lon)
        return ((lon - xll)/cellsize).astype(int)

    def col2lon(col):
        return col * cellsize + xll

    def latitude2idx(lat):
        lat = np.array(lat)
        return (nrows - (lat - yll)/cellsize).astype(int)

    def row2lat(row):
        return (nrows - row) * cellsize + yll
  
    specs['lon_f'] = longitude2idx
    specs['lat_f'] = latitude2idx
    specs['row_g'] = row2lat
    specs['col_g'] = col2lon
    
    return specs


def augment(lon, lat, specs, scan=30, avg=15): #resolution in degrees
    data = specs['data']
    lon_f = specs['lon_f']
    lat_f = specs['lat_f']
    row_g = specs['row_g']
    col_g = specs['col_g']
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
    lon_max = col_g(col_max)
    lat_max = row_g(row_max)
    # assert lon_f(lon_max) == col_max, f'{lon_f(lon_max)} != {col_max}'
    # assert lat_f(lat_max) == row_max, f'{lat_f(lat_max)} != {row_max}'
    # print(f'({lon_f(lon_max)}, {col_max})', f'({lat_f(lat_max)}, {row_max})')
    
    horz_pixel = deg_dist_at_lat(specs['row_g'](row)) * cellsize
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
        'row': row,
        'col': col,
        'density': density, 
        'row_max': row_max,
        'col_max': col_max,
        'lon_max': lon_max,
        'lat_max': lat_max,
        'vert_window': vert_window,
        'horz_window': horz_window, 
        'vert_window_avg': vert_window_avg,
        'horz_window_avg': horz_window_avg, 
        'row_top': row + vert_window,
        'row_bottom': row - vert_window,
        'col_left': col - horz_window,
        'col_right': col + horz_window,
        'horz_pixel': horz_pixel,
        'vert_pixel': vert_pixel,
        'idx': idx,
        'shape': shape,
    }
    pt = lambda x: [print(k, v) for k, v in x.items()]
    assert ret['row_max'] <= ret['row_top'] and ret['row_max'] >= ret['row_bottom'], pt(ret)
    assert ret['col_max'] <= ret['col_right'] and ret['col_max'] >= ret['col_left'], pt(ret)
    
    return ret
