import numpy as np
from scipy.optimize import fsolve
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import  gray, plasma 
from matplotlib import gridspec
import pdb
from functools import partial
import os
mpl.rcParams['axes.linewidth'] = 0.1 #set the value globally


from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import loaders

for lib in ['pix']:
    if not os.path.exists(f'./{lib}'):
        os.mkdir(f'./{lib}')
    if not os.path.exists(f'./{lib}/africa'):
        os.mkdir(f'./{lib}/africa')
    if not os.path.exists(f'./{lib}/india'):
        os.mkdir(f'./{lib}/india')
    if not os.path.exists(f'./{lib}/global'):
        os.mkdir(f'./{lib}/global')
if not os.path.exists(f'./tables'):
    os.mkdir(f'./tables')

################ Constants #############################    
airport_list = ['JFK', 'EWR', 'LGA', #NYC
                #'PUQ', ## Shithole in south Chile
                #'HNL', # Hawaii
                'NRT', 'HND',# Tokyo
                #'FRA', #Frankfurt
                'CPT', # Capetown
                'KEF', 'RKV', # Reykjavik Iceland
                'EZE', ## Buenos aires
                'SCL', ## Santiago Chile
                #'TLV', 'SDV', 
                'WUH', #Wuhan
                'ICN', # Seoul
                'DEL', # New Delhi
                #'CDG', 'ORY', #Paris
                'LTN', 'LHR', 'LGW', #London
                'TXL', 'SXF', #Berlin
                #'SFO', 'SJC', 'OAK', #San Francisco
                #'MIA', 'FLL' #Miami
               ]
XLIMS={'global':[-180,180],
       'africa':[-100,160],
       'india':[-90,145]}
YLIM=[-90+36,90-6]
MAP_SIZE = np.array([20, 10])
H_CB_SIZE=(20,2)
V_CB_SIZE=(2,20)
TICK_FONT_SIZE=22
QUALITY=95
DPI=200
DOT_SIZE=30
VMAX = {'global': 0.75, 'africa': 0.1, 'india': 0.5}


def _add_features(ax, region='global'):
    ax.set_ylim(*YLIM)
    ax.set_xlim(*XLIMS[region])
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def _annotate(ax, text, color, region):
    xlim = XLIMS[region]
    ax.text(xlim[0]+5, YLIM[0]+5, text, fontsize=50, color='k')
    
    
def plot_monthly_risks(travel, kappa=1, wuhan_R0=3, region='global'):
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX[region], clip=True)
    def make_it(ax, df, month):
        
        ax.scatter(df.origin_lon,
                   df.origin_lat,
                   color=plasma(norm(df.risk_i)),
                   s=DOT_SIZE)
        _add_features(ax)
        _annotate(ax, text=month, color='k', region='global')
       

    fig = plt.figure(figsize=MAP_SIZE)
    gs = fig.add_gridspec(2, 8)

    ax1 = fig.add_subplot(gs[0, 0:4], projection=ccrs.PlateCarree())
    df = travel[1]
    make_it(ax1, df, 'January')

    ax2 = fig.add_subplot(gs[0, 4:8], projection=ccrs.PlateCarree())
    df = travel[4]
    make_it(ax2, df, 'April')

    ax3 = fig.add_subplot(gs[1:, 0:4], projection=ccrs.PlateCarree())
    df = travel[7]
    make_it(ax3, df, 'July')

    ax4 = fig.add_subplot(gs[1, 4:8], projection=ccrs.PlateCarree())
    df = travel[10]
    make_it(ax4, df, 'October')

    plt.tight_layout()
    plt.savefig(f'./pix/{region}/risks_monthly_wuhan{wuhan_R0}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_airport_risks(df, wuhan_R0=3, kappa=1):
    fig = plt.figure(figsize=MAP_SIZE)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5, clip=True)
    ax.scatter(df.Lon, 
               df.Lat,
               color = plasma(norm(df.p_outbreak_from_one)),
               s=DOT_SIZE)
    _add_features(ax)
    plt.tight_layout()
    plt.savefig(f'./pix/airports_p_outbreak_from_one_wuhan{wuhan_R0}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
    fig = plt.figure(figsize=H_CB_SIZE)
    ax = fig.add_subplot()
    ax.yaxis.set_tick_params(labelsize=20)    
    sm = plt.cm.ScalarMappable(cmap=plasma, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, orientation='horizontal', cax=ax)
    cbar.ax.tick_params(labelsize=35) 
    plt.tight_layout()
    plt.savefig(f'./pix/airports_p_outbreak_from_one_wuhan{wuhan_R0}_kappa{kappa}_cb.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
    
        
def plot_rep_risks(df, kappa=1, wuhan_R0=3, region='global'):
    fig = plt.figure(figsize=MAP_SIZE)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX[region], clip=True)
    ax.scatter(df.origin_lon, 
               df.origin_lat,
               color = plasma(norm(df.risk_i)),
               s=DOT_SIZE)
    _add_features(ax, region=region)
    plt.tight_layout()

    plt.savefig(f'./pix/{region}/risks_rep_wuhan{wuhan_R0}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    if kappa==1 and wuhan_R0==3:
        _annotate(ax, text='b', color='k', region=region)
        plt.savefig(f'./pix/{region}/risks_rep_wuhan{wuhan_R0}_kappa{kappa}_main.jpg', quality=QUALITY, dpi=DPI)
            
    plt.close('all')

    
def plot_cb(orientation='horizontal', region='global'):
    if orientation == 'horizontal':
        fig = plt.figure(figsize=H_CB_SIZE)
    else:
        fig = plt.figure(figsize=V_CB_SIZE)
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX[region], clip=True)
    ax = fig.add_subplot(1, 1, 1)
    sm = plt.cm.ScalarMappable(cmap=plasma, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, orientation=orientation, cax=ax)
    cbar.ax.tick_params(labelsize=35) 
    plt.tight_layout()
    plt.savefig(f'./pix/cb_{region}_{orientation}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_geodesics(df, destinations, region):
    print("Plotting geodesics")
    df = df.query('Dest in @destinations')
    fig = plt.figure(figsize=MAP_SIZE)
    plateCr = ccrs.PlateCarree()
    plateCr._threshold = plateCr._threshold/10.
    ax = plt.axes(projection=plateCr)
    
    #mx = df.Prediction.max()
    if region == 'global':
        cutoff = 0.005
    else:
        cutoff = 0.1
    opacity = np.maximum(df.risk_ij, cutoff)
    lines = plt.plot(df[['origin_lon', 'dest_lon']].T,
                     df[['origin_lat', 'dest_lat']].T, 
                     color='r',
                     transform=ccrs.Geodetic())
    [line.set_alpha(alpha) for alpha, line in zip(opacity, lines)]
    
    _annotate(ax, text='a', color='k', region=region)
    _add_features(ax, region=region)

    plt.tight_layout()
    plt.savefig(f'./pix/{region}/geodesics.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_airports(airports, density):
    print("Plotting airports")
    vis = airports.loc[airport_list]
    dd = np.log(1+density)
    fig, ax= plt.subplots(1,1, figsize=MAP_SIZE)
    im = ax.imshow(dd, cmap=gray)
    ax.scatter(vis.col_left, vis.row, label='Window', color='b')
    ax.scatter(vis.col_right, vis.row, color='b')
    ax.scatter(vis.col, vis.row_top, color='b')
    ax.scatter(vis.col, vis.row_bottom, color='b')    
    ax.scatter(vis.col, vis.row, label='Airports', color='r')
    ax.scatter(vis.col_max, vis.row_max, label='Maximizers', color='g')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    
    for _, row in vis.iterrows():
        ax.annotate(row['NodeName'], (row['col']+1, row['row']), color='r', fontsize=15)
        ax.annotate('Density= ' + str(int(row['density'])), (row['col_max']+1, row['row_max']), color='g', fontsize=15)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('./pix/airports.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
        
    
def plot_density(specs):
    print("Plotting density")
    rows, cols = specs['data'].shape
    cols, rows = np.meshgrid(range(cols), range(rows))
    lats = specs['row_g'](rows)
    lons = specs['col_g'](cols)
    density = specs['data']

    fig = plt.figure(figsize=MAP_SIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.contourf(lons, lats, np.log(1+density),
                 transform=ccrs.PlateCarree(), cmap=gray)
    ax.coastlines(color='w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
   
    #ax.text(-175, -80, 'b', fontsize=35, color='r')     
    plt.tight_layout()
    plt.savefig('./pix/density.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
        

    
def plot_p_outbreak(n=5):
    print("Plotting probability of outbreak function")
    p = np.linspace(0, 1, num=10**5)
    kappas = np.random.normal(loc=2.5, scale=0.5, size=n)
    Rs = np.random.normal(loc=4, scale=1, size=n)
    kappas[0] = 1
    Rs[0] = 4

    for kappa, R in zip(kappas, Rs):
        g = partial(loaders.f, kappa=kappa, R=R)
        root = fsolve(g, x0=np.array(0.99))[0]
        plt.plot(p, g(p), label='kappa={:2.1f}, R={:2.1f}, p={:2.3f}'.format(kappa, R, root))
        plt.scatter(root, 0)
    
    plt.legend()
    
    plt.hlines(y=0, xmin=0, xmax=1)
    plt.xlabel('p')
    plt.ylabel(r'$f(p) = (1-p)( 1 + pR / \kappa)^{\kappa} - 1$"')
    plt.tight_layout()
    plt.savefig('./pix/p_outbreak.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
