import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pdb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import config
from matplotlib import gridspec
from matplotlib.colors import Normalize
from functools import partial
from scipy.optimize import fsolve
import os

import loaders


if not os.path.exists('./pix'):
    os.mkdir('./pix')
if not os.path.exists('./pix/africa'):
    os.mkdir('./pix/africa')
if not os.path.exists('./pix/india'):
    os.mkdir('./pix/india')
if not os.path.exists('./pix/global'):
    os.mkdir('./pix/global')

    
airport_list = ['JFK', 'EWR', 'LGA', #NYC
                #'PUQ', 
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
                'DEL', 
                #'CDG', 'ORY', #Paris
                'LTN', 'LHR', 'LGW', #London
                'TXL', 'SXF', #Berlin
                #'SFO', 'SJC', 'OAK', #San Francisco
                #'MIA', 'FLL' #Miami
               ]

XLIM = np.array([-180, 180])
YLIM = np.array([-90, 90])
FIG_SIZE=np.array([20, 20 * (YLIM[1]-YLIM[0]) / (XLIM[1]-XLIM[0])], dtype=int)
CB_SIZE=np.array([1,0])
TICK_FONT_SIZE=22
QUALITY=95
DPI=200


def _add_features(ax):
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    return ax

def _annotate(ax, text, color):
    ax.text(-175, -80, text, fontsize=35, color=color)
    
def plot_monthly_risks(travel, kappa=1, n=1, wuhan_R0=3, region='global'):
    def make_it(ax, df, month):
        ax.scatter(df.origin_lon, df.origin_lat, color=cm.coolwarm(df.risk_i))
        _add_features(ax)
        _annotate(ax, text=month, color='r')
       

    fig = plt.figure(figsize=FIG_SIZE)
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
    plt.savefig(f'./pix/{region}/risks_monthly_wuhan{wuhan_R0}_n{n}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_R0(df):
    wuhan_R0 = int(df.loc['WUH', 'R0'])
    cm = plt.cm.coolwarm

    figsize = FIG_SIZE + CB_SIZE
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(figsize[1], figsize[0])
    ax = fig.add_subplot(gs[:figsize[1], 0:FIG_SIZE[0]], projection=ccrs.PlateCarree())
    
    ax.scatter(df.Lon, 
               df.Lat,
               color = cm(df.R0))
    _add_features(ax)
    
    ax = fig.add_subplot(gs[:figsize[1], FIG_SIZE[0]:figsize[0]])
    ax.yaxis.set_tick_params(labelsize=20)
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm._A = []
    plt.colorbar(sm, orientation='vertical', cax=ax)
    plt.tight_layout()
    plt.savefig(f'./pix/R0_wuhan{wuhan_R0}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_annual_risks(df, kappa=1, n=1, wuhan_R0=3, region='global'):
    cm = plt.cm.coolwarm
                         
    figsize = FIG_SIZE+CB_SIZE
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(figsize[1],figsize[0])
    ax = fig.add_subplot(gs[:figsize[1], 0:FIG_SIZE[0]], projection=ccrs.PlateCarree())
    
    ax.scatter(df.origin_lon, 
               df.origin_lat,
               color = cm(df.risk_i))
    _add_features(ax)
    
    if kappa==1 and n==1 and wuhan_R0==3:
        _annotate(ax, text='b', color='r')
    
    ax = fig.add_subplot(gs[:figsize[1], FIG_SIZE[0]:figsize[0]])
    ax.yaxis.set_tick_params(labelsize=20)
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm._A = []
    plt.colorbar(sm, orientation='vertical', cax=ax)
    
    plt.tight_layout()
    plt.savefig(f'./pix/{region}/risks_annual_wuhan{wuhan_R0}_n{n}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
    
def plot_geodesics(df, destinations, region):
    print("Plotting geodesics")
    df = df.query('Dest in @destinations')
    fig = plt.figure(figsize=FIG_SIZE)
    plateCr = ccrs.PlateCarree()
    plateCr._threshold = plateCr._threshold/10.
    ax = plt.axes(projection=plateCr)
    _add_features(ax)

    
    mx = df.Prediction.max()
    cutoff = 0.005 if region == 'global' else 0.05
    opacity = np.maximum(df.Prediction.values/mx, cutoff)
    lines = plt.plot(df[['origin_lon', 'dest_lon']].T,
                     df[['origin_lat', 'dest_lat']].T, 
                     color='r',
                     transform=ccrs.Geodetic())
    [line.set_alpha(alpha) for alpha, line in zip(opacity, lines)]

    _annotate(ax, text='a', color='r')
    plt.tight_layout()
    plt.savefig(f'./pix/{region}/geodesics.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_airports(airports, density):
    print("Plotting airports")
    vis = airports.loc[airport_list]
    dd = np.log(1+density)
    fig, ax= plt.subplots(1,1, figsize=FIG_SIZE)
    im = ax.imshow(dd, cmap=cm.gray)
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

    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.contourf(lons, lats, np.log(1+density),
                 transform=ccrs.PlateCarree(), cmap=cm.gray)# cmap='coolwarm')
    ax.coastlines(color='w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
   
    ax.text(-175, -80, 'b', fontsize=35, color='r')     
    plt.tight_layout()
    plt.savefig('./pix/density.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
    

def plot_geodesics_density():
    df = df.query('Dest in @destinations')
    fig = plt.figure(figsize=FIG_SIZE)
    plateCr = ccrs.PlateCarree()
    plateCr._threshold = plateCr._threshold/10.
    ax = plt.axes(projection=plateCr)
    _add_features(ax)
    

    mx = df.Prediction.max()
    cutoff = 0.005 if region == 'global' else 0.05
    opacity = np.maximum(df.Prediction.values/mx, cutoff)
    lines = plt.plot(df[['origin_lon', 'dest_lon']].T,
                     df[['origin_lat', 'dest_lat']].T, 
                     color='r',
                     transform=ccrs.Geodetic())
    [line.set_alpha(alpha) for alpha, line in zip(opacity, lines)]
    plt.tight_layout()
    if region == 'global':
        _annotate(ax, text='a', color='r')
    plt.savefig(f'./pix/{region}/geodesics.jpg', quality=QUALITY, dpi=DPI)
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
    ## plt.suptitle("Plots of $f(p) = (1-p)( 1 + pR / \kappa)^{\kappa} - 1$", fontsize=22)
    plt.hlines(y=0, xmin=0, xmax=1)
    plt.xlabel('p')
    plt.ylabel('f')
    plt.tight_layout()
    plt.savefig('./pix/p_outbreak.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
