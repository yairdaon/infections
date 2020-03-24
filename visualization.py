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

import loaders

airport_list = ['JFK', 'EWR', 'LGA', #NYC
                #'PUQ', 
                'FRA', #Frankfurt
                #'TLV', 'SDV', 
                'WUH', #Wuhan
                'ICN', # Seoul
                #'DEL', 
                #'CDG', 'ORY', #Paris
                'LTN', 'LHR', 'LGW', #London
                'TXL', 'SXF', #Berlin
                'SFO', 'SJC', 'OAK', #San Francisco
                #'MIA', 'FLL' #Miami
               ]
FIG_SIZE = (30,20)
TICK_FONT_SIZE = 16


def plot_R0(df):
    print('Plotting R0')
    cm = plt.cm.coolwarm
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.scatter(df.Lon, 
               df.Lat,
               color = cm(df.R0))
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    norm = Normalize(vmin=0, vmax=df.R0.max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cb = plt.colorbar(sm, orientation='horizontal')
    cb.ax.tick_params(labelsize=TICK_FONT_SIZE)

    title = f"Basic Reproductive Number $R_0$"
    plt.suptitle(title, fontsize=33)
    plt.tight_layout()
    plt.savefig(f'./pix/R0.png')
    plt.close('all')

    
    
def plot_risks(df, kappa=1, n=1):
    print("Plotting risks")
    # cm = plt.cm.plasma
    cm = plt.cm.coolwarm
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.scatter(df.origin_lon, 
               df.origin_lat,
               color = cm(df.risk_i))
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    sm = plt.cm.ScalarMappable(cmap=cm)
    sm._A = []
    cb = plt.colorbar(sm, orientation='horizontal')
    title = f"Risk of Outbreak, Introducing {n} Infected Individual(s) ($\kappa={kappa}$)"
    plt.suptitle(title, fontsize=33)
    plt.tight_layout()
    plt.savefig(f'./pix/risks_n{n}_kappa{kappa}.png')
    plt.close('all')

    df = df.query('Origin == "WUH"')
    if len(df) > 0:
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([40, 150, -20, 85], crs=ccrs.PlateCarree())
        ax.scatter(df.dest_lon, 
                   df.dest_lat,
                   color = cm(df.risk_ij))
        plt.scatter(df.origin_lon.iloc[0],
                    df.origin_lat.iloc[0],
                    color='k')
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        sm = plt.cm.ScalarMappable(cmap=cm)
        sm._A = []
        cb = plt.colorbar(sm, orientation='horizontal')
        title = f"Risk of Outbreak, Introducing {n} Infected Individual(s) at WUH ($\kappa={kappa}$)"
        plt.suptitle(title, fontsize=33)
        plt.tight_layout()
        plt.savefig(f'./pix/wuhan_risks_n{n}_kappa{kappa}.png')
        plt.close('all')

    
def plot_geodesics(df):
    print("Plotting geodesics")
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.stock_img()

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    mx = df.Prediction.max()
    df = df.assign(opacity=np.maximum(df.Prediction.values, 1000*np.ones(len(df)))/mx)
    lines = plt.plot(df[['origin_lon', 'dest_lon']].T,
                     df[['origin_lat', 'dest_lat']].T, 
                     color='r',
                     transform=ccrs.Geodetic())
    [line.set_alpha(alpha) for alpha, line in zip(df.opacity, lines)]

    plt.tight_layout()
    plt.suptitle('Weighted flight connectivity', fontsize=33)
    plt.savefig('./pix/geodesics.png')
    plt.close('all')

    
def plot_airports(airports, density):
    print("Plotting airports")
    vis = airports.loc[airport_list]
    dd = np.log(1+density)
    fig, ax= plt.subplots(1,1, figsize=FIG_SIZE)
    im = ax.imshow(dd, cmap=cm.gray)
    #fig.colorbar(im, ax=ax)
    ax.scatter(vis.col_left, vis.row, label='Window', color='b')
    ax.scatter(vis.col_right, vis.row, color='b')
    ax.scatter(vis.col, vis.row_top, color='b')
    ax.scatter(vis.col, vis.row_bottom, color='b')
    
    ax.scatter(vis.col, vis.row, label='Airports', color='r')
    ax.scatter(vis.col_max, vis.row_max, label='Maximizers', color='g')

    for _, row in vis.iterrows():
        ax.annotate(row['NodeName'] + ' airport', (row['col']+1, row['row']), color='r', fontsize=15)
        ax.annotate('Density= ' + str(int(row['density'])), (row['col_max']+1, row['row_max']), color='g', fontsize=15)
    
    plt.legend()
    plt.tight_layout()
    plt.suptitle("World Density and Selected Airports with Associated Population Centers", fontsize=33)
    plt.savefig('./pix/airports.png')
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

    plt.suptitle("log(1 + Density)", fontsize=33)
    plt.tight_layout()
    plt.savefig('./pix/density.png')
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
    plt.suptitle("Plots of $f(p) = (1-p)( 1 + pR / \kappa)^{\kappa} - 1$", fontsize=22)
    plt.hlines(y=0, xmin=0, xmax=1)
    plt.xlabel('p')
    plt.ylabel('f')
    plt.tight_layout()
    plt.savefig('./pix/p_outbreak.png')
    plt.close('all')
