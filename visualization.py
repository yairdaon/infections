import numpy as np
from scipy.optimize import fsolve
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import  gray
from matplotlib.cm import plasma_r as cmap  
from matplotlib import gridspec
import pdb
from functools import partial
import os
mpl.rcParams['axes.linewidth'] = 0.05 #set the value globally

from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import loaders
from constants import *
import geography


for lib in ['pix']:
    if not os.path.exists(f'./{lib}'):
        os.mkdir(f'./{lib}')
    for region in XLIMS.keys():
        path = f'./{lib}/{region}'
        if not os.path.exists(path):
            os.mkdir(path)

            
if not os.path.exists(f'./tables'):
    os.mkdir(f'./tables')

opacity_scaler = lambda x, s: np.maximum(np.arctan(1/(1-x)**s - 1/x**s) / np.pi + 0.5, 0.005)


def _add_features(ax, region='global'):
    ax.set_ylim(*YLIM)
    ax.set_xlim(*XLIMS[region])
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def _annotate(ax, text, color, region):
    xlim = XLIMS[region]
    ax.text(xlim[0]+5, YLIM[0]+5, text, fontsize=50, color='k')
    
    
def plot_monthly_risks(travel, kappa=1, wuhan_R0=3, region='global'):
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX.get(region, 0.75), clip=True)
    def make_it(ax, df, month):
        
        ax.scatter(df.origin_lon,
                   df.origin_lat,
                   color=cmap(norm(df.risk_i)),
                   s=(DOT_SIZE*norm(df.risk_i)**2))
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
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    ax.scatter(df.Lon, 
               df.Lat,
               color = cmap(norm(df.p_outbreak_from_one)),
               s=(DOT_SIZE*df.p_outbreak_from_one)**2)
    _add_features(ax)
    plt.tight_layout()
    plt.savefig(f'./pix/airports_p_outbreak_from_one_wuhan{wuhan_R0}_kappa{kappa}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
    fig = plt.figure(figsize=H_CB_SIZE)
    ax = fig.add_subplot()
    ax.yaxis.set_tick_params(labelsize=20)    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, orientation='horizontal', cax=ax)
    cbar.ax.tick_params(labelsize=35) 
    plt.tight_layout()
    plt.savefig(f'./pix/airports_p_outbreak_from_one_wuhan{wuhan_R0}_kappa{kappa}_cb.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')
    
        
def plot_rep_risks(df, kappa=1, wuhan_R0=3, region='global'):
    fig = plt.figure(figsize=MAP_SIZE)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX.get(region, 0.75), clip=True)
    ax.scatter(df.origin_lon, 
               df.origin_lat,
               color = cmap(norm(df.risk_i)),
               s=(DOT_SIZE*norm(df.risk_i))**2)
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
    norm = mpl.colors.Normalize(vmin=0, vmax=VMAX.get(region, 0.75), clip=True)
    ax = fig.add_subplot(1, 1, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, orientation=orientation, cax=ax)
    cbar.ax.tick_params(labelsize=35) 
    plt.tight_layout()
    plt.savefig(f'./pix/cb_{orientation}_{region}.jpg', quality=QUALITY, dpi=DPI)
    plt.close('all')

    
def plot_geodesics(df, destinations, region, opacity_s=0.7):
    print("Plotting geodesics")
    df = df.query('Dest in @destinations')
    fig = plt.figure(figsize=MAP_SIZE)
    plateCr = ccrs.PlateCarree()
    plateCr._threshold = plateCr._threshold/10.
    ax = plt.axes(projection=plateCr)

    ## Add FSI
    cm = plt.cm.winter
    shp = shpreader.natural_earth(resolution='10m',category='cultural',
                                  name='admin_0_countries')
    reader = shpreader.Reader(shp)
    for n in reader.records():
        
        fsi, _ = geography.get_fsi(n, FSI_DF)
        
        if fsi is None:
            continue

        fsi = min(max(fsi-20, 0), 100)
        if fsi < 25:
            clr = cm(0)#'green'
        elif fsi >= 25 and fsi < 50:
            clr = cm(0.33)#'purple'
        elif fsi >= 50 and fsi < 75:
            clr = cm(0.66)#'blue'
        else:
            clr = cm(1)#'red'
        try:
            ax.add_geometries(n.geometry, ccrs.PlateCarree(), facecolor=clr, 
                              alpha = 0.5, linewidth =0.15, edgecolor = "black",
                              label=n.attributes['ADM0_A3'])
        except:
            ax.add_geometries([n.geometry], ccrs.PlateCarree(), facecolor=clr, 
                              alpha = 0.5, linewidth =0.15, edgecolor = "black",
                              label=n.attributes['ADM0_A3'])


    if region == 'global':
        cutoff = 0.005
    else:
        cutoff = 0.1
    opacity = np.maximum(df.risk_ij, cutoff)
    lines = plt.plot(df[['origin_lon', 'dest_lon']].T,
                     df[['origin_lat', 'dest_lat']].T, 
                     color='r',
                     transform=ccrs.Geodetic())
    [line.set_alpha(opacity_scaler(alpha, opacity_s)) for alpha, line in zip(opacity, lines)]
    
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
