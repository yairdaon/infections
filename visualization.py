import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pdb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec

import loaders


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


def plot_risks(df):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.scatter(df.origin_lon, 
               df.origin_lat,
               color = df[['red', 'green', 'blue']].values)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def plot_geodesics(df):
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.stock_img()

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)


    plt.plot(df[['origin_lon', 'dest_lon']].T,
             df[['origin_lat', 'dest_lat']].T,
             alpha=0.05, 
             color='r',
             transform=ccrs.Geodetic())
    plt.tight_layout()
    plt.show()


def show_airports(airports, density):
    vis = airports.loc[airport_list]
    dd = np.log(1+density)
    fig, ax= plt.subplots(1,1, figsize=(30,20))
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
        ax.annotate('Density= ' + str(int(row['density'])) + ' ' + row['NodeName'], (row['col_max']+1, row['row_max']), color='g', fontsize=15)
    
    plt.legend()
    plt.tight_layout()
    plt.show()



## Old code for plotting a panel of monthly infection risk.
