import numpy as np
import pandas as pd
import os
import pickle
import math
import pdb

import loaders
import geography
import visualization as vis


def main(debug: ("Debug mode", 'flag', 'd')):
    specs = loaders.load_density()
    specs = geography.get_coordinate_functions(specs)
    airports = loaders.load_airports(specs)
    if debug:
        airports = airports.loc[vis.airport_list]
    vis.plot_airports(airports, specs['data'])
    vis.plot_density(specs['data'])
    
    travel = loaders.load_travel(airports)
    valid = airports.NodeName.values
    travel['annual'] = travel['annual'].query('Origin in @valid and Dest in @valid')
    
    kappas = [1, 1, 2, 5]
    infected = [1, 2, 1, 1]
    for kappa, n in zip(kappas, infected):
        airports = loaders.calculate_outbreaks(airports, kappa=kappa, n=n)
        travel['annual'] = loaders.augment_travel(travel['annual'], airports)
        # travel = {time: augment_travel(df, airports) for time, df in travel.items()}
        vis.plot_risks(travel['annual'], n=n, kappa=kappa)
    
    vis.plot_geodesics(travel['annual'])

    
if __name__ == '__main__':
    try:
        import plac; plac.call(main)
    except KeyboardInterrupt as e:
        pass
    except:
        import pdb, sys, traceback
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
