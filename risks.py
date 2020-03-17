import numpy as np
import pandas as pd
import os
import pickle
import math
import pdb

import loaders
import geography
import visualization as vis


def main():
    specs = loaders.load_density()
    specs = geography.get_coordinate_functions(specs)
    airports = loaders.load_airports(specs)
    # vis.show_airports(airports, specs['data'])

    travel = loaders.load_travel(airports)
    travel = geography.augment_travel(travel['annual'], airports)
    #travel = {time: augment_travel(df, airports) for time, df in travel.items()}
    vis.plot_geodesics(travel)
    vis.plot_risks(travel)
    
    
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
    
