import pdb
from itertools import product
import numpy as np
import pandas as pd


import loaders
import geography
import visualization as vis


def main(debug: ("Debug mode", 'flag', 'd')):
    vis.plot_p_outbreak()
    print('Loading density')
    specs = loaders.load_density()

    print('calculating coordinate functions')
    specs = geography.get_coordinate_functions(specs)
    if not debug:
        vis.plot_density(specs)

    if debug:
        R0s = [4]
        regions = ['global']
        regions = ['global', 'africa', 'india']
        kappas = [1]
        infected = [1]
    
    else:
        R0s = [1, 2, 4, 6]
        regions = ['global', 'africa', 'india']
        kappas   = [1, 2, 3, 4, 5, 6]
        infected = [1, 2, 3]

    times = ['annual', 1, 4, 7, 10]

    print('loading airport data')
    airports = loaders.load_airports(specs)
    if debug:
        airports = airports.loc[vis.airport_list]

    if not debug:
        vis.plot_airports(airports, specs['data'])
        
    
    travel = loaders.load_travel(airports)
    travel = {k: travel[k] for k in times}
  
    destinations = pd.read_csv('./data/airports.csv')
    valid = airports.NodeName.values
    dest_dict = {}
    dest_dict['africa'] = np.intersect1d(valid, destinations.query('continent == "AF"').iata_code.dropna().unique())
    dest_dict['india']  = np.intersect1d(valid, destinations.query('iso_country == "IN"').iata_code.dropna().unique())
    dest_dict['global'] = valid
    

    for region in regions:
        print('Making', region, 'plots')
        dest = dest_dict[region]
        vis.plot_geodesics(travel['annual'], destinations=dest, region=region)

        for wuhan_R0 in R0s:
            wuhan_factor =  wuhan_R0 / airports.loc['WUH', 'density'] # R_0 / density for Wuhan
            airports['R0'] = airports.density * wuhan_factor
            vis.plot_R0(airports)
    
            for kappa, n in product(kappas, infected):
                airports['p_outbreak'] = loaders.calculate_outbreaks(airports, kappa=kappa, n=n)
                tmp_travel = {k: loaders.augment_travel(df, airports, destinations=dest) for k, df in travel.items()}
                vis.plot_annual_risks(tmp_travel['annual'], n=n, kappa=kappa, wuhan_R0=wuhan_R0, region=region)
                vis.plot_monthly_risks(tmp_travel, kappa=kappa, n=n, wuhan_R0=wuhan_R0, region=region)

   
    
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
