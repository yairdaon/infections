import pdb
from itertools import product

import loaders
import geography
import visualization as vis


def main(debug: ("Debug mode", 'flag', 'd')):
    vis.plot_p_outbreak()
    print('Loading density')
    specs = loaders.load_density()

    print('calculating coordinate functions')
    specs = geography.get_coordinate_functions(specs)
    vis.plot_density(specs)

    if debug:
        R0s = [4]
    else:
        R0s = [1, 2, 4, 6]
    for wuhan_R0 in R0s:
        
        print('loading airport data')
        airports = loaders.load_airports(specs, wuhan_R0=wuhan_R0)
        if debug:
            airports = airports.loc[vis.airport_list]
        vis.plot_airports(airports, specs['data'])
        vis.plot_R0(airports)

        travel = loaders.load_travel(airports)
        travel = {k: travel[k] for k in ['annual', 1, 4, 7, 10]}
        
        kappas   = [1, 2, 3, 4, 5, 6]
        infected = [1, 2, 3]
        if debug:
            kappas = kappas[:1]
            infectved = infected[:1]
        for kappa, n in product(kappas, infected):
            airports = loaders.calculate_outbreaks(airports, kappa=kappa, n=n)
            tmp_travel = {k: loaders.augment_travel(df, airports) for k, df in travel.items()}
            vis.plot_annual_risks(tmp_travel['annual'], n=n, kappa=kappa, wuhan_R0=wuhan_R0)
            vis.plot_monthly_risks(tmp_travel, kappa=kappa, n=n, wuhan_R0=wuhan_R0)

    vis.plot_geodesics(tmp_travel['annual'])
   
    
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
