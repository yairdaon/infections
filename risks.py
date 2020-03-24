import pdb

import loaders
import geography
import visualization as vis


def main(debug: ("Debug mode", 'flag', 'd')):
    # vis.plot_p_outbreak()
    print('Loading density')
    specs = loaders.load_density()
    print('calculating coordinate functions')
    specs = geography.get_coordinate_functions(specs)
    vis.plot_density(specs)
    print('loading airport data')
    airports = loaders.load_airports(specs)
    if debug:
        airports = airports.loc[vis.airport_list]
    vis.plot_airports(airports, specs['data'])
    vis.plot_R0(airports)

    travel = loaders.load_travel(airports)
    valid = airports.NodeName.values
    travel['annual'] = travel['annual'].query('Origin in @valid and Dest in @valid')
    
    kappas   = [1, 1, 2, 3, 4, 5, 6]
    infected = [1, 2, 1, 1, 1, 1, 1]
    if debug:
        kappas = kappas[:1]
        infectved = infected[:1]
    for kappa, n in zip(kappas, infected):
        airports = loaders.calculate_outbreaks(airports, kappa=kappa, n=n)
        for time, df in travel.items():
            if time not in ['annual']:
                continue
            tmp = loaders.augment_travel(travel['annual'], airports)
            vis.plot_risks(tmp, n=n, kappa=kappa)
     
        
    vis.plot_geodesics(tmp)
   
    
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
    
