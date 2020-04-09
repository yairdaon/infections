import pdb
from itertools import product
import numpy as np
import pandas as pd
from unittest.mock import Mock

import loaders
import geography
import visualization as vis


def main(debug: ("Debug mode", 'flag', 'd'),
         skip_figs: ("Skip making figures", 'flag', 's')):

    # if skip_figs:
    #     vis = Mock()

    if debug:
        vis.DPI = 50
        vis.QUALITY = 75

    vis.plot_p_outbreak()
    
    print('Loading density')
    specs = loaders.load_density()

    print('Calculating coordinate functions')
    specs = geography.get_coordinate_functions(specs)
    if not debug:
        vis.plot_density(specs)

    if debug:
        R0s = [3]
        regions = ['global', 'africa', 'india']
        kappas = [1]
    
    else:
        R0s = [2, 3, 4]
        regions = ['global', 'africa', 'india']
        kappas   = [1, 3, 6]


    print('loading airport data')
    airports = loaders.load_airports(specs)
    if debug:
        airports = airports.loc[vis.airport_list]

    vis.plot_airports(airports, specs['data'])        
    
    travel = loaders.load_travel(airports)
    months = [1, 4, 7, 10]
    travel = {month: travel.query('Month == @month') for month in months}

    ## Get vailid destinations for every region
    valid = airports.NodeName.values
    dest_dict = loaders.get_destinations_dict(valid)

    
    for wuhan_R0 in R0s:
        wuhan_factor =  wuhan_R0 / airports.loc['WUH', 'density'] # R_0 / density for Wuhan
        airports['R0'] = airports.density * wuhan_factor

        for kappa in kappas:
            airports['p_no_outbreak_from_one'] = loaders.calculate_p_no_outbreaks(airports, kappa=kappa)
            airports['p_outbreak_from_one'] = 1 - airports.p_no_outbreak_from_one
                    
            if wuhan_R0 == 3 and kappa == 1:
                vis.plot_airport_risks(airports, wuhan_R0=wuhan_R0, kappa=kappa)
                cols = ['OAGName', 'Name', 'City', 'density', 'R0', 'p_outbreak_from_one']
                filename = f'./tables/TableS2.csv'
                airports[cols].sort_values('p_outbreak_from_one', ascending=False).to_csv(filename)

            for region in regions:
                dest = dest_dict[region]
                tmp_travel = {k: loaders.augment_travel(df, airports, destinations=dest) for k, df in travel.items()}
                vis.plot_rep_risks(tmp_travel[10], kappa=kappa, wuhan_R0=wuhan_R0, region=region)
                vis.plot_monthly_risks(tmp_travel, kappa=kappa, wuhan_R0=wuhan_R0, region=region)

                if kappa == 1 and wuhan_R0 == 3:
                    vis.plot_geodesics(tmp_travel[10], destinations=dest, region=region) 
                    vis.plot_cb(orientation='horizontal', region=region)
                    vis.plot_cb(orientation='vertical', region=region)
                    
                    df = tmp_travel[10][['Origin', 'risk_i']].drop_duplicates()
                    df = df.sort_values('risk_i', ascending=False)
                    df = loaders.desc_from_iata_code(df, 'Origin').set_index('Origin', drop=True)
                    df.to_csv(f'./tables/TableS1.csv')

                    if region == 'global':
                        with open('./tables/percentiles.txt', 'a') as f:
                            print(f'Average risk for airports by continent of flight origin:', file=f)
                            g = lambda x: x.sort_values().iloc[-10:].mean() ## Average on 10 largest elements
                            print(df.groupby('continent').risk_i.apply(g).sort_values(ascending=False), file=f)

                        df = tmp_travel[10].query("Origin == 'WUH'")
                        df = df.drop(['lower', 'upper', 'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat', 'Origin', 'risk_i'], axis=1)
                        df = df.sort_values('risk_ij', ascending=False)
                        df = loaders.desc_from_iata_code(df, 'Dest').set_index('Dest', drop=True)
                        df.to_csv(f'./tables/risks_from_wuhan_wuhan{wuhan_R0}_kappa{kappa}.csv')

                        
                        df = tmp_travel[10]
                        df = loaders.desc_from_iata_code(df, 'Dest', prefix='dest_')
                        df = df.sort_values('risk_ij', ascending=False).reset_index(drop=True)
                        df.to_csv(f'./tables/TableS3.csv')
                        
                                                        
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
