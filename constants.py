import numpy as np
airport_list = ['JFK', 'EWR', 'LGA', #NYC
                #'PUQ', ## Shithole in south Chile
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
                'DEL', # New Delhi
                #'CDG', 'ORY', #Paris
                'LTN', 'LHR', 'LGW', #London
                'TXL', 'SXF', #Berlin
                #'SFO', 'SJC', 'OAK', #San Francisco
                #'MIA', 'FLL' #Miami
               ]

XLIMS={'global': [-180,180], #OK
       'africa': [-100,160], #OK
       'india': [-20,145], #OK
       'usa': [-180,180], #OK
       'brazil': [-120,50], #OK
       'russia': [-20,180], #OK
       'europe': [-180,180], #OK
       'china': [-180,180]
}

ALL_REGIONS = XLIMS.keys()

YLIM=[-90+36,90-6]
MAP_SIZE = np.array([20, 10])
H_CB_SIZE=(20,2)
V_CB_SIZE=(2,20)
TICK_FONT_SIZE=22
QUALITY=95
DPI=200
DOT_SIZE=30
VMAX = {'global': 0.75,
        'africa': 0.1,
        'india': 0.5,
        'brazil': 0.15,
        'europe': 0.21,
        'russia': 0.07,
        'usa': 0.15,
        'china': 0.67}
