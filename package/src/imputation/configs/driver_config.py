# train test time-stamps
timestamps = {'begin_date':'2013-01-01 00:00:00',
              'end_date':'2013-12-31 23:59:59',
              'deltat':300}

# variables
variables = {'xvar': ['ACCESS_Ah_01',
                      'AWS_Ah_1',
                      'BIOS2_Ah',
                      #'Swsi(day)',
                      # "Tai(day)",
                      # "RHi(day)", 
                      # "Wsi(day)", 
                      # 'AWS_Precip_1', 
                      # 'BIOS2_Fn', 
                      # 'ACCESS_Ps_01'
                      #'sinwt',
                      #'coswt',
                      #'day', 'month', 'week', 'hour', 'minutes', 'dayofyear'
                      ], 
             'yvar': 'Ah',    # Driver name
             'tvar':'DateTime',
             'xvar_derived': []#['day', 'month', 'week', 'hour', 'minutes', 'dayofyear']
            } 

# data
data = {'tower': 'Calperum', #'TiTreeEast', #'AliceSpringsMulga', #
        'yobs_file': 'L3',
        'ancillary_files': ['ACCESS', 'BIOS2', 'AWS'], # 'AWS', 'BIOS2'

        'ustar':False,
        'ustar_map':{2013:0.23},
        'ustar_exclude_hour':{'begin':18, 'end':7},

        'Climatology': False,
        'Climatology_xvars':['Swsi(day)'
                             #"Ahi(day)" 
                             #"RHi(day)", 
                             #"Wsi(day)",
                             #"Tai(day)"
                            ],
        'PanelData':False,
        'second_tower':'Gingin', #'TiTreeEast',

        'SOLO': False,
        'path2solo': '/home/atbin/workd/SOLO/source',
        
        'fbprophet': False
       }

# solvers
solvers = ['Classical Linear',
           'Random Forest',
           #'eXtreme Gradient Boost',
           'Support Vector Regression',
           'ridge'
           ]

# saving results
result = {'save_plots':True,
          'save_summary':True, 
          'save_imputed':True}
