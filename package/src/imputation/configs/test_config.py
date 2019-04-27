# train test time-stamps
timestamps = {'begin_date':'2013-07-01 00:00:00', 
              'end_date':'2013-08-01 23:59:59',
              'deltat':10}

# variables
variables = {'xvar': ['ACCESS_Ts_01',
                      #'AWS_Wd_0', 
                      'BIOS2_Ts',
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
             'yvar': 'Ts',    # Driver name
             'tvar':'DateTime'
            } 

# data
data = {'tower': 'Calperum', #'TiTreeEast', #'AliceSpringsMulga', #
        'yobs_file': 'L3',
        'ancillary_files': ['ACCESS', 'BIOS2'], # 'AWS', 'BIOS2'

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
solvers = [#'Classical Linear',
           'Random Forest',
           #'eXtreme Gradient Boost',
           #'Support Vector Regression',
           #'ridge'
           ]

# saving results
result = {'save_plots':True,
          'save_summary':True, 
          'save_imputed':True}
