# train test time-stamps
timestamps = {'begin_date':'2013-01-01 00:00:00',
              'end_date':'2013-12-31 23:59:59',
              'deltat':300}

# variables
variables = {'xvar': [#'Fa',
                      #'SHD', 
                      'Ta',
                      #'Ws'
                      'Fg',
                      'VPD',
                      'Fn',
                      'q',
                      'Ts'
                      ], 
             'yvar': 'Fc',    # Driver name
             'tvar':'DateTime',
             'xvar_derived': ['day', 'month', 'week', 'hour', 'minutes', 'dayofyear',
                              #'sinwt', 'coswt'
                              ]
            } 

# data
data = {'tower': 'Calperum', 
        'yobs_file': 'L4',
        'ancillary_files': [ ],

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
            #'Random Forest',
            #'eXtreme Gradient Boost',
            #'Support Vector Regression',
            #'ridge'

           ]

# saving results
result = {'save_plots':True,
          'save_summary':True, 
          'save_imputed':True}