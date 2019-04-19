# train test time-stamps
timestamps = {'begin_date':'2013-04-01 00:00:00', 
              'end_date':'2013-04-30 23:59:59',
              'deltat':7}

# variables
variables = {'xvar': ['Fa', 
                      #'SHD', 
                      'Ta',
                      'Ws'
                      ], 
             'yvar': 'Fh',    # Driver name
             'tvar':'DateTime'
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

        'SOLO': True,
        'path2solo': '/home/atbin/workd/SOLO/source'

        }

# solvers
solvers = ['Classical Linear',
           'Random Forest',
           'eXtreme Gradient Boost',
           'Support Vector Regression',
           'ridge'
           ]

# saving results
result = {'save_plots':True,
          'save_summary':True, 
          'save_imputed':True}
