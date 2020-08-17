# train test time-stamps
timestamps = {'begin_date':'2013-03-03 00:00:00',
              'end_date':'2013-05-31 23:30:00',
              'deltat':10,
              'random_train_test_split':True}

# variables
variables = {'xvar': [#'Fa',
                      #'SHD',
                      # 'Ta',
                      # 'Ws',
                      # 'Fg',
                      # 'VPD',
                      # 'Fn',
                      # 'Fm',
                      # 'q',
                      # 'Ts',
                      # 'Sws',
                      #'EVI'
                      #'Wd',
                      #'Flu',
                      #'Fsd',
                      #'Fsu',
                      #'RH',
                      #'ps'                      
                      ], 
             'yvar': 'Fc',    # Driver name
             'tvar':'DateTime',
             'xvar_derived': [#'day', 'month', 'week', 
                              #'hour', 'minutes', 
                              #'dayofyear',
                              #'sinwt', 'coswt'
                              ]
            } 


# data
data = {'tower': 'Gingin',   # HowardSprings   AliceSpringsMulga
        'yobs_file': 'EVI',
        'file_suffix':'',
        'ancillary_files': [ ],

        'ustar':False,
        #'ustar_map':{2013:0.2211, 2018:0.23867},  #Gingin
        #'ustar_map':{2013:0.2429},   #Calperum
        #'ustar_map':{2013:0.514924879899},  #Tumbarumba
        #'ustar_map':{2013:0.262428731155},  #HowardSprings
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
        'path2solo': '.',

       'fbprophet': False
        }

# solvers
solvers = [ 'Classical Linear',
            'LSTM'
            #'Random Forest',
            #'eXtreme Gradient Boost',
            #'Support Vector Regression',
            #'ridge',
            #'ANN',
            #'Gradient Boost Regression',
            #'bayes_ridge',
            #'lasso',
            #'elasticnet'
           ]

# saving results
result = {'save_plots':False,
          'save_summary':True, 
          'save_imputed':True}
