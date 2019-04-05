# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 08:42:33 2018

@author: 22372102
"""

from sklearn.svm import SVR 
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import xgboost
from sklearn.model_selection import GridSearchCV

def model(solver):
    
    # Linear Models 
    if solver=='Classical Linear':
        parameters = {'fit_intercept':[True], 'normalize':[True]}
        scale_data = False
        model = linear_model.LinearRegression()

    elif solver=='eXtreme Gradient Boost':
        parameters = {'max_depth':[1,4,8], 'learning_rate':[0.001,0.01,0.1], 
                      'subsample':[0.2,0.5,0.8], 'reg_alpha':[0.1,0.01,0.001]}
        scale_data = False
        model = xgboost.XGBRegressor(n_estimators=1500, objective='reg:linear',
                                     )

    elif solver == 'Support Vector Regression':
        parameters = {'C':[1, 10, 1e4], 
                      'gamma':[0.001, 0.01, 0.1]}
        scale_data = True
        model = SVR(kernel='rbf')
        
    elif solver =='Gradient Boost Regression':
        scale_data = False
        parameters = {'learning_rate':[0.01],
                      'n_estimators':[150], 
                      'max_depth':[5]}
        model = GBR()

    elif solver=='Random Forest':
        scale_data = False
        parameters = {'max_depth':[5, 10, 15], 'n_estimators':[100, 500, 1000]}
        model = RandomForestRegressor(random_state= 42)
        
    elif solver=='ridge':
        
        parameters = {'fit_intercept':[True, False], 'alpha':[0.01, 0.1, 1.0, 10.0], 'normalize':[True, False],
                     'max_iter':[5000]}
        scale_data = False
        model = linear_model.Ridge()
            
    else:
        raise ValueError('Unknown Solver Name.')
        
    scores = None #['r2']
    model = GridSearchCV(model, parameters, cv=2, scoring=scores,# n_jobs=-1,
                         verbose=1)
    
    return model, scale_data