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
from sklearn import neural_network as nn
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

def model(solver):
    
    # Linear Models 
    if solver=='clr':
        parameters = {'fit_intercept':[True, False], 'normalize':[True, False]}
        model = linear_model.LinearRegression()

    elif solver=='lr_ransac':
        parameters = {}
        model = linear_model.RANSACRegressor()
    elif solver=='lr_theil':
        parameters = {}
        model = linear_model.TheilSenRegressor()
    elif solver=='lr_huber':
        parameters = {}
        model = linear_model.HuberRegressor()       
        
    elif solver=='ridge':
        parameters = {'fit_intercept':[True, False], 'alpha':[0.01, 0.1, 1.0, 10.0], 'normalize':[True, False],
                     'max_iter':[5000]}
        model = linear_model.Ridge()
    
    elif solver=='bayes_ridge':
        parameters = {'fit_intercept':[True, False], 
                      'alpha_1':[1e-3, 1e-6], 'alpha_2':[1e-3, 1e-6], 
                      'lambda_1':[1e-3, 1e-6], 'lambda_2':[1e-3, 1e-6], 
                      'normalize':[True, False],
                      'n_iter':[500,5000]}
        model = linear_model.BayesianRidge()

    elif solver=='lasso':
        parameters = {'fit_intercept':[True, False], 'alpha':[0.01, 0.1, 1.0, 10.0], 'normalize':[True, False],
                     'max_iter':[5000]}
        model = linear_model.Lasso()
        
    elif solver=='elasticnet':
        parameters = {'fit_intercept':[True, False], 'alpha':[0.01, 0.1, 1.0, 10.0], 'normalize':[True, False],
                     'max_iter':[5000]}
        model = linear_model.ElasticNet()
        
    elif solver=='xgb':
        parameters = {'max_depth':[1,2,4,6,8], 'learning_rate':[0.001,0.01,0.1], 
                      'subsample':[0.2,0.4,0.6,0.8], 'reg_alpha':[0.1,0.01,0.001]}
        model = xgboost.XGBRegressor(n_estimators=1500, objective='reg:linear',
                                     )
    elif solver =='gbr':
        parameters = {'learning_rate':[0.1, 0.01],
                      'n_estimators':[1500], 
                      'max_depth':[5, 8, 10]}
        model = GBR()

    elif solver == 'svr':
        parameters = {'C':[1, 10, 1e4], 
                      'gamma':[0.001, 0.01, 0.1]}
        model = SVR(kernel='rbf')
        
                
    elif solver=='nn':
        parameters = {'hidden_layer_sizes':[150, 250], 
                      'max_iter':[100,500]}
        model = nn.MLPRegressor(activation='logistic') 

    elif solver=='rfr':
        parameters = {'max_depth':[5, 10, 15], 'n_estimators':[250, 500, 1000]}
        #parameters = {'max_depth': [5], 'n_estimators': [250]}
        model = RandomForestRegressor(random_state=42)

    else:
        raise ValueError('Unknown Solver Name.')
        
    scores = None #['r2']
    model = GridSearchCV(model, 
                         parameters, 
                         cv=3, 
                         scoring=scores, 
                         n_jobs=-1,
                         verbose=1)
    
    return model