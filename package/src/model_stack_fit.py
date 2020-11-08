import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from copy import deepcopy

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense

import sys
sys.path.insert(0, 'configs/')
import utils
import test_config as conf
import train_test_split

import logging
logging.basicConfig(format="%(lineno)s:%(funcName)s:%(message)s",
                    level=logging.INFO)

Xvar = conf.variables['xvar']
yvar = conf.variables['yvar']
frac = 0.7    #If missing data > 48 %, find a new test window

path_to_package = '/Users/pluto/Desktop/bag/tutoring/atbin/imputation/package/'

# ----- Data preprocessing
df = pd.read_csv(path_to_package + 'data_out/Gingin_L4_processed.csv', parse_dates=['DateTime'])

test_df_, train_df_ = train_test_split.layer_train_test_set(df, conf, missing_frac=frac)

# Combining data frame for scaling
full_df = pd.concat([train_df_, test_df_])
ymean, ystd = full_df[yvar].mean(), full_df[yvar].std()
yvar_val = full_df[yvar].values
yvar_val = (yvar_val - ymean)/ystd
yscale = (ymean, ystd)

setrank = full_df['Set_rank'].values
dtime = full_df['DateTime'].values

scaler = StandardScaler()
full_df = scaler.fit_transform(full_df[Xvar])
full_df = pd.DataFrame.from_records(full_df, columns=Xvar)
full_df['Set_rank'] = setrank
full_df['DateTime'] = dtime
full_df[yvar] = yvar_val

full_df.sort_values('DateTime', inplace=True)

# --------- Training
# Layer 1 training parameters
N_FOLDS = 3
N_CALLS = 51
THRESHOLD = 0.05

# List of models

model_library = {
    'LGBM':
        {
            'params_space': [Integer(2, 10, name='num_leaves'),
                             Categorical(['regression'], name="objective"),
                             Integer(2, 10, name='min_data_in_leaf'),
                             Real(10 ** -4, 10 ** 0, "uniform", name='learning_rate'),
                             Integer(100, 500, name='n_estimators')
                             ],
            'model_instance': lgb.LGBMRegressor(),
            'data': 'subset1'},

    'RFE':
        {
            'params_space': [Integer(2, 25, name='max_depth'),
                             Integer(2, 15, name='min_samples_leaf'),
                             Integer(2, 15, name='min_samples_split'),
                             Integer(100, 500, name='n_estimators')
                           ],
            'model_instance': RandomForestRegressor(),
            'data': 'subset2'
        },

    'SVM':
        {
            'params_space': [Integer(2, 6, name='degree'),
                             Categorical(['scale'], name='gamma'),
                             Categorical(['rbf', 'poly', 'sigmoid'], name='kernel'),
                             Real(1, 100, "uniform", name='C')],
            'model_instance': SVR(),
            'data': 'subset3'
        },

    'GP':
        {
            'params_space': [Categorical([None], name='kernel'),
                             Categorical(['fmin_l_bfgs_b', 'adam'], name='optimizer'),
                             Real(1e-4, 1, "uniform", name='alpha')],
            'model_instance': GaussianProcessRegressor(),
            'data': 'subset4'
        },

    'ANN':
        {
            'params_space': [Integer(2, 20, name='hidden_layer_sizes'),
                             Categorical(['relu'], name='activation'),
                             Categorical(['adam', 'lbfgs'], name='solver'),
                             Real(1e-4, 1, "uniform", name='alpha'),
                             Real(1e-3, 0.1, "uniform", name='learning_rate_init'),
                             Categorical(['constant', 'adaptive'], name='learning_rate')],
            'model_instance': MLPRegressor(),
            'data': 'subset5'
        },

    'LASSO':
        {'params_space': None, 'model_instance': Lasso(), 'data': 'subset6'}
}

solvers_layer1 = conf.solvers_layer1
solvers_layer2 = conf.solvers_layer2

# -------- LAYER 1 Training ---------------------------------------
for key in solvers_layer1:
    label = f'val_Layer1_{key}'
    val = deepcopy(model_library[key])
    subset = val['data']
    reg = val['model_instance']
    params_space = val['params_space']

    print(subset, reg)

    # Train/test split. Ignoring NaN yvar in training set
    train_layer1_df = full_df[full_df['Set_rank'] == f'layer1_{subset}'].copy()
    train_layer1_df = train_layer1_df[~train_layer1_df[yvar].isna()]
    X_train_layer1 = train_layer1_df[Xvar]
    y_train_layer1 = train_layer1_df[yvar]

    if key == 'LASSO':
        ith_model = make_pipeline(RobustScaler(), Lasso(alpha=1e-4, random_state=1))
        score = utils.rmsle_cv(ith_model, X_train_layer1, y_train_layer1, N_FOLDS)
        print("\nLasso score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))

    else:
        # Bayesian opt. part
        @use_named_args(params_space)
        def ith_objective(**params):
            cls = reg.set_params(**params)
            return utils.objective_core(cls, X_train_layer1, y_train_layer1,
                                        label, yscale,
                                        nfolds=N_FOLDS, **params)
        res = gp_minimize(ith_objective, params_space, n_calls=N_CALLS, random_state=0)
        "Best score=%.4f" % res.fun

        # Generating final optimized model instance
        print("Optimal parameters")
        params = {}
        for param, value in zip(params_space, res.x):
            print(f"Param: {param.name}, value: {value}")
            params[param.name] = value

        ith_model = reg.set_params(**params)

    ith_model.fit(X_train_layer1.values, y_train_layer1.values)

    # Overriding model instance by fitted model
    model_library[key]['model_instance_layer1'] = ith_model


# --- Training set for layer 2
layer2_filter = (full_df['Set_rank']=='layer2')
X_train_layer2 = full_df.loc[layer2_filter & (~full_df[yvar].isna()), Xvar]


test_filter = (full_df['Set_rank']=='test')
X_test_layer2 = full_df.loc[test_filter & (~full_df[yvar].isna()), Xvar]

X_all_layer2 = full_df[Xvar]

# Saving prediction for layer 2 using model trained on layer 1
for key in solvers_layer1:
    val = model_library[key]
    subset = val['data']
    reg = val['model_instance_layer1']

    y_pred_layer2 = reg.predict(X_train_layer2)
    full_df.loc[layer2_filter & (~full_df[yvar].isna()),
                yvar + f'_predicted_layer2_{key}'] = y_pred_layer2 #* ystd + ymean

    y_pred_test = reg.predict(X_test_layer2)
    full_df.loc[test_filter & (~full_df[yvar].isna()),
                yvar + f'_predicted_test_{key}'] = y_pred_test #* ystd + ymean

    # Prediction for all Xs
    y_predicted_entire = reg.predict(X_all_layer2)
    full_df[yvar + f'_predicted_layer2_Xs_all_ensemble_{key}'] = y_predicted_entire

# ---------------------- Layer 2 training
Xvar_pred_layer2 = [yvar + f'_predicted_layer2_{j}' for j in solvers_layer1]
X_train_pred_layer2 = full_df.loc[layer2_filter & (~full_df[yvar].isna()), Xvar_pred_layer2]
y_train_layer2 = full_df.loc[layer2_filter & (~full_df[yvar].isna()), yvar]

# ------- Ensemble run ------
# for solvers in layer 2, ie. using Xtrain = ypredicted from layer 1 models
for key in solvers_layer2:
    label = f'val_Layer2_ensemble_{key}'
    val = deepcopy(model_library[key])
    reg = val['model_instance']
    params_space = val['params_space']


    # Bayesian opt. part
    @use_named_args(params_space)
    def ith_objective(**params):
        cls = reg.set_params(**params)
        return utils.objective_core(cls, X_train_pred_layer2, y_train_layer2,
                                    label, yscale,
                                    nfolds=N_FOLDS, **params)


    res = gp_minimize(ith_objective, params_space, n_calls=N_CALLS, random_state=0)
    "Best score=%.4f" % res.fun

    # Generating final optimized model instance
    print("Optimal parameters")
    params = {}
    for param, value in zip(params_space, res.x):
        print(f"Param: {param.name}, value: {value}")
        params[param.name] = value

    ith_model = reg.set_params(**params)
    ith_model.fit(X_train_pred_layer2.values, y_train_layer2.values)

    # Model instance for ensemble
    model_library[key]['model_instance_ensemble'] = ith_model


# Final ENSEMBLE prediction on Xtest
Xvar_pred_test = [yvar + f'_predicted_test_{j}' for j in solvers_layer1]
X_test_pred_layer2 = full_df.loc[test_filter & (~full_df[yvar].isna()), Xvar_pred_test]
y_test_layer2 = full_df.loc[test_filter & (~full_df[yvar].isna()), yvar]

Xvar_all_pred_layer2 = [yvar + f'_predicted_layer2_Xs_all_ensemble_{key}' for j in solvers_layer1]
X_all_pred_layer2 = full_df[Xvar_all_pred_layer2]

# Prediction
for key in solvers_layer2:
    val = model_library[key]
    cls = val[f'model_instance_ensemble']
    y_predicted_test = cls.predict(X_test_pred_layer2)
    full_df.loc[test_filter & (~full_df[yvar].isna()),
                yvar + f'_predicted_test_ensemble_{key}'] =  y_predicted_test * ystd + ymean

    # Prediction for all Xs
    y_predicted_entire = cls.predict(X_all_pred_layer2)
    full_df[yvar + f'_predicted_for_allXs_ensemble_{key}'] = y_predicted_entire


# ------- Single model run ------
Xtrain_bothlayers = full_df.loc[(full_df['Set_rank'] != 'test') & (~full_df[yvar].isna()), Xvar]
ytrain_bothlayers = full_df.loc[(full_df['Set_rank'] != 'test') & (~full_df[yvar].isna()), yvar]

for key in solvers_layer2:
    label = f'val_Layer2_single_{key}'
    val = deepcopy(model_library[key])
    reg = val['model_instance']
    params_space = val['params_space']


    # Bayesian opt. part
    @use_named_args(params_space)
    def jth_objective(**params):
        cls = reg.set_params(**params)
        return utils.objective_core(cls, Xtrain_bothlayers, ytrain_bothlayers,
                                    label, yscale,
                                    nfolds=N_FOLDS, **params)


    res = gp_minimize(jth_objective, params_space, n_calls=N_CALLS, random_state=0)
    "Best score=%.4f" % res.fun

    # Generating final optimized model instance
    print("Optimal parameters")
    params = {}
    for param, value in zip(params_space, res.x):
        print(f"Param: {param.name}, value: {value}")
        params[param.name] = value

    jth_model = reg.set_params(**params)
    jth_model.fit(Xtrain_bothlayers.values, ytrain_bothlayers.values)

    # Model instance for ensemble
    model_library[key]['model_instance_single'] = jth_model


# Final SINGLE prediction on Xtest
X_test = full_df.loc[test_filter, Xvar]
y_test = full_df.loc[test_filter, yvar]

# Prediction
for key in solvers_layer2:
    val = model_library[key]
    cls = val[f'model_instance_single']
    y_predicted_test = cls.predict(X_test)
    full_df.loc[test_filter, yvar + f'_predicted_test_single_{key}'] =  y_predicted_test * ystd + ymean

    # Prediction for all Xs
    y_predicted_entire = cls.predict(full_df[Xvar])
    full_df[yvar + f'_predicted_for_allXs_single_{key}'] = y_predicted_entire * ystd + ymean

# ---------- Summary stats -------------------
ytest = full_df.loc[test_filter, yvar] * ystd + ymean

for key in solvers_layer2:
    for j in ['single', 'ensemble']:
        ytest_predicted = full_df.loc[test_filter, yvar + f'_predicted_test_{j}_{key}']

        print('Layer2', key, j)
        ametric = utils.diagnostic_stats(ytest, ytest_predicted)
        #print(ascore)
        all_scores = {}
        for k, metric_name in enumerate(['rmse', 'rsqr', 'mbe', 'corr', 'stddev']):
            all_scores[metric_name] = ametric[k]

        utils.SCORES['Layer2' + '_' + key + '_' + j] = all_scores



# LSTM runs ---------------------------------

# Filling training gap with Primary Solver [LGBM]
PRIMARY_SOLVER = ['LGBM']

full_lstm_df = pd.concat([train_df_, test_df_])

train_primary = train_df_[~train_df_[yvar].isna()]
X_train_primary = train_primary[Xvar]
y_train_primary = train_primary[yvar]

for key in PRIMARY_SOLVER:
    label = f'val_single_{key}'
    val = deepcopy(model_library[key])
    reg = val['model_instance']
    params_space = val['params_space']

    # Bayesian opt. part
    @use_named_args(params_space)
    def jth_objective(**params):
        cls = reg.set_params(**params)
        return utils.objective_core(cls, X_train_primary, y_train_primary,
                                    label, [1,0],
                                    nfolds=N_FOLDS, **params)


    res = gp_minimize(jth_objective, params_space, n_calls=N_CALLS, random_state=0)
    "Best score=%.4f" % res.fun

    # Generating final optimized model instance
    print("Optimal parameters")
    params = {}
    for param, value in zip(params_space, res.x):
        print(f"Param: {param.name}, value: {value}")
        params[param.name] = value

    jth_model = reg.set_params(**params)
    jth_model.fit(X_train_primary.values, y_train_primary.values)

    # Model instance for ensemble
    model_library[key]['model_instance_primary'] = jth_model

# Predicting for entire X
for key in PRIMARY_SOLVER:
    val = model_library[key]
    cls = val[f'model_instance_primary']
    full_lstm_df['ytempall_predicted'] = cls.predict(full_lstm_df[Xvar])

assert full_lstm_df['ytempall_predicted'].shape[0] == full_lstm_df.shape[0]


# Labelling test as nan so that in next step we can predict for Xtest and gaps in X_train
# at once.
new_yvar = yvar + '_filled'

full_lstm_df[new_yvar] = full_lstm_df[yvar].values
full_lstm_test_filter = (full_lstm_df['Set_rank']=='test')
full_lstm_df.loc[full_lstm_test_filter, new_yvar] = np.nan

assert test_df_.shape[0] == full_lstm_df[full_lstm_test_filter].shape[0]

full_lstm_df[new_yvar] = full_lstm_df[new_yvar].fillna(full_lstm_df['ytempall_predicted'])
full_lstm_df.drop(columns={'ytempall_predicted'}, inplace=True)

# Combining data frame for scaling
ymean_lstm, ystd_lstm = full_lstm_df[new_yvar].mean(), full_lstm_df[new_yvar].std()
yvar_val = full_lstm_df[new_yvar].values
yvar_val = (yvar_val - ymean_lstm)/ystd_lstm
yscale_lstm = (ymean_lstm, ystd_lstm)


dtime = full_lstm_df['DateTime'].values
set_rank = full_lstm_df['Set_rank'].values

scaler = StandardScaler()
full_lstm_df = scaler.fit_transform(full_lstm_df[Xvar])
full_lstm_df = pd.DataFrame.from_records(full_lstm_df, columns=Xvar)
full_lstm_df['DateTime'] = dtime
full_lstm_df['Set_rank'] = set_rank
full_lstm_df[new_yvar + '_scaled'] = yvar_val
full_lstm_df.sort_values('DateTime', inplace=True)


# Secondary solver LSTM run------
Xtrain_ = full_lstm_df.loc[full_lstm_df['Set_rank']!='test', Xvar]
ytrain_ = full_lstm_df.loc[full_lstm_df['Set_rank']!='test', new_yvar + '_scaled']

Xtest_ = full_lstm_df.loc[full_lstm_df['Set_rank']=='test', Xvar]
ytest_ = full_lstm_df.loc[full_lstm_df['Set_rank']=='test', new_yvar + '_scaled']

print('Train data:', Xtrain_.shape, ytrain_.shape)
print('Test data:', Xtest_.shape, ytest_.shape)


# LSTM -- Single
NSTEPS = 5
NFEATURES = Xtrain_.shape[1]

# convert into input/output sequences
dataset_train = np.column_stack((Xtrain_, ytrain_))
dataset_trainX, dataset_trainy = utils.split_sequences(dataset_train, NSTEPS)
print(dataset_trainX.shape, dataset_trainy.shape)

# define model
model_lstm = Sequential()
model_lstm.add(Bidirectional(LSTM(5, input_shape=(NSTEPS, NFEATURES), activation='relu', dropout=0.5, recurrent_dropout=0.5)))
model_lstm.add(Dense(1, activation='linear'))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
history = model_lstm.fit(dataset_trainX, dataset_trainy,
                            validation_split=0.5, shuffle=False,
                            epochs=50, batch_size=32, verbose=2)

dataset_test = np.column_stack((Xtest_, ytest_))
dataset_testX, dataset_testy = utils.split_sequences(dataset_test, n_steps=NSTEPS)
yhat_test = model_lstm.predict(dataset_testX, verbose=0)

metric_lstm = utils.diagnostic_stats(dataset_testy*ystd_lstm + ymean_lstm,
                                     yhat_test.squeeze()*ystd_lstm + ymean_lstm)

utils.SCORES['LSTM' + '_' + 'single'] = {'rmse':metric_lstm[0],
                                          'rsqr':metric_lstm[1],
                                          'mbe':metric_lstm[2],
                                          'corr':metric_lstm[3],
                                          'stddev':metric_lstm[4]}

score_df = pd.DataFrame.from_dict(utils.SCORES).T.round(3)
score_df.to_csv(path_to_package + 'data_out/temp_full_score.csv', index_label='Models')
print(score_df)

yhat_test = np.concatenate((np.array([np.nan]*(NSTEPS-1)), yhat_test.squeeze()))

full_lstm_df[yvar + f'_predicted_test_filled_LSTM'] = np.nan
full_lstm_df.loc[full_lstm_df['Set_rank']=='test', yvar + f'_predicted_test_filled_LSTM'] = yhat_test * ystd_lstm + ymean_lstm

# Predicting for the entire dataset
Xentire_ = full_lstm_df[Xvar]
yentire_ = full_lstm_df[new_yvar + '_scaled']
dataset_entire_ = np.column_stack((Xentire_, yentire_))
dataset_entireX_, _ = utils.split_sequences(dataset_entire_, NSTEPS)
yhat_entire_ = model_lstm.predict(dataset_entireX_, verbose=0)
yhat_entire_ = np.concatenate((np.array([np.nan]*(NSTEPS-1)), yhat_entire_.squeeze()))
full_lstm_df[yvar + f'_predicted_for_allXs_LSTM'] = yhat_entire_

# Scaling back yvar.
# Merging LSTM results with ensemble
final_df = pd.merge(full_df, full_lstm_df[['DateTime', yvar + '_filled_scaled', yvar + '_predicted_test_filled_LSTM',
                                           yvar + f'_predicted_for_allXs_LSTM']],
                    on ='DateTime', how='left')
final_df[yvar] = final_df[yvar] * ystd + ymean
final_df.to_csv(path_to_package + 'data_out/temp_full.csv')