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
from keras.layers import LSTM
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

# ----- Data preprocessing
df = pd.read_csv('data_out/Gingin_L4_processed.csv',
                     parse_dates=['DateTime'])

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
                             Integer(100, 500, name='n_estimators')],
            'model_instance': lgb.LGBMRegressor(),
            'data': 'subset1'},

    'RFE':
        {
            'params_space': [Integer(2, 25, name='max_depth'),
                             Integer(2, 15, name='min_samples_leaf'),
                             Integer(2, 15, name='min_samples_split'),
                             Integer(100, 500, name='n_estimators')],
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

# Prediction
for key in solvers_layer2:
    val = model_library[key]
    cls = val[f'model_instance_ensemble']
    y_predicted_test = cls.predict(X_test_pred_layer2)
    full_df.loc[test_filter & (~full_df[yvar].isna()),
                yvar + f'_predicted_test_ensemble_{key}'] =  y_predicted_test * ystd + ymean

# ------- LSTM at layer 2 -------
# LSTM -- Ensemble
NSTEPS = 5
NFEATURES = X_train_pred_layer2.shape[1]

# convert into input/output sequences
dataset_train = np.column_stack((X_train_pred_layer2, y_train_layer2))
dataset_trainX, dataset_trainy = utils.split_sequences(dataset_train, NSTEPS)
print(dataset_trainX.shape, dataset_trainy.shape)

# define model
model_lstm = Sequential()
model_lstm.add(LSTM(5, input_shape=(NSTEPS, NFEATURES), activation='relu', dropout=0.5, recurrent_dropout=0.5))
model_lstm.add(Dense(1, activation='linear'))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
history = model_lstm.fit(dataset_trainX, dataset_trainy,
                            validation_split=0.5, shuffle=False,
                            epochs=50, batch_size=32, verbose=2)

dataset_test = np.column_stack((X_test_pred_layer2, y_test_layer2))
dataset_testX, dataset_testy = utils.split_sequences(dataset_test, n_steps=NSTEPS)
yhat_test = model_lstm.predict(dataset_testX, verbose=0)

metric_lstm = utils.diagnostic_stats(dataset_testy*ystd + ymean,
                                     yhat_test.squeeze()*ystd + ymean)

yhat_test = np.concatenate((np.array([np.nan]*(NSTEPS-1)), yhat_test.squeeze()))

full_df.loc[test_filter & (~full_df[yvar].isna()),
            yvar + f'_predicted_test_ensemble_LSTM'] =  yhat_test * ystd + ymean

utils.SCORES['Layer2' + '_' + 'LSTM' + '_' + 'ensemble'] = {'rmse':metric_lstm[0],
                                                            'rsqr':metric_lstm[1],
                                                            'mbe':metric_lstm[2],
                                                            'corr':metric_lstm[3], 'stddev':metric_lstm[4]}

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

# LSTM -- Single
NSTEPS = 5
NFEATURES = Xtrain_bothlayers.shape[1]

# convert into input/output sequences
dataset_train = np.column_stack((Xtrain_bothlayers, ytrain_bothlayers))
dataset_trainX, dataset_trainy = utils.split_sequences(dataset_train, NSTEPS)
print(dataset_trainX.shape, dataset_trainy.shape)

# define model
model_lstm = Sequential()
model_lstm.add(LSTM(5, input_shape=(NSTEPS, NFEATURES), activation='relu', dropout=0.5, recurrent_dropout=0.5))
#model_lstm.add(Dense(3, kernel_initializer='normal', activation='relu'))
model_lstm.add(Dense(1, activation='linear'))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
history = model_lstm.fit(dataset_trainX, dataset_trainy,
                            validation_split=0.5, shuffle=False,
                            epochs=50, batch_size=32, verbose=2)

dataset_test = np.column_stack((X_test, y_test))
dataset_testX, dataset_testy = utils.split_sequences(dataset_test, n_steps=NSTEPS)
yhat_test = model_lstm.predict(dataset_testX, verbose=0)

metric_lstm = utils.diagnostic_stats(dataset_testy*ystd + ymean,
                                     yhat_test.squeeze()*ystd + ymean)

yhat_test = np.concatenate((np.array([np.nan]*(NSTEPS-1)), yhat_test.squeeze()))

full_df.loc[test_filter,
            yvar + f'_predicted_test_single_LSTM'] = yhat_test * ystd + ymean

utils.SCORES['Layer2' + '_' + 'LSTM' + '_' + 'single'] = {'rmse':metric_lstm[0],
                                                          'rsqr':metric_lstm[1],
                                                          'mbe':metric_lstm[2],
                                                          'corr':metric_lstm[3],
                                                          'stddev':metric_lstm[4]}

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

score_df = pd.DataFrame.from_dict(utils.SCORES).T.round(3)
score_df.to_csv('data_out/temp_full_score.csv', index_label='Models')

# Scaling back yvar.
full_df[yvar] = full_df[yvar] * ystd + ymean
full_df.to_csv('data_out/temp_full.csv')
print(score_df)
