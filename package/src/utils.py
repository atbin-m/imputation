import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd


def diagnostic_stats(ytrue, ypred):
    """
    https://stats.stackexchange.com/questions/142248/difference-between-r-square-and-rmse-in-linear-regression
    
    https://www.sciencedirect.com/topics/engineering/mean-bias-error
    """
    if isinstance(ytrue, pd.core.series.Series):
        yleft = ytrue.values
    else:
        yleft = ytrue

    if isinstance(ypred, pd.core.series.Series):
        yright = ypred.values
    else:
        yright = ypred

    yleft_nonnan = yleft[~np.isnan(yleft)]
    yright_nonnan = yright[~np.isnan(yleft)]
    corr = np.corrcoef(yleft_nonnan, yright_nonnan)[0,1]


    n = len(yleft_nonnan)

    # Check that the ytrue and ypred are equal length vector.
    assert n == len(yright_nonnan)
    
    # sum squared error
    sse = np.sum((yleft_nonnan - yright_nonnan)**2)
    
    # root mean square error
    rmse = np.sqrt(sse/n)

    # total sum of squares
    tss = np.sum((yleft_nonnan - np.mean(yleft_nonnan))**2)
    tst = np.sum((yright_nonnan - np.mean(yright_nonnan))**2)
    tstp = tst**0.5
    tssp = tss**0.5
    
    soorat = np.sum((yleft_nonnan-np.mean(yleft_nonnan))*(yright_nonnan-np.mean(yright_nonnan)))
    
    # Rsquare
    ##rsqr = 1 - sse/tss
    rsqr = (soorat/(tssp*tstp))**2

    # Mean biased error
    mbe = np.mean(yleft_nonnan - yright_nonnan)

    corr = np.corrcoef(yleft_nonnan, yright_nonnan)[0,1]

    stddev = np.std(yright_nonnan)
    print("RMSE: %1.3f, R^2: %1.3f, MBE: %1.3f"%(rmse, rsqr, mbe))
    
    return rmse, rsqr, mbe, corr, stddev


### Custom cross-validation function: weighted F1 score on specific threshold
SCORES = {}
def objective_core(cls, Xtrain, ytrain, label, yscale, nfolds=3, **params):
    """
    Objective function to minimize, using custom cross-validation. Default CV is limited
    """
    cls_ = cls
    Xtrain = Xtrain.copy()
    ytrain = ytrain.copy()
    ystd, ymean = yscale

    # Update parameters with default parameters

    params = {**params, **{'n_jobs': -1}}

    print("------ Sampling new data point ------")

    kfold = KFold(n_splits=nfolds, shuffle=True)
    cval_results = defaultdict(list)
    all_scores = defaultdict(list)
    for train_index, test_index in kfold.split(Xtrain, ytrain):
        X_train_, X_test_ = Xtrain.iloc[train_index, :], Xtrain.iloc[test_index, :]
        y_train_, y_test_ = ytrain.iloc[train_index], ytrain.iloc[test_index]

        # Train regressor on current fold
        cls_.fit(X_train_, y_train_)
        y_test_predicted = cls_.predict(X_test_)

        cval_results['mse'].append(mean_squared_error(y_test_ * ystd + ymean,
                                                      y_test_predicted * ystd + ymean))

        ametric = diagnostic_stats(y_test_ * ystd + ymean, y_test_predicted * ystd + ymean)
        for k, metric_name in enumerate(['rmse', 'rsqr', 'mbe', 'corr', 'stddev']):
            all_scores[metric_name].append(ametric[k])

    score = np.mean(np.sqrt(cval_results['mse']))

    record_score = {}
    for k in ['rmse', 'rsqr', 'mbe', 'corr', 'stddev']:
        record_score[k] = np.mean(all_scores[k])
    SCORES[label] = record_score

    print("Params:", params)
    print("Score:", score)

    return score

#Validation function
def rmsle_cv(model, X_train, y_train, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    clf = cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf)
    rmse= np.sqrt(-clf)
    return(rmse)


# split a multivariate sequence into samples for LSTM
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)