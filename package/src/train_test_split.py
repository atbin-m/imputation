import datetime
import pandas as pd
import numpy as np
import logging
logging.basicConfig(format="%(lineno)s:%(funcName)s:%(message)s", 
                    level=logging.INFO)

def _train_test_split(df, config, begin_test_timestamp, deltat, end_test_timestamp, fbprophet):
    tvar = config.variables['tvar']
    yvar = config.variables['yvar']

    # Test set   
    if end_test_timestamp is None:
        end_test_timestamp = begin_test_timestamp + datetime.timedelta(deltat)
  
    test_set = df[(df[tvar]>= begin_test_timestamp) &
                  (df[tvar]< end_test_timestamp)]

    # Train set
    end_train_timestamp = begin_test_timestamp
    train_set = df[df[tvar]< end_train_timestamp] #setting upper timelimit in train_set

    if not fbprophet:
        # No. of days with fully or partially missing data
        number_of_missing_days = train_set[train_set[yvar].isna()][tvar].map(lambda x: "%s-%s-%s"%(x.year, x.month, x.day) ).nunique()

        # to compensate this loss of data from the training set we need to extend
        # no. of days in training set
        extended_deltat = deltat + number_of_missing_days
        begin_train_timestamp = end_train_timestamp -\
                                datetime.timedelta(extended_deltat)
    else:
        begin_train_timestamp = df[tvar].min()
        #begin_train_timestamp = datetime.datetime.strptime("2013-03-05 00:00:00", "%Y-%m-%d %H:%M:%S")

    train_set = train_set[train_set[tvar]>= begin_train_timestamp]  #setting lower timelimit in train_set

    
    logging.info("Train set Start: {} End: {}".format(train_set[tvar].min().strftime("%Y-%m-%d"),
                                                         train_set[tvar].max().strftime("%Y-%m-%d")))
    logging.info("Test set Start: {} End: {}".format(test_set[tvar].min().strftime("%Y-%m-%d"),
                                                         test_set[tvar].max().strftime("%Y-%m-%d")))
    
    # Removing slices of data from train set that contain NaN yvar. 
    train_set = train_set[~train_set[yvar].isna()]
    return test_set, train_set, end_test_timestamp
    

def groups_of_train_test_set(df, config, fbprophet=None):
    """
    Divides given dataframe into list of test and train sets, within 
    the time period of interest.
    """
   
    tvar = config.variables['tvar']
    xvar = config.variables['xvar'] + config.variables['xvar_derived']
    
    begin_date = config.timestamps['begin_date']
    end_date = config.timestamps['end_date']
    deltat =  config.timestamps['deltat']

    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

   
    # Check if ustar threshold is provided for year of interest
    if config.data['ustar']==True:
        if not begin_date.year in config.data['ustar_map'].keys():
            raise ValueError('{} is missing from config/data/ustar_map'.format(begin_date.year))
        if not end_date.year in config.data['ustar_map'].keys():
            raise ValueError('{} is missing from config/data/ustar_map'.format(end_date.year))


    if (end_date - begin_date).days < deltat:
        raise ValueError("Time difference in days between begin and end date" + 
                         "must be greater than deltat.")


    df = df.loc[df[tvar] <= end_date]

    number_of_train_test_sets = int((end_date - begin_date).total_seconds()/\
                                    datetime.timedelta(deltat).total_seconds())

    begin_test_timestamp = begin_date
    

    test_df = []
    train_df = []    
    for i in range(number_of_train_test_sets):
        if i == number_of_train_test_sets-1:
            end_test_timestamp = end_date
        else:
            end_test_timestamp = None


        i_test_set, i_train_set, end_test_timestamp =\
                              _train_test_split(df.copy(), config,
                                                begin_test_timestamp,
                                                deltat, end_test_timestamp,
                                                fbprophet)
        begin_test_timestamp = end_test_timestamp
                 
        # Interpolating where x-var is nan.
        i_test_set[xvar] = i_test_set[xvar].interpolate()
        i_train_set[xvar] = i_train_set[xvar].interpolate()
                        

        i_test_set['Set_rank'] = i
        i_train_set['Set_rank'] = i
        
        if i == 0:
            test_df = i_test_set
            train_df = i_train_set
        else:
            test_df = pd.concat((test_df, i_test_set))
            train_df = pd.concat((train_df, i_train_set))


    return test_df, train_df


def layer_train_test_set(df, config, missing_frac=0.5):
    tvar = config.variables['tvar']
    yvar = config.variables['yvar']

    begin_date = config.timestamps['begin_date']
    end_date = config.timestamps['end_date']
    deltat =  config.timestamps['deltat']

    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    # Check if ustar threshold is provided for year of interest
    if config.data['ustar'] == True:
        if not begin_date.year in config.data['ustar_map'].keys():
            raise ValueError('{} is missing from config/data/ustar_map'.format(begin_date.year))
        if not end_date.year in config.data['ustar_map'].keys():
            raise ValueError('{} is missing from config/data/ustar_map'.format(end_date.year))

    if (end_date - begin_date).days < deltat:
        raise ValueError("Gap between begin and end date must be greater than deltat.")

    logging.info(f"Data used only between dates {begin_date} and {end_date} (both inclusive).")
    df = df.loc[(df[tvar] <= end_date) & (df[tvar] >= begin_date)]

    # Finding arbitrary test window of gap length = deltatdays
    # If 50% of data is missing set new test start date.
    number_of_missing_days = deltat
    while number_of_missing_days > missing_frac*deltat:
        test_end_day = pd.to_datetime('2100-01-01', format='%Y-%M-%d')  # arbitrarily large back date
        while test_end_day > df[tvar].max():
            test_start_day = pd.to_datetime(np.random.choice(df[tvar], 1)[0])
            test_end_day = test_start_day + np.timedelta64(deltat, 'D')

        test_df = df.loc[(df[tvar] >= test_start_day) &
                         (df[tvar] <= test_end_day)].copy()
        train_df = pd.concat([df.loc[df[tvar] < test_start_day],
                              df.loc[df[tvar] > test_end_day]]).copy()

        number_of_missing_days = test_df[test_df[yvar].isna()][tvar].map(lambda x: "%s-%s-%s"%(x.year, x.month, x.day) ).nunique()

    logging.info(f'Test interval start: {test_start_day} end: {test_end_day}')


    # Diving train into further train subsets for layer 1, layer 2 trainings
    ntrain = train_df.shape[0]
    ntrain_layer1_ = ntrain//2   #50% is train for layer 1
    ntrain_layer2 = ntrain - ntrain_layer1_

    # Further dividing ntrain_layer1 into 6 subsets
    ntrain_subsets_layer1 = ntrain_layer1_//6
    ntrain_layer1 = [ntrain_subsets_layer1]*5 + [ntrain_layer1_ - 5*ntrain_subsets_layer1]

    ntrain_whole = ntrain_layer1 + [ntrain_layer2]

    if sum(ntrain_whole) != ntrain:
        raise ValueError('LHS!=RHS for train sample size')

    train_labels = []
    for i, j in zip(ntrain_whole,
                    ['layer1_subset1', 'layer1_subset2', 'layer1_subset3',
                     'layer1_subset4', 'layer1_subset5','layer1_subset6', 'layer2']):
        train_labels.extend([j] * i)

    np.random.shuffle(train_labels)

    train_df['Set_rank'] = train_labels
    test_df['Set_rank'] = 'test'

    return test_df, train_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'configs')
    import test_config as conf
    import pandas as pd

    import importlib
    importlib.reload(conf)

    """
    try:
        df = pd.read_csv('../../data_out/Calperum_L3_processed.csv', parse_dates=['DateTime'])
    except FileNotFoundError:
        df = pd.read_csv('/media/atbin/Backup/Data_Processing/Package/data_out/Calperum_L3_processed.csv',
                         parse_dates=['DateTime'])

    test, train = groups_of_train_test_set(df, conf, fbprophet=False)
    """

    df = pd.read_csv('data_out/Gingin_L4_processed.csv', parse_dates=['DateTime'])
    test, train = groups_of_train_test_set(df, conf)

    #df = pd.read_csv('data_out/Gingin_L4_processed.csv', parse_dates=['DateTime'])
    #test, train = layer_train_test_set(df, conf)

