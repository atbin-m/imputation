import numpy as np
import pandas as pd
import train_test_split
import utils
from fbprophet import Prophet

class FBprophet(object):

    def __init__(self, conf, df):
        self.tower = conf.data['tower']

        self.yobs_file = conf.data['yobs_file']
        self.result = conf.result

        # For primary tower
        self.yvar = conf.variables['yvar']
        self.tvar = conf.variables['tvar']
        self.deltat = conf.timestamps['deltat']

        test_df, train_df = train_test_split.groups_of_train_test_set(df,
                                                                      conf,
                                                                      fbprophet=True)
        self.test_df, self.train_df = test_df, train_df

        # Unique set ranks
        self.set_ranks = self.test_df['Set_rank'].unique()

    def _solver(self, atrain, atest):

        m = Prophet()
        m.fit(atrain)

        future = m.make_future_dataframe(periods=self.deltat)
        self.forecast = m.predict(future)


    def _iterate_over_train_test_sets(self):



if __name__ == "__main__":
    import test_config as conf
    import importlib

    importlib.reload(conf)

    try:
        df = pd.read_csv('../../data_out/Calperum_L3_processed.csv', parse_dates=['DateTime'])
    except FileNotFoundError:
        df = pd.read_csv('/media/atbin/Backup/Data_Processing/Package/data_out/Calperum_L3_processed.csv',
                         parse_dates=['DateTime'])

    f = FBprophet(conf, df)
    #f._iterate_over_train_test_sets()