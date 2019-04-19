import pandas as pd
import train_test_split
import os

class Solo_fit(object):

    def __init__(self, conf, df):

        self.station = conf.data['tower']
        self.yobs_file = conf.data['yobs_file']

        self.xvar = conf.variables['xvar']
        self.yvar = conf.variables['yvar']
        self.tvar = conf.variables['tvar']

        self.test_df, self.train_df = train_test_split.groups_of_train_test_set(df, conf)

        # Unique set ranks
        self.set_ranks = self.test_df['Set_rank'].unique()
        self.solver_names = conf.solvers

        self.path2solo = conf.data['path2solo']

    def _prep_data(self, set_rank, atrain, atest):
        # train data
        # Preparing data for SOFM
        atrain[self.xvar].to_csv('../../data_in/sofm_sws_input.csv',
                                 header=False, index=False)

        # Preparing data for SOLO
        atrain[self.xvar + [self.yvar]].to_csv('../../data_in/solo_sws_input.csv',
                                               header=False, index=False)

        # test data
        # Preparing data for SEQSOLO
        atest[self.xvar + [self.yvar]].to_csv('../../data_in/seqsolo_sws_input.csv',
                                              header=False, index=False)

    def _solver(self):
        os.system(self.path2solo + '/sofm/sofm ../../data_in/sws_sofm.inf')
        os.system(self.path2solo +  '/solo/solo ../../data_in/sws_solo.inf')
        os.system(self.path2solo +  '/seqsolo/seqsolo ../../data_in/sws_seqsolo.inf')

    def _iterate_over_train_test_sets(self):

        set_rank = 1
        atest = self.test_df.loc[self.test_df['Set_rank'] == set_rank]
        atrain = self.train_df.loc[self.train_df['Set_rank'] == set_rank]

        self._prep_data(set_rank, atrain, atest)

        #self._solver(atrain.copy(), atest.copy())
        self._solver()


        # pred_stats = self._fit_stats(atest_record,
        #                              temp_df[yvar_predicted])
        # summary = {'Solver': asolver + '_Panel Data',
        #            'Best_params': np.nan,
        #            'RMSE': pred_stats['rmse'],
        #            'R2': pred_stats['r2'],
        #            'std': pred_stats['std'],
        #            'Corr.': pred_stats['corr'],
        #            'MBE': pred_stats['mbe'],
        #            'Test_begin': ttest_min,
        #            'Test_end': ttest_max,
        #            }
        # summaries.append(summary)

    def _save_summary(self):
        pass

    def _save_predicted_table(self):
        pass

    def _merge_predicted_and_original_tables(self):
        pass

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/atbin/workd/imputation/package/src/imputation/configs/")
    import test_config as conf
    import importlib
    importlib.reload(conf)

    df = pd.read_csv('/media/atbin/Backup/Data_Processing/Package/data_out/Calperum_L3_processed.csv',
                     parse_dates=['DateTime'])

    f= Solo_fit(conf, df)
    f._iterate_over_train_test_sets()