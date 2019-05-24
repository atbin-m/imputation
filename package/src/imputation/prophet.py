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
        ttest_min = atest[self.tvar].min().strftime('%Y-%m-%d')
        ttest_max = atest[self.tvar].max().strftime('%Y-%m-%d')

        atrain = self._modify_column_header_(atrain)

        m = Prophet()
        m.fit(atrain)

        #future = m.make_future_dataframe(periods=self.deltat)
        ytest, ds =  atest.reset_index()[self.yvar], atest[self.tvar].values

        future = pd.DataFrame({'ds':ds})
        self.forecast = m.predict(future)
        ytest_predicted = self.forecast['yhat']

        pred_stats = self._fit_stats(ytest, ytest_predicted)

        asolver = 'fbprophet'
        summary = {'Solver': asolver,
                   'Best_params': np.nan,
                   'RMSE': pred_stats['rmse'],
                   'R2': pred_stats['r2'],
                   'std': pred_stats['std'],
                   'Corr.': pred_stats['corr'],
                   'MBE': pred_stats['mbe'],
                   'Test_begin': ttest_min,
                   'Test_end': ttest_max,
                   }

        predicted_table = pd.DataFrame({self.tvar:ds,
                                        self.yvar: ytest.values,
                                        self.yvar + '_predicted_%s' % asolver: ytest_predicted,
                                        self.yvar + '_imputed_%s' % asolver: np.nan})
        return summary, predicted_table

    def _modify_column_header_(self, tempdf):
        return pd.DataFrame({'ds':tempdf[self.tvar], 'y':tempdf[self.yvar]})

    def _iterate_over_train_test_sets(self):

        full_summary = []
        full_predicted_table = {}

        for set_rank in self.set_ranks:
            atest = self.test_df.loc[self.test_df['Set_rank'] == set_rank]
            atrain = self.train_df.loc[self.train_df['Set_rank'] == set_rank]

            summary, predicted_table = self._solver(atrain, atest)
            full_summary.append(summary)

            if isinstance(full_predicted_table, dict):
                full_predicted_table = predicted_table
            else:
                full_predicted_table = pd.concat((full_predicted_table,
                                                  predicted_table))

        full_summary = pd.DataFrame(full_summary)

        if self.result['save_summary'] == True:
            self._save_summary(full_summary)

        if self.result['save_imputed'] == True:
            self.imputed_df = self._merge_predicted_and_original_tables(full_predicted_table)
            self._save_predicted_table(self.imputed_df)

    def _fit_stats(self, ytest, ytest_predicted):
        ytest_not_nan = ytest.values[~ytest.isna()]
        ytest_predicted_not_nan = ytest_predicted[~ytest.isna()]

        rmse, r2, mbe = utils.diagnostic_stats(ytest_not_nan, ytest_predicted_not_nan)
        std = ytest_predicted_not_nan.std()
        corr = np.corrcoef(ytest_not_nan, ytest_predicted_not_nan)[0, 1]
        pred_stats = {'rmse': rmse,
                      'r2': r2,
                      'mbe': mbe,
                      'std': std,
                      'corr': corr}

        return pred_stats

    def _save_summary(self, full_summary_prophet):
        # reading full summary of primary tower
        title = self.tower + '_' + self.yobs_file
        fn = '../../data_out/' + title + '_summary_stats.csv'
        full_summary_ml = pd.read_csv(fn)

        self.full_summary = pd.concat((full_summary_ml,
                                       full_summary_prophet))

        self.full_summary.sort_values(['Test_begin', 'RMSE'],
                                      ascending=[True, True], inplace=True)
        self.full_summary.to_csv(fn, index=False)

    def _save_predicted_table(self, imputed_df_prophet):
        title = self.tower
        yobs_file = self.yobs_file

        fn = '../../data_out/' + title + '_' + yobs_file + '_imputed.csv'

        imputed_df_prophet.to_csv(fn, index=False)

    def _merge_predicted_and_original_tables(self, impt_df):
        original_df = pd.read_csv("../../data_out/" +
                                  self.tower + '_' + self.yobs_file + '_imputed.csv',
                                  parse_dates=[self.tvar])

        pred_df = pd.merge(original_df, impt_df, on=self.tvar, how='left',
                           suffixes=('', '_%s' % self.tower))

        # Creating imputed df
        asolver = 'fbprophet'

        mask = (pred_df[self.yvar + '_imputed_%s'%asolver].isna())
        pred_df.loc[mask, self.yvar + '_imputed_%s'%asolver]  = pred_df.loc[mask, self.yvar]
        mask = (pred_df[self.yvar + '_imputed_%s'%asolver].isna())
        pred_df.loc[mask, self.yvar + '_imputed_%s'%asolver]  = pred_df.loc[mask,
                                                                            self.yvar + '_predicted_%s' % asolver]

        return pred_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'configs/')
    import test_config as conf
    import importlib

    importlib.reload(conf)

    try:
        df = pd.read_csv('../../data_out/Calperum_L3_processed.csv', parse_dates=['DateTime'])
    except FileNotFoundError:
        df = pd.read_csv('/media/atbin/Backup/Data_Processing/Package/data_out/Calperum_L3_processed.csv',
                         parse_dates=['DateTime'])

    f = FBprophet(conf, df)
    f._iterate_over_train_test_sets()