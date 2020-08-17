import data_preparation
import pandas as pd
from train_test_split import groups_of_train_test_set
import model_fit
import panel_data
import plot_utils
import utils
import matplotlib.pyplot as plt
import prophet
import numpy as np


class ModelPipeline(object):
    def __init__(self, configuration):
        self.conf = configuration
        self.primary_st = self.conf.data['tower']
        
    def imputation_run(self, timeofday=None):
        # ------- Configs
        data_config, var_config =  self.conf.data, self.conf.variables
        print('Imputing tower: {}'.format(self.conf.data['tower']))        
        # ------ Data Preprocessing
        print("Data preprocessing: start")
        df = data_preparation.data_preprocessing(data_config, var_config)
        #df = pd.read_csv('data_out/Calperum_L4_processed.csv', parse_dates=['DateTime'])
        print("Data preprocessing: end")

        if timeofday=='night':
            cut = ((df['DateTime'].dt.hour > 18) | (df['DateTime'].dt.hour <= 7))
            df = df[cut].copy()
        elif timeofday=='day':
            cut = ((df['DateTime'].dt.hour > 7) & (df['DateTime'].dt.hour <= 18))
            df = df[cut].copy()           
        else:
            pass
                    
        # ------ Train test split
        test_df, train_df = groups_of_train_test_set(df, self.conf)

        # ------ Model fitting
        modelf = model_fit.Model_runner(self.conf, test_df, train_df)
        modelf.iterate_over_traintest_sets()
        self.df_imputed = modelf.imputed_df
        
    def imputation_run_sec_tower(self):
        # ----- Panel Data analysis
        self.conf.data['tower'] = self.conf.data['second_tower']

        print('Imputing tower: {}'.format(self.conf.data['tower']))
        
        # ------- Configs
        data_config =  self.conf.data
        var_config = self.conf.variables
         
        # ------ Data Preprocessing
        df_2nd_tower = data_preparation.data_preprocessing(data_config, var_config)
        
        # ------ Train test split
        test_df, train_df = groups_of_train_test_set(df_2nd_tower, self.conf)
        
        # ------ Model fitting
        modelf = model_fit.Model_runner(self.conf, test_df, train_df)
        modelf.iterate_over_traintest_sets()
        self.df_2nd_tower_imputed = modelf.imputed_df

        # restoring the name of the primary tower
        self.conf.data['tower'] = self.primary_st
        
    def panel_data_run(self):
        f = panel_data.PanelData_fit(self.conf, self.df_imputed, 
                                     self.df_2nd_tower_imputed)
        f._iterate_over_train_test_sets()

    def solo_run(self):
        pass

    def fbprophet_run(self):
        fb = prophet.FBprophet(self.conf, self.df_imputed)
        fb._iterate_over_train_test_sets()

    def taylor_diagram(self):
        summary_df, _ = self._read_summary_file()
        ttest_min_set = summary_df['Test_begin'].unique()

        for t in ttest_min_set:
            a_summary_df = summary_df[summary_df['Test_begin']==t]
            self._make_save_taylor_diagram(a_summary_df, t)        

    def _make_save_taylor_diagram(self, a_summary_df, ttest_min):
        sample = a_summary_df[['std', 'Corr.', 'Solver']].values
        refstd = 0.75*sample[:,0].min()
        srange = (0, 1.5*sample[:,0].astype(float).max()/refstd)
        
        ttest_min = pd.to_datetime(str(ttest_min)) 
        ttest_min = ttest_min.strftime("%Y-%m-%d")
        
        title = self.primary_st + '_' +  ttest_min
        fn = 'plot/' + title

        if np.isnan(refstd):
            pass
        else:
            plot_utils.taylor_diagram(sample, refstd, srange, title)

        if self.conf.result['save_plots'] == True:
            plt.savefig(fn + '.png')
            plt.close()

    def _read_summary_file(self):
        title = self.primary_st + '_%s'%self.conf.data['yobs_file'] + \
                                  self.conf.data['file_suffix']  
        fn = 'data_out/' + title + '_summary_stats.csv'
        full_summary = pd.read_csv(fn, parse_dates=['Test_begin', 'Test_end'])
        return full_summary, fn

    def _read_imputed_file(self):
        title = self.primary_st + '_%s'%self.conf.data['yobs_file'] + \
                                  self.conf.data['file_suffix']
        fn = 'data_out/' + title + '_imputed.csv'

        tvar = self.conf.variables['tvar']
        begin_date = self.conf.timestamps['begin_date']
        end_date = self.conf.timestamps['end_date']
        imputed_df = pd.read_csv(fn, parse_dates=[tvar])

        imputed_df = imputed_df.loc[(imputed_df[tvar] <= end_date) &
                                    (imputed_df[tvar] >= begin_date)]
        return begin_date, end_date, imputed_df

    def overall_rmse(self, extra_solvers=None):
        yvar = self.conf.variables['yvar']
        tower = self.conf.data['tower']
        yvar_tower = yvar + '_' + tower

        yvar_predicted = yvar + '_predicted'

        ttest_min, ttest_max, imputed_df = self._read_imputed_file()
        solvers = self.conf.solvers
        
        if extra_solvers:
            solvers  = solvers + extra_solvers
            
        summaries = []
        for asolver in solvers:
            ytest_not_nan = imputed_df[yvar_tower].values[~imputed_df[yvar_tower].isna()]

            yvar_predicted_solver = yvar_predicted + '_'+ asolver
            ytest_predicted_not_nan = imputed_df[yvar_predicted_solver][~imputed_df[yvar_tower].isna()]

            rmse, r2, mbe = utils.diagnostic_stats(ytest_not_nan, ytest_predicted_not_nan)
            std = ytest_predicted_not_nan.std()
            corr = np.corrcoef(ytest_not_nan, ytest_predicted_not_nan)[0, 1]

            pred_stats = {'rmse': rmse, 'r2': r2, 'mbe': mbe, 'std': std, 'corr': corr}

            summary = {'Solver': asolver+ '- Overall',
                       'Best_params': {},
                       'RMSE': pred_stats['rmse'],
                       'R2': pred_stats['r2'],
                       'std': pred_stats['std'],
                       'Corr.': pred_stats['corr'],
                       'MBE': pred_stats['mbe'],
                       'Test_begin': ttest_min,
                       'Test_end': ttest_max,
                       }
            summaries.append(summary)

        summary_file_before, summary_fn = self._read_summary_file()
        summary_for_overall_data =  pd.DataFrame(summaries)
        summary_file_after = pd.concat([summary_file_before,
                                        summary_for_overall_data])
        summary_file_after.sort_values(['Test_begin', 'RMSE'], ascending=[True, True],
                                       inplace=True)
        summary_file_after.reset_index(drop=True, inplace=True)

        if self.conf.result['save_summary']==True:
            summary_file_after.to_csv(summary_fn, index=False)
        return summary_file_after

if __name__=="__main__":
    """
    there are two configuration files that need to be checked everytime
    a. test_config
    b. test_list_of_models
    """
    import importlib
    import sys
    sys.path.insert(0, 'configs/')

    # L3 configy
    #import driver_config as confs

    #L4 config
    import test_config as confs
    
    # --------------- full run --------------------------
    importlib.reload(confs)
    p = ModelPipeline(confs)
    p.imputation_run(timeofday=None)
    p.overall_rmse(extra_solvers)
    p.taylor_diagram()