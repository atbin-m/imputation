import numpy as np
import pandas as pd
import test_list_of_models as lm
from sklearn import preprocessing
import utils
import logging
logging.basicConfig(format="%(lineno)s:%(funcName)s:%(message)s", 
                    level=logging.INFO)
import plot_utils
import matplotlib.pyplot as plt


class Model_fit(object):
    def __init__(self, solver_name, ytrain, xtrain, ytest, xtest, ttest):
        self.solver_name = solver_name
        self.ytrain = ytrain
        self.xtrain = xtrain
        self.ytest = ytest
        self.xtest = xtest
        
        self.ttest = ttest    # time variable, not used in analysis
        self.ttest_min = self.ttest.min().strftime('%Y-%m-%d')
        self.ttest_max = self.ttest.max().strftime('%Y-%m-%d')
        
    def model_fit(self):
        
        logging.info("Solver in use: {}".format(self.solver_name))    
        logging.info("Time window: {} to {}".format(self.ttest_min, 
                                                 self.ttest_max))    
        
        # Loading solver
        model, scale_data = lm.model(self.solver_name)
        
        # Scaling features if required
        if scale_data==True:
            scaler = preprocessing.MinMaxScaler( )
            ytrain_processed =\
                        scaler.fit_transform(self.ytrain.values.reshape(-1,1))       
            ytrain_processed = ytrain_processed.squeeze()
        else:
            ytrain_processed, ytest_processed = self.ytrain, self.ytest
    
        
        # Fitting
        yfit = model.fit(self.xtrain, ytrain_processed)
    
        print("{}: Best parameters set:".format(self.solver_name))
        print(model.best_params_)
        print("Best score: {0:1.4f}".format(model.best_score_))  
        
        self.best_hyper_params = str(model.best_params_)
    
        # Prdiction based on fitted model
        ytest_predicted_temp = model.best_estimator_.predict(self.xtest)
    
        # Inverse scaling the features
        if scale_data==True:                
            ytest_predicted =\
                  scaler.inverse_transform(ytest_predicted_temp.reshape(-1,1))
            self.ytest_predicted = ytest_predicted.squeeze()
        else:
            self.ytest_predicted = ytest_predicted_temp
            
    def fit_stats(self):
        ytest_not_nan = self.ytest.values[~self.ytest.isna()]
        ytest_predicted_not_nan = self.ytest_predicted[~self.ytest.isna()]
        
        rmse, r2, mbe =\
            utils.diagnostic_stats(ytest_not_nan, ytest_predicted_not_nan)
        std = ytest_predicted_not_nan.std()
        corr = np.corrcoef(ytest_not_nan, ytest_predicted_not_nan)[0,1]
        self.pred_stats = {'rmse':rmse, 
                           'r2':r2, 
                           'mbe':mbe,
                           'std':std,
                           'corr':corr}
        
        summary = {'Solver': self.solver_name,
                      'Best_params': self.best_hyper_params,
                      'RMSE': self.pred_stats['rmse'], 
                      'R2': self.pred_stats['r2'],
                      'std': self.pred_stats['std'], 
                      'Corr.': self.pred_stats['corr'],
                      'MBE': self.pred_stats['mbe'],
                      'Test_begin': self.ttest_min,
                      'Test_end': self.ttest_max,
                      }

        return summary
            
    def impute_table(self, config):       
        predicted_y = [self.ttest.values, self.ytest.values, self.ytest_predicted]
        return predicted_y
        
    def make_plots(self, config):
        title = self._make_title(config.data['tower'])
        
        fig, _=  plot_utils.plot_diagnostics(self.ttest, self.ytest, 
                                             self.ytest_predicted, 
                                             self.xtest, 
                                             [self.pred_stats['rmse'],
                                              self.pred_stats['r2'],
                                              self.pred_stats['mbe']], 
                                             
                                             title)
        if config.result['save_plots'] == True:
            filepath = '../../plot/'
            plt.savefig(filepath + title + '.jpeg')
            plt.close()
        return fig

    def _make_title(self, tower):
        
        title = tower + "_(%s)"%self.solver_name +\
                        "_%s"%self.ttest_min
        
        return title
        
class Model_runner(Model_fit):
    """
    Fits model for all train/test pairs for all solvers.
    """
    def __init__(self, config, test, train):
        self.conf = config

        self.xvar = self.conf.variables['xvar']
        self.yvar = self.conf.variables['yvar']
        self.tvar = self.conf.variables['tvar']
        
        self.test = test
        self.train = train
        
        assert self.test['Set_rank'].nunique() == self.train['Set_rank'].nunique()
        self.set_ranks = self.test['Set_rank'].unique()
        
        self.solver_names = self.conf.solvers

        
    def iterate_over_solvers(self, atrain, atest):
        summaries = []
        predicted_table = {}
        for asolver in self.solver_names:
            print(asolver)
            modelf = Model_fit(asolver, 
                               atrain[self.yvar], atrain[self.xvar],
                               atest[self.yvar], atest[self.xvar],
                               atest[self.tvar])
            modelf.model_fit()
            
            summary = modelf.fit_stats()
            summaries.append(summary)
            
            modelf.make_plots(self.conf)
            
            impt = modelf.impute_table(self.conf)
            predicted_table[self.yvar + '_predicted_%s'%asolver] = impt[2]
            predicted_table[self.yvar + '_imputed_%s'%asolver] = np.nan 
            
            # DateTime array for all solvers must be exactly same
            if self.tvar not in predicted_table:
                predicted_table[self.tvar] = impt[0]
                predicted_table[self.yvar] = impt[1]
            else:
                if np.array_equal(impt[0], predicted_table[self.tvar]):
                    pass
                else:
                    raise AttributeError('DateTime array for a train/test' +\
                                         'set must be identical.')        
        return summaries, predicted_table
    
    def iterate_over_traintest_sets(self):
        
        full_summary = []
        full_predicted_table = {}
        
        for set_rank in self.set_ranks:
            atest = self.test.loc[self.test['Set_rank'] == set_rank]
            atrain = self.train.loc[self.train['Set_rank'] == set_rank]
            
            summaries, predicted_table = self.iterate_over_solvers(atrain, atest)
            full_summary.extend(summaries)
            
            predicted_table = pd.DataFrame(predicted_table)
            if isinstance(full_predicted_table, dict):
                full_predicted_table = predicted_table
            else:
                full_predicted_table = pd.concat((full_predicted_table, 
                                                predicted_table))
                        
            sample = pd.DataFrame(summaries)[['std', 'Corr.', 'Solver']].values
            refstd = atest[~atest[self.yvar].isna()][self.yvar].std()
            srange = (0, 1.5*sample[:,0].astype(float).max()/refstd)
            self._make_taylor_diagram(sample, refstd, srange, 
                                      atest[self.tvar].min().strftime("%Y-%m-%d"))
            
        self.full_summary = pd.DataFrame(full_summary)
        self.full_summary.sort_values(['Test_begin', 'RMSE'], 
                                      ascending=[True, True], inplace=True)
        
        if self.conf.result['save_summary']==True:        
            fn = self._save_summary()
            self.full_summary.to_csv(fn, index=False)

    
        if self.conf.result['save_imputed']==True:        
            self.imputed_df = self._merge_predicted_and_original_tables(full_predicted_table)
            
            fn = self._save_predicted_table()            
            self.imputed_df.to_csv(fn, index=False)

            
    def _save_summary(self):
        title = self.conf.data['tower']
        fn = '../../data_out/' + title + '_summary_stats.csv'
        return fn
    
    def _save_predicted_table(self):
        title = self.conf.data['tower']
        yobs_file = self.conf.data['yobs_file']
        
        fn = '../../data_out/' + title + '_' + yobs_file + '_imputed.csv'
        return fn
    
    def _merge_predicted_and_original_tables(self, impt_df):
        original_df = pd.read_csv("../../data_out/" + 
                                  self.conf.data['tower'] + '_' + 
                                  self.conf.data['yobs_file'] +
                                 '_processed.csv', parse_dates=['DateTime'])
        
        pred_df =  pd.merge(original_df, impt_df, 
                       on= self.tvar, how='left', 
                       suffixes=('', '_%s'%self.conf.data['tower']))
        
        # Creating imputed df
        for asolver in self.solver_names:
            pred_df[self.yvar + '_imputed_%s'%asolver].fillna(
                    pred_df[self.yvar], inplace=True)
            pred_df[self.yvar + '_imputed_%s'%asolver].fillna(
                    pred_df[self.yvar + '_predicted_%s'%asolver], inplace=True)
        
        return pred_df

    def _make_taylor_diagram(self, samples,  refstd, srange, ttest_min):
        
        title = self.conf.data['tower'] + '_' +  ttest_min
        fn = '../../plot/' + title
                        
        plot_utils.taylor_diagram(samples,  refstd, srange, title)    

        if self.conf.result['save_plots'] == True:
            plt.savefig(fn + '.jpeg')
            plt.close()
    
    
        
        
if __name__=="__main__":
    
    import test_config as conf
    import train_test_split
    
   
    df = pd.read_csv('../../data_out/Calperum_L3_processed.csv',
                     parse_dates=['DateTime'])
    test, train = train_test_split.groups_of_train_test_set(df, conf)

    # Fit
   
    modelf = Model_runner(conf, test, train)
    modelf.iterate_over_traintest_sets()