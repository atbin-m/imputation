import numpy as np
import statsmodels.formula.api as sm
import pandas as pd
import train_test_split
import utils

class PanelData_fit(object):
    
    def __init__(self, conf, df_primary, df_secondary):
        self.primary_st = conf.data['tower']
        self.secondary_st =  conf.data['second_tower']
        self.yobs_file = conf.data['yobs_file']
        self.result = conf.result
        
        # For primary tower
        self.xvar = conf.variables['xvar']
        self.yvar = conf.variables['yvar']
        self.tvar = conf.variables['tvar']
        
        # For secondary tower
        self.suffix = '_TowerB'
        self.xvar_sec = [i + self.suffix for i in self.xvar]
        self.yvar_sec = self.yvar + self.suffix
        self.tvar_sec = self.tvar
               
        joint_df = pd.merge(df_primary, df_secondary, on=self.tvar, how='left', 
                            suffixes = ('', self.suffix))
        test_df, train_df = train_test_split.groups_of_train_test_set(joint_df, 
                                                                      conf)
        self.test_df, self.train_df = test_df, train_df
            
        # Unique set ranks
        self.set_ranks = self.test_df['Set_rank'].unique()       
        self.solver_names = conf.solvers
            
    def _iterate_over_solver(self, atrain, atest):
        
        ttest_min = atest[self.tvar].min().strftime('%Y-%m-%d')
        ttest_max = atest[self.tvar].max().strftime('%Y-%m-%d')

        print("Running panel data for %s-%s interval".format(ttest_min, 
                                                             ttest_max))
        
        atest_record = atest[self.yvar].reset_index()[self.yvar]
        atest[self.yvar] = np.nan
        
        # Combine test with train data for each tower separately
        # only keeping useful variables
        listofvars = self.xvar + [self.yvar] + [self.tvar] 
        joint_df_prm_tower = pd.concat((atrain[listofvars], 
                                        atest[listofvars]))
        joint_df_prm_tower['Station'] = self.primary_st

        summaries = []        
        predicted_table = {}
        for asolver in self.solver_names:
            print("Running panel data for %s solver".format(asolver))
            
            listofvars_sec = self.xvar_sec + [self.yvar + '_predicted_' +
                                              asolver + self.suffix] +\
                                              [self.tvar_sec] 
            joint_df_sec_tower = pd.concat((atrain[listofvars_sec], 
                                            atest[listofvars_sec]))
            joint_df_sec_tower['Station'] = self.secondary_st 
            
            # renaming the column header of second tower to match that 
            # of the first tower so that they can be merged for panel data
            # input
            joint_df_sec_tower.columns = joint_df_prm_tower.columns
            joint_df_arank = pd.concat((joint_df_prm_tower, joint_df_sec_tower))

            # Panel data fit                               
            intercept, xvar_coeffs, tvar_coeffs_df = self._fit(joint_df_arank)
         
            temp_df = pd.merge(atest, tvar_coeffs_df, on=self.tvar, 
                               how='left')

            # yvar = \Sum xcoeff*X + intercept + station coeff
            yvar_predicted = '%s_predicted_%s_Panel Data'%(self.yvar, asolver)
            temp_df[yvar_predicted] = intercept +\
            (temp_df[self.xvar]*xvar_coeffs).sum(axis=1) + temp_df['tcoeff']
            temp_df.drop([self.yvar], axis=1, inplace=True)

            
            pred_stats = self._fit_stats(atest_record, 
                                      temp_df[yvar_predicted])
            summary = {'Solver': asolver + '_Panel Data',
                      'Best_params': np.nan,
                      'RMSE': pred_stats['rmse'], 
                      'R2': pred_stats['r2'],
                      'std': pred_stats['std'], 
                      'Corr.': pred_stats['corr'],
                      'MBE': pred_stats['mbe'],
                      'Test_begin':ttest_min,
                      'Test_end': ttest_max,
                      }
            summaries.append(summary)
            
            predicted_table[yvar_predicted] = temp_df[yvar_predicted].values
            predicted_table[self.yvar + '_imputed_%s_Panel Data'%asolver] = np.nan 
            
            # DateTime array for all solvers must be exactly same
            if self.tvar not in predicted_table:
                predicted_table[self.tvar] = temp_df[self.tvar].values
            else:
                if np.array_equal(temp_df[self.tvar].values, 
                                  predicted_table[self.tvar]):
                    pass
                else:
                    raise AttributeError('DateTime array for a train/test' +\
                                         'set must be identical.')        
        return summaries, predicted_table
    
    
    def _iterate_over_train_test_sets(self):
        full_summary_sec_t = []
        full_predicted_table_sec_t = {}
        for set_rank in self.set_ranks:
            predicted_table = {}

            atest = self.test_df.loc[self.test_df['Set_rank'] == set_rank]
            atrain = self.train_df.loc[self.train_df['Set_rank'] == set_rank]
                
            summaries, predicted_table = self._iterate_over_solver(atrain.copy(),
                                                                   atest.copy())
            full_summary_sec_t.extend(summaries)

            predicted_table = pd.DataFrame(predicted_table)
            
            if isinstance(full_predicted_table_sec_t, dict):
                full_predicted_table_sec_t = predicted_table
            else:
                full_predicted_table_sec_t = pd.concat((full_predicted_table_sec_t, 
                                                        predicted_table))
                                
        self.full_summary_sec_t = pd.DataFrame(full_summary_sec_t)
        if self.result['save_summary']==True:     
            self._save_summary(self.full_summary_sec_t)            
            
        if self.result['save_imputed']==True:        
            self.imputed_df =\
            self._merge_predicted_and_original_tables(full_predicted_table_sec_t)        
            self._save_predicted_table(self.imputed_df)
            
    def _save_summary(self, full_summary_sec_t):
        # reading full summary of primary tower
        title = self.primary_st
        fn = '../../data_out/' + title + '_summary_stats.csv'
        full_summary_primary_t = pd.read_csv(fn)

        self.full_summary = pd.concat((full_summary_primary_t, 
                                       full_summary_sec_t))
        
        self.full_summary.sort_values(['Test_begin', 'RMSE'], 
                                      ascending=[True, True], inplace=True)
        self.full_summary.to_csv(fn, index=False)    
    
    def _save_predicted_table(self, imputed_df_sec_t):
        title = self.primary_st
        yobs_file = self.yobs_file
        
        fn = '../../data_out/' + title + '_' + yobs_file + '_imputed.csv'
        
        imputed_df_sec_t.to_csv(fn, index=False)
    

    def _merge_predicted_and_original_tables(self, impt_df):
        original_df = pd.read_csv("../../data_out/" + 
                                  self.primary_st + '_' + 
                                  self.yobs_file +
                                 '_imputed.csv', parse_dates=[self.tvar])
        
        pred_df =  pd.merge(original_df, impt_df, on= self.tvar, how='left', 
                            suffixes=('', '_%s'%self.primary_st))
                
        # Creating imputed df
        for asolver in self.solver_names:
            pred_df[self.yvar + '_imputed_%s_Panel Data'%asolver].fillna(
                    pred_df[self.yvar], inplace=True)
            pred_df[self.yvar + '_imputed_%s_Panel Data'%asolver].fillna(
                    pred_df[self.yvar + '_predicted_%s_Panel Data'%asolver], 
                    inplace=True)
        
        return pred_df


    def _fit(self, prelim_second_towers_stacked_df):
        FE_ols = sm.ols(formula=self._formula(), 
                        data=prelim_second_towers_stacked_df).fit()
        
        summary_df = (FE_ols.summary2().tables[1])

        # y-Intercept and coefs per independent variable
        intercept = summary_df.loc['Intercept', 'Coef.']
        xvar_coeffs = []
        for i in self.xvar:
            xvar_coeffs.append(summary_df.loc[i, 'Coef.'])
         
        # t-coeffs    
        # Extracting only datetime coeffs from the summary dataframe
        # first -1 is to avoid intercept, second -1 is for second tower intercept
        n = summary_df.shape[0] - 1 - 1 - len(self.xvar) 
        prelim_tower_tcoeff = summary_df.reset_index().loc[1:n,
                                                    ['index','Coef.']]
        prelim_tower_tcoeff['DateTime'] =\
        prelim_tower_tcoeff['index'].str.extract('(....-..-.. ..:..:..)', 
                           expand=True)
        prelim_tower_tcoeff['DateTime'] =\
                         pd.to_datetime(prelim_tower_tcoeff['DateTime'])
        prelim_tower_tcoeff.rename(columns={'Coef.':'tcoeff'}, 
                                   inplace=True)
        prelim_tower_tcoeff.drop(['index'], axis=1, inplace=True)
        
        return intercept, xvar_coeffs, prelim_tower_tcoeff
    
    def _formula(self):
        xvart = ''
        for j, val in enumerate(self.xvar):
            if j==0:
                xvart = val
            else:
                xvart += ' + ' + val
                
        expr = self.yvar + ' ~ ' + xvart + ' + ' + 'C(%s) + '%self.tvar + 'C(Station)'
        print("Panel data relationship: {}".format(expr))
        return expr
    
    def _fit_stats(self, ytest, ytest_predicted):
        ytest_not_nan = ytest.values[~ytest.isna()]
        ytest_predicted_not_nan = ytest_predicted[~ytest.isna()]
        
        rmse, r2, mbe =\
            utils.diagnostic_stats(ytest_not_nan, ytest_predicted_not_nan)
        std = ytest_predicted_not_nan.std()
        corr = np.corrcoef(ytest_not_nan, ytest_predicted_not_nan)[0,1]
        pred_stats = {'rmse':rmse, 
                      'r2':r2, 
                      'mbe':mbe,
                      'std':std,
                      'corr':corr}
        
        return pred_stats
    
if __name__=="__main__":
    import test_config as conf
    import importlib
    importlib.reload(conf)
    
    df = pd.read_csv('../../data_out/Calperum_L3_imputed.csv', parse_dates=['DateTime'])
    dfII = pd.read_csv('../../data_out/Gingin_L3_imputed.csv', parse_dates=['DateTime'])
    
    f = PanelData_fit(conf, df, dfII)
    f._iterate_over_train_test_sets()