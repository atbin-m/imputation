import data_preparation
import pandas as pd
from train_test_split import groups_of_train_test_set
import model_fit
import panel_data
import plot_utils
import matplotlib.pyplot as plt
import prophet

class ModelPipeline(object):
    def __init__(self, conf):
        self.conf = conf
        self.primary_st = self.conf.data['tower']
        
    def imputation_run(self):
        # ------- Configs
        data_config, var_config =  self.conf.data, self.conf.variables
        print('Imputing tower: {}'.format(self.conf.data['tower']))        
        # ------ Data Preprocessing
        df = data_preparation.data_preprocessing(data_config, var_config)

        # ------ Train test split
        test_df, train_df = groups_of_train_test_set(df, conf)

        # ------ Model fitting
        modelf = model_fit.Model_runner(conf, test_df, train_df)
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
        test_df, train_df = groups_of_train_test_set(df_2nd_tower, conf)
        
        # ------ Model fitting
        modelf = model_fit.Model_runner(conf, test_df, train_df)
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
        summary_df = self._read_summary_file()

        ttest_min_set = summary_df['Test_begin'].unique()

        for t in ttest_min_set:
            a_summary_df = summary_df[summary_df['Test_begin']==t]
            self._make_save_taylor_diagram(a_summary_df, t)        

    def _make_save_taylor_diagram(self, a_summary_df, ttest_min):
        
        sample = a_summary_df[['std', 'Corr.', 'Solver']].values
        refstd = 0.75*sample[:,0].min()
        srange = (0, 1.5*sample[:,0].astype(float).max()/refstd)

        ttest_min = pd.to_datetime(str(ttest_min)) 
        ttest_min =  ttest_min.strftime("%Y-%m-%d")
        
        title = self.primary_st + '_' +  ttest_min
        fn = '../../plot/' + title

        plot_utils.taylor_diagram(sample,  refstd, srange, title)    

        if self.conf.result['save_plots'] == True:
            plt.savefig(fn + '.jpeg')
            plt.close()

        pass
    
    def _read_summary_file(self):
        title = self.primary_st
        fn = '../../data_out/' + title + '_summary_stats.csv'
        full_summary = pd.read_csv(fn, parse_dates=['Test_begin', 'Test_end'])
        
        return full_summary
    
    
if __name__=="__main__":
    """
    there are two configuration files that need to be checked everytime
    a. test_config
    b. test_list_of_models
    """
    import importlib
    import sys
    sys.path.insert(0, 'configs')
    
    # ------- L3 --------------------------
    import test_config as conf
    importlib.reload(conf)

    p = ModelPipeline(conf)
    p.imputation_run()
    # if conf.data['PanelData']==False:
    #      p.imputation_run_sec_tower()
    #      p.panel_data_run()
    if conf.data['fbprophet'] == True:
        p.fbprophet_run()
    p.taylor_diagram()

    # ------- L4 --------------------------
    """
    import config_fluxes as conf
    importlib.reload(conf)

    p = ModelPipeline(conf)
    p.imputation_run()
    if conf.data['PanelData']==True:
         p.imputation_run_sec_tower()
         p.panel_data_run()
    if conf.data['SOLO'] == True:
        p.solo_run()

    p.taylor_diagram()
    """
    