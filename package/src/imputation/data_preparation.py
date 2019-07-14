# -*- coding: utf-8 -*-

import nc_read
import numpy as np
import pandas as pd
import logging
logging.basicConfig(format="%(lineno)s:%(funcName)s:%(message)s", level=logging.INFO)

def _primary_data_preprocessing(data_config, vars_config, data_in_path=None):
    """
    Loads and read file from ancillary and main files.
    """    
    # File configs
    if not data_in_path:
        data_in_path = "../../data_in/"    
    file_head = data_config['tower'] + '_'
    yobs_file = data_config['yobs_file']
    ancillary_files = data_config['ancillary_files']

    # Data variable configs
    yvar, tvar = vars_config['yvar'], vars_config['tvar']
    
    ds = nc_read.nc_read_series(data_in_path + file_head + yobs_file + ".nc",
                                checktimestep=True,
                                fixtimestepmethod="r")
    
    logging.info("Loading file: {}".format(data_in_path + file_head + 
                 yobs_file + ".nc"))

   
    #df = pd.DataFrame({yvar: ds.series[yvar]['Data'],
    #                   tvar: ds.series[tvar]['Data']})
    
    df = pd.DataFrame({yvar: ds.series[yvar]['Data'],
                       tvar: ds.series[tvar]['Data'], 
                       'ustar': ds.series['ustar']['Data']})

    # Adding ustar_threshold information in the dataframe   
    if data_config['ustar'] == True:        
        df.ustar.replace({-9999.0: np.nan}, inplace=True)  
                
        for year, uvalue_threshold in data_config['ustar_map'].items():
            year_cut = (df[tvar].dt.year == year) 
            
            ustar_exclude_hour = data_config['ustar_exclude_hour']
            hour_cut = ((df[tvar].dt.hour > ustar_exclude_hour['end']) & 
                        (df[tvar].dt.hour < ustar_exclude_hour['begin']))

            ustar_cut = (df['ustar'] < uvalue_threshold)
            
            # Emptying yvar where yvar < ustar_threshold             
            df.loc[ustar_cut & year_cut & ~hour_cut, yvar] = -9999.0
            
    # Run when a list of ancillary files are provided
    if len(ancillary_files)>0:
            
        for i in ancillary_files:
            logging.info("Loading file: {}".format(data_in_path + 
                         file_head + i + '.nc'))
            
            temp_ds = nc_read.nc_read_series(data_in_path + file_head + i + '.nc',
                                             checktimestep=True,
                                             fixtimestepmethod="")
            if i=='ACCESS':
                yvars = ['%s_00'%yvar, '%s_01'%yvar, '%s_02'%yvar, '%s_10'%yvar, 
                        '%s_11'%yvar, '%s_12'%yvar, '%s_20'%yvar, '%s_21'%yvar, 
                        '%s_22'%yvar]
                
            elif i=='AWS':
                yvars = ['%s_0'%yvar, '%s_1'%yvar, '%s_2'%yvar]
                
            elif i=='BIOS2':
                yvars = ['%s'%yvar]
                            
            time = temp_ds.series[tvar]['Data']
            
            temp_Ah = {tvar: time}
            
            for j in yvars:
                temp_Ah[i+'_'+j] = temp_ds.series[j]['Data']
    
            temp_df = pd.DataFrame(temp_Ah)    
    
            df = pd.merge(df, temp_df, how='left', on=tvar, 
                          validate='one_to_one')
    # Run when no ancillary file is provided, e.g. in the case of L4
    else:
        xvars = vars_config['xvar']
        
        for j in xvars:
            df[j] = ds.series[j]['Data']

    df.replace({-9999.0: np.nan}, inplace=True)    

    return df

def _climatology_data_preprocessing_single_sheet(data_config, sheet_name, 
                                                 data_in_path=None):
    tower_name = data_config['tower']
    yobs_file = data_config['yobs_file']
    
    if not data_in_path:
        data_in_path = "../../data_in/"    

    climatology_file = data_in_path + '%s_%s_Climatology.xls'%(tower_name, yobs_file)
    logging.info("Reading climatology data from file {}".format(climatology_file))
    df = pd.read_excel(climatology_file, sheet_name=sheet_name)

    # Time in floating point 0.5-24
    column_header = df.columns[1:].values

    # Time in hh:mm:ss format
    column_header_in_hms = []
    for i in column_header:
        hhmm = str(i).split('.')

        if len(hhmm)<2:
            if hhmm[0] == '24':
                hh = '00'
            else:
                hh = "%02d"%int(hhmm[0])
            mm = ':00'
        else:
            hh="%02d"%int(hhmm[0])
            mm = ':30'        

        column_header_in_hms.append(hh+mm+':00')

    # Date
    row_header = df['Day'].dt.strftime('%Y-%m-%d').values

    # Merging date and time to create a full timestamp
    t = []
    for date in row_header:
        for time in column_header_in_hms:
            t.append(date+' '+time)

    assert len(t)  == df[column_header].size
    
    # Reshaping matrix wide to long
    y = df[df.columns[1:]].values.reshape(df[df.columns[1:]].size)
    
    # Mapping time to data into dataframe
    single_df = pd.DataFrame(list(zip(t,y)), columns=['DateTime', sheet_name])
    
    single_df.loc[:,'DateTime'] = pd.to_datetime(single_df['DateTime'])
    
    return single_df

def _climatology_data_preprocessing_all_sheets(data_config):
    list_of_sheet_names = data_config['Climatology_xvars']
    
    df = {}
    for j, sheet_name in enumerate(list_of_sheet_names):
        temp_df = _climatology_data_preprocessing_single_sheet(data_config, sheet_name)
        
        if j==0:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on='DateTime')
            
        assert df.shape, temp_df.shape
    return df

def _minutes_in_year(t):
    # First day of the year
    fday = pd.to_datetime(t.dt.strftime('%Y'))
    tdiff_in_minutes = (t-fday).astype('timedelta64[m]')
    
    days_in_year = fday.apply(lambda x: 366 if x.is_leap_year else 365)
    minutes_in_year = days_in_year*24*60.
    
    theta = 2*np.pi*(tdiff_in_minutes/minutes_in_year)
    
    sinwt = np.sin(theta)
    coswt = np.cos(theta)
    
    return sinwt, coswt

def data_preprocessing(data_config, vars_config, 
                     data_in_path=None,
                     data_out_path = None):

    primary_df = _primary_data_preprocessing(data_config, vars_config,
                                             data_in_path)

    climatology = data_config['Climatology']
    if climatology == True:
        climatology_df = _climatology_data_preprocessing_all_sheets(data_config,
                                                                    data_in_path)

        date_overlap_primary_climate = len(set(primary_df.DateTime.values) - 
                                           set(climatology_df.DateTime.values))
        if date_overlap_primary_climate!=0:
            print("{} number of DateTime present in primary_df but missing in climatology_df".format(date_overlap_primary_climate))
            raise ValueError('For some dates in primary_df there are no equivalent dates in climatology_df.')
        
        df = pd.merge(primary_df, climatology_df, how='left', 
                      left_on=vars_config['tvar'], 
                      right_on=vars_config['tvar'])
    else:
        df = primary_df
        
    yobs_file = data_config['yobs_file']
 
    tfeatures = vars_config['xvar_derived']
    tvar = df[vars_config['tvar']]
    for t in tfeatures:
        if t=='day':
            df[t] = tvar.dt.day
        elif t=='dayofyear':
            df[t] = tvar.dt.dayofyear
        elif t=='month':
            df[t] = tvar.dt.month
        elif t=='week':
            df[t] = tvar.dt.week
        elif t=='hour':
            df[t] = tvar.dt.hour
        elif t=='minutes':
            df[t] = tvar.dt.minute
        elif t=='sinwt':
            sinwt, coswt = _minutes_in_year(tvar)
            df['sinwt'] = sinwt
            df['coswt'] = coswt

    if not data_out_path:
        data_out_path = "../../data_out/"

    df.to_csv(data_out_path + data_config['tower'] + '_' + yobs_file +
              '_processed.csv', index=False)
    
    logging.info("Processed file saved at : {}".format("../../data_out/" + 
                 data_config['tower'] + '_' + yobs_file + '_processed.csv'))

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'configs')
    import test_config as conf
    #import config_fluxes as conf

    import importlib
    importlib.reload(conf)
    data_config, var_config =  conf.data, conf.variables
    
    df = data_preprocessing(data_config, var_config)