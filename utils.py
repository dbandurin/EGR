import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime, timedelta
import nav_connect as nv


def get_mh_data(table_name, vins_str, day_min):
  odbc = nv.odbc()
  q = 'select vin, dte, cast(miles as int) as miles, cast(hours as int) as hours \
  from {} where vin in {} and dte>="{}"'.format(table_name, vins_str, day_min)
  df_mh = odbc.read_sql(q)
  return df_mh

def get_eng_bin_data(var, bin_size, vins_str, day_min):
  odbc = nv.odbc()
  vname = 'tq' if var == 'engine_torque_pct' else 'hp'
  q = 'select vin, dte, \
  concat("{}", cast(floor(cast({} as int)/{}) as string)) as bin, \
  sum(bin_percent) as bin_percent_sum \
  from analyticsplayground.occ_dpt_eng_res_map_hp \
  where vin in {} and dte>="{}" \
  and engine_speed_rpm_bin > 900 \
  and engine_torque_pct > 0 and engine_torque_pct <= 120 \
  and avg_hp > 0 and avg_hp < 400 \
  group by vin, dte, bin \
  order by vin, dte, bin'.format(vname, var, bin_size, vins_str, day_min)
  df = odbc.read_sql(q)
  return df

def get_bp_bin_data(var, bin_size, vins_str, day_min):
  odbc = nv.odbc()
  q = 'select vin, dte, \
  concat("bp", cast(floor(cast({} as int)/{}) as string)) as bin, \
  sum(counts) as bin_sum \
  from analyticsplayground.egr_bp_binned \
  where vin in {} and dte>="{}" \
  and bp_bin < 400 \
  group by vin, dte, bin \
  order by vin, dte, bin'.format(var, bin_size, vins_str, day_min)
  df = odbc.read_sql(q)
  return df

def get_all_data(vins_str, day_min = "2020-01-01"):
    #Get Torque data
    odbc = nv.odbc()
    df_tq = get_eng_bin_data('engine_torque_pct', 25, vins_str, day_min)
    df_tq_bin = pd.pivot_table(df_tq, values='bin_percent_sum', index=['vin', 'dte'],
                               columns=['bin'], fill_value=0).reset_index()  
    #Get HorsePower data
    df_hp = get_eng_bin_data('avg_hp', 80, vins_str, day_min)
    df_hp_bin = pd.pivot_table(df_hp, values='bin_percent_sum', index=['vin', 'dte'],
                               columns=['bin'], fill_value=0).reset_index() 
    #Get BackPressure data
    df_bp = get_bp_bin_data('bp_bin', 80, vins_str, day_min)
    df_bp['total'] = df_bp.groupby(['vin','dte'])['bin_sum'].transform(sum)
    df_bp['bin_percent_sum'] = df_bp['bin_sum'] / df_bp['total']
    df_bp_bin = pd.pivot_table(df_bp, values='bin_percent_sum', index=['vin', 'dte'],
                               columns=['bin'], fill_value=0).reset_index()                                                     
    df_bp_bin['dte'] = df_bp_bin['dte'].astype('datetime64')
    
    #Merge all
    df_all_bin = pd.merge(df_bp_bin, df_tq_bin, on=['vin','dte'])
    df_all_bin = pd.merge(df_all_bin,df_hp_bin, on=['vin','dte'])
        
    ### Get mileage, hrs monthly data
    odbc = nv.odbc()
    tname = 'vehicle_analytics.usage_metrics_cleaned'
    df_odo = get_mh_data(tname, vins_str, day_min)
    df_odo = df_odo[['vin','dte','miles','hours']]
    
    df_all_bin = pd.merge(df_all_bin,df_odo, on=['vin','dte'])
    df_all_bin.rename(columns={'dte':'day'},inplace=True)
        
    return df_all_bin.sort_values(['vin','day'])

#Get FC data
def get_fc_data(vins_str, spn_str):
    odbc = nv.odbc()
    q = '''
    select vin, dtc_date, odometer_miles, engine_hours, device_description, 
    sa, spn,fmi, fault_status, fault_severity, fault_description   
    from engineering.cso_occ_hd_merged_v16d_mat 
    where vin in {} 
    and spn in {}
    '''.format(vins_str, spn_str)
    return odbc.read_sql(q)

def fc_preselections(data_fc, event_class, TSP, ndays=90):
    #Keep last 90 days of FC history
    t_diff_min = -ndays
    t_diff_max = 0
    data_fc = data_fc.query('t_diff > {} and t_diff < {}'.format(t_diff_min, t_diff_max))
    
    #Keep just known TSP providers
    data_fc = data_fc[data_fc.device_description.isin(TSP)]
    
    #Limit FCs by miliage
    if event_class == 'Good':
        data_fc = data_fc.query('odometer_miles < 180000')
    elif event_class == 'Bad': 
        data_fc = data_fc.query('odometer_miles >= 50000')
        
    #Filter out Synthetic and None codes
    fault_status = ['ACTIVE','PREVIOUSLY ACTIVE','PENDING','INACTIVE']
    data_fc = data_fc[data_fc.fault_status.isin(fault_status)]
    
    #Keep just sa=0
    sa_to_take = ['0']
    return data_fc[data_fc.sa.isin(sa_to_take)]

def get_fc_counts_by_day(data_fc, df_occ):
    #data_fc = data_fc_bad_sel#.query('vin == "3HSDZTZR0JN527613"')
    dfDummies = pd.get_dummies(data_fc['spn'])
    data = pd.concat([data_fc, dfDummies], axis=1)
    cols = list(dfDummies.columns)
    dft = pd.DataFrame()
    for vin, tmp in data.groupby('vin'):
        days = list(df_occ.query('vin == "{}"'.format(vin)).day)
        for d in days:
            tmp_d = tmp.query('dtc_date <= "{}"'.format(d))
            t_diff = -1*tmp_d.t_diff.values
            w = 0.95**t_diff
            wts = tmp_d[cols].apply(lambda x: sum(w*x), axis=0)
            t = pd.DataFrame([dict(zip(cols,wts))])
            t['vin'] = vin
            t['day'] = d
            dft = dft.append(t)
    return dft, cols  
  