import pandas as pd
import numpy as np
import nav_connect as nv
import pickle
import time
from datetime import datetime, timedelta
from dateutil import parser
import EGR.EGR_model.utils as utils

#EGR FCs
EGR_FC = ['SPN2791','SPN2659','SPN3251','SPN3216','SPN3226','SPN4765','SPN4364','SPN637',
          'SPN102','SPN132','SPN188','SPN1761']

#TSPs
df_tsp = pd.read_excel('~/EGR/data/Telematic devices.xlsx')
TSP = df_tsp.Ameritrak.values

#Get all A26 data
odbc = nv.odbc()
q = '''select * from analyticsplayground.a26_vins_with_esn'''
df_vins = odbc.read_sql(q)
df_vins['model_mjr'] = df_vins.mdl_7_cd.apply(lambda x: x[:2])
all_vins = list(set(df_vins.vin.values))

#Confirmed EGR Failures
df_fails = pd.ExcelFile('~/EGR/data/EGR D.xlsx').parse('Claim comments')
df_fails = df_fails[(df_fails['Failure Mode'] == 'Bushing Wear')  
              | (df_fails['Failure Mode'] == 'Sticking EGR Valve') 
              | (df_fails['Failure Mode'] == 'Pin/Butterfly Broken') 
              | (df_fails['Failure Mode'] == 'Broken Pin/Butterfly and High Bushing wear')]

#Fails dictionary
d_fails = df_fails.set_index('VIN')['Fail Date'].to_dict()

# bad vins
vins_bad = list(d_fails.keys())

# good vins (All A26 that have DIS<365 and not failed)
today = datetime.now() - timedelta(days=1)
#today = parser.parse('2020-03-31')
  
df_vins['DIS'] = df_vins['trk_build_date'].apply(lambda x: (today - x).days) 
df_good = df_vins.query('DIS < 365')
vins_good = list(set(df_good.vin) - set(vins_bad))

#Form strings for DB requests
vins_str_bad = str(tuple(vins_bad)).replace('\'','\"')
vins_str_good = str(tuple(vins_good)).replace('\'','\"')
spn_str = str(tuple(EGR_FC)).replace('\'','\"')

#Get FC data for failed and good vins 
df_fc_bad  = utils.get_fc_data(vins_str_bad, spn_str)
df_fc_good = utils.get_fc_data(vins_str_good, spn_str)

#Get time difference (in days) to failure. For good vins, it is time to 'today'.
df_fc_bad['t_diff']  = df_fc_bad.apply(lambda r: (parser.parse(r['dtc_date']) - d_fails[r['vin']]).days, axis=1)
df_fc_good['t_diff'] = df_fc_good.dtc_date.apply(lambda x: (parser.parse(x) - today).days)

#Make qualitative preselections for good and bad vins
ndays_occ_back = 15
ndays = 90 + ndays_occ_back
data_fc_bad_sel  = utils.fc_preselections(df_fc_bad,  'Bad',  TSP, ndays)
data_fc_good_sel = utils.fc_preselections(df_fc_good, 'Good', TSP, ndays)
print('len(data_fc_bad_sel), len(data_fc_good_sel)', len(data_fc_bad_sel), len(data_fc_good_sel))

### Get OCC data 
#
def get_test_data(vins_str, data_fc_sel, day_min, data_type):   
    df_data = utils.get_all_data(vins_str, day_min)
    
    if data_type == 'Bad':
      df_data['t_diff'] = df_data.apply(lambda r: (r['day'] - d_fails[r['vin']]).days, axis=1)
    else:
      df_data['t_diff'] = df_data.apply(lambda r: (r['day'] - today).days, axis=1)
    
    #Keep just OCC data ndays_occ_back
    df_data = df_data.query('t_diff >= {} and t_diff <= {}'.format(-ndays_occ_back, 0))
        
    data_m = pd.merge(df_data, df_vins[['vin',
                               'mdl_7_cd',
                               'model_mjr',
                               'engine_hp',
                               'trans_fwd_gears',
                               'def_tanks_gallons',
                              ]], on ='vin')
    
    #Get weighted FC counts for each day of OCC data back in time. 
    df_fc, cols_fc = utils.get_fc_counts_by_day(data_fc_sel, data_m)
    #Merge FC and OCC data
    data_mm = pd.merge(data_m, df_fc, on=['vin','day'], how='left')
    #Fill NA FC data with zeros
    data_mm[cols_fc] = data_mm[cols_fc].fillna(value=0)

    return data_mm
  
#  
#Bad vins first
#
odbc = nv.odbc()
t0 = time.time()
date_min_bad = "2020-01-01"
data_bad_mm = get_test_data(vins_str_bad, data_fc_bad_sel, date_min_bad, 'Bad')
print('Get all bad data:',time.time() - t0, len(data_bad_mm))

#Store results
data_bad_mm.to_csv('EGR/EGR_model/data/data_bad_egr_hist.csv',index=False)

#
#Now select good vins
#
odbc = nv.odbc()
t0 = time.time()
#Get good train data
N_good_train = 2000
date_min_good = (datetime.now() - timedelta(days=ndays_occ_back)).strftime("%Y-%m-%d") 
#date_min_good = "2020-03-01"
good_vins_str = str(tuple(vins_good[:N_good_train])).replace('\'','\"')
data_good_mm = get_test_data(good_vins_str, data_fc_good_sel, date_min_good, 'Good')

print('Get all good data:',time.time() - t0, len(data_good_mm))

#Store results
data_good_mm.to_csv('EGR/EGR_model/data/data_good_egr_hist.csv',index=False)


#
#Create Test sample
#
vins_test = list(set(all_vins) - set(vins_good) - set(vins_bad))

date_min_test = "2020-05-01"
n = len(vins_test)
step = 1000
n1 = 0
n2 = n1+step
i = 1
t0 = time.time() 
while n2-step < n:
    print(n1, n2)
    odbc = nv.odbc()
    vins_str_test = str(tuple(vins_test[n1:n2])).replace('\'','\"')
    df_fc_test  = utils.get_fc_data(vins_str_test, spn_str)
    df_fc_test['t_diff'] = df_fc_test.dtc_date.apply(lambda x: (parser.parse(x) - today).days)
    data_fc_test_sel = utils.fc_preselections(df_fc_test, 'Test', TSP, ndays)
    data_all_mm = get_test_data(vins_str_test, data_fc_test_sel, date_min_test, 'Test')
    print('Get all test data:',time.time() - t0, len(data_all_mm))
    data_all_mm.to_csv('EGR/EGR_model/data/data_all_egr_hist_{}.csv'.format(i), index=False)
    n1, n2 = n2, n2+step
    i += 1 






















