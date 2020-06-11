import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import Counter
from dateutil import parser

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import xgboost
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from random import sample
from mlxtend.plotting import plot_confusion_matrix as conf_mat
#from plot_cm_matrix import plot_confusion_matrix

#Read data (run prepare_data.py first)
data_bad = pd.read_csv('EGR/EGR_model/data/data_bad_egr_hist.csv')
data_good = pd.read_csv('EGR/EGR_model/data/data_good_egr_hist.csv')

#Create bad and good subsamples
ngood = min(len(set(data_good.vin)),160)
nbad = min(len(set(data_bad.vin)),40)
vin_good_train = sample(set(data_good.vin), ngood)
data_good_train = data_good[data_good.vin.isin(vin_good_train)]
vin_bad_train = sample(list(set(data_bad.vin)), nbad)
data_bad_train = data_bad[data_bad.vin.isin(vin_bad_train)]
data_bad_train['Failed'] = 1
data_good_train['Failed'] = 0

### Create data mix
df_good = data_good_train.groupby(['model_mjr','engine_hp'], group_keys=False).\
                          apply(pd.DataFrame.sample, frac=.2, random_state=0)
data_m = pd.concat([data_bad_train, df_good])
print(len(df_good), len(data_bad_train), len(data_m))

data_m['engine_hp_cat'] = data_m['engine_hp'].astype('category')

spn = [x for x in data_m.columns if 'spn' in x.lower()]
bp = [x for x in data_m.columns if 'bp' in x.lower()]
tq = [x for x in data_m.columns if 'tq' in x.lower()]
hp = [x for x in data_m.columns if 'hp' in x.lower() and len(x)==3]

#
#Analysis methods
#
def get_prob(clf,Xd,yd, test=False, logy=True):
    y_prob = clf.predict_proba(Xd)[:,1]
    df_prob = pd.DataFrame({'EngQual':yd.ravel(),'prob':y_prob.ravel()})
    if not test:
        _=plt.hist(df_prob.query('EngQual == 0')['prob'],bins=10,range=(0,1),alpha=0.5,label='Good',normed=True)
        _=plt.hist(df_prob.query('EngQual == 1')['prob'],bins=10,range=(0,1),alpha=0.5,label='Bad',normed=True)
    else:
        _=plt.hist(df_prob.query('EngQual == "Unknown"')['prob'],bins=10,alpha=0.5,label='Test')
    if logy:
        plt.yscale('log')
    plt.legend(fontsize=12)
    _=plt.xlabel('prob of failure',fontsize=12,fontweight='bold')
    _=plt.xticks(fontsize=12,fontweight='bold')
    _=plt.yticks(fontsize=12,fontweight='bold')
    plt.show()
    return y_prob

def get_results(m, Xd, yd, rolling=False):
    y_pred = m.predict(Xd)
    
    if rolling:
        Xd['prob'] = m.predict_proba(Xd)[:,1]
        y_pred = Xd.groupby('vin').apply(lambda tmp: tmp['prob'].rolling(7, min_periods=1).mean().round(0)).values
        yd = Xd.groupby('vin').apply(lambda tmp: tmp['Failed'])
    
    print(classification_report(yd, y_pred))
    cm = confusion_matrix(yd, y_pred)
    print(cm)

#
#Models
#
def run_model(rs, df_data, numeric_features, categorical_features, test_size=0.3, rolling=False, show=True):
    def preprocess():
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        return preprocessor
    
    #Target and input data
    y_class = df_data['Failed']
    X = df_data[numeric_features + categorical_features + ['Failed','vin']]
    print('Counter(y_class): ',Counter(y_class))

    #Split into train and test samples
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_class, test_size=test_size, 
                                                          shuffle=True, random_state=rs)
    
    print('X_train:',len(X_train.query('Failed == 0')), len(X_train.query('Failed == 1')))
    print('X_valid:',len(X_valid.query('Failed == 0')), len(X_valid.query('Failed == 1')))
        
    # Class weights
    class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(X_train['Failed']),
                                             X_train['Failed']))
    bins = np.unique(X_train['Failed'])
    dw = {}
    for ix,v in enumerate(bins):
        dw[v] = class_weights[ix]

    #Models
    nb  = Pipeline(steps=[('preprocessor', preprocess()),
                      ('classifier', BernoulliNB()) ])
        
    xgb = Pipeline(steps=[('preprocessor', preprocess()),
                      ('classifier', xgboost.XGBClassifier(
                        #  max_depth=3,
                        #  gamma=1,
                        )) ])

    svc = Pipeline(steps=[('preprocessor', preprocess()),
                      ('classifier',  svm.SVC(kernel='rbf', gamma='auto', 
                                              probability=True, class_weight=dw)) ]) 

    ann = Pipeline(steps=[('preprocessor', preprocess()),
                      ('classifier', MLPClassifier(hidden_layer_sizes=(50,25,25), 
                                     max_iter=300, activation = 'relu',solver='adam')) ])

    mlp =  MLPClassifier()
    params_mlp = {'hidden_layer_sizes': [(100,50,50),(50,25,25),(100,50),(50,25)],
                  'activation': ['relu','softmax']}
    ann_opt = Pipeline(steps=[('preprocessor', preprocess()),
                      ('classifier', GridSearchCV(mlp, params_mlp, cv=3)) ])

    # Create Ensemble
    #create a dictionary of our models
    estimators=[('xgb', xgb), ('svc', svc), ('ann', ann_opt), ('nb', nb)]
    #create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft',weights=[1,1,1,1])
    
    #Train
    nb.fit(X_train[numeric_features + categorical_features], y_train)
    xgb.fit(X_train[numeric_features + categorical_features], y_train)
    svc.fit(X_train[numeric_features + categorical_features], y_train)
    ann.fit(X_train[numeric_features + categorical_features], y_train)
    ann_opt.fit(X_train[numeric_features + categorical_features], y_train)
    ensemble.fit(X_train[numeric_features + categorical_features], y_train)
    print('ANN opt:',ann_opt['classifier'].best_params_)
    
    #Test   
    if show:
        print('\nNB:')
        get_results(nb, X_valid, y_valid,rolling)
        print('\nANN:')
        get_results(ann_opt, X_valid, y_valid,rolling)
        #print('\nANN-opt:')
        #get_results(ann_opt, X_valid, y_valid)
        print('\nSVM:')
        get_results(svc, X_valid, y_valid,rolling)
        print('\nXGB:')
        get_results(xgb, X_valid, y_valid,rolling)  
        print('\nEnsemble:')
        get_results(ensemble, X_valid, y_valid,rolling) 
    
    #return X_train, y_train, X_valid, y_valid, xgb, ensemble   
    return ensemble
    
def make_data(model):
    numeric_features = ['miles','hours'] + bp + tq + spn 
    categorical_features = ['model_mjr','engine_hp_cat'] 
    if model == 'm1':
        data = data_m.dropna(subset=numeric_features)
    elif model == 'm2':
        numeric_features = bp + tq + spn
        data = data_m.dropna(subset=numeric_features)
    
    return data, numeric_features,categorical_features

df_data, numeric_features, categorical_features = make_data('m1')
model = run_model(1, df_data, numeric_features, categorical_features, test_size=0.2, rolling=True)


#
# Test data
#
def predict_bad_egr_vins(data_test, thres=0.7):
    data_test = data_test.sort_values(['vin','day'])
    bad_vins = []
    miles = []
    probs = []
    for vin, tmp in data_test.groupby('vin'):
        if len(tmp)<1:
            continue
        tmp['prob_roll_ewm'] = tmp[['prob']].ewm(span=7, min_periods=1).mean()
        prob_last3 = np.mean(tmp['prob_roll_ewm'].iloc[-3:])
        if prob_last3 >= thres:
            probs.append(prob_last3)
            bad_vins.append(vin)
            miles.append(tmp.miles.iloc[-1])
    data_bad_egr = pd.DataFrame({'vin':bad_vins,'prob':probs,'miles':miles})
    return data_bad_egr

data_bad_egr = pd.DataFrame()
for ifile in range(1,16):
  f = 'EGR/EGR_model/data/data_all_egr_hist_{}.csv'.format(ifile)
  if not os.path.exists(f):
    continue
    
  data_test = pd.read_csv(f)
  data_test['engine_hp_cat'] = data_test['engine_hp'].astype('category')
  data_test['fail_date'] = 'Unknown'

  X_test = data_test[numeric_features+categorical_features]
  data_test['prob'] = model.predict_proba(X_test)[:,1]
  #data_test['label'] =  model.predict(X_test)

  tmp = predict_bad_egr_vins(data_test, 0.8)
  print('ifile, #vins:',ifile,len(tmp))
  data_bad_egr = data_bad_egr.append(tmp)

print('Total #vins:',len(data_bad_egr))
  
#Store data
data_bad_egr.to_csv('EGR/EGR_model/data/data_bad_egr.csv')

print('All done.')










