import numpy as np
import xgboost as xgb
import pandas as pd
from glob import glob
from collections import Counter
from itertools import izip

df_total_num = pd.read_csv('df_total_num.csv')
df_total_cat = pd.read_csv('df_total_cat.csv')
df_test_num = pd.read_csv('df_test_num.csv')
df_test_cat = pd.read_csv('df_test_cat.csv')
Ytotal = df_total_num['price_doc'].as_matrix()
testId = list(df_test_num['id'])
df_total_num.drop('price_doc', axis = 1, inplace = True)
df_test_num.drop('id', axis = 1, inplace = True)

dall = xgb.DMatrix(df_total_num, label=Ytotal)

hyperparam = {'max_depth': [5],
  'learning_rate': [0.01],
  'n_estimators': [1000],
  'colsample_bytree' :[0.25,0.5,0.75,1],
  'lambda':[100],
  'min_child_weight':[6],
  'scale_pos_weight':[1],
  'max_delta_step':[0]
  #'gamma':[None]
  }
dicts_params = pd.DataFrame(pd.tools.util.cartesian_product([hyperparam[i] for i in hyperparam]),index=hyperparam.keys()).to_dict()
SAVEFILENAME = '0502_house.csv'

#print dicts_params
myseed = 2017
for p in dicts_params:
    par = dicts_params[p]
    print par['max_depth'].astype(int)
    param = {'max_depth': par['max_depth'].astype(int),
    'eta': par['learning_rate'],
    'subsample': 0.8,
    'colsample_bytree': par['colsample_bytree'],
    'lambda': par['lambda'],
    'min_child_weight': par['min_child_weight'],
    'scale_pos_weight': par['scale_pos_weight'],
    'objective':'reg:linear',
    'nthread' : 3,
    'eval_metric' : 'rmse'}
    rlt = xgb.cv(params=param,
    dtrain= dall,
    num_boost_round=par['n_estimators'].astype(int),
    verbose_eval=True,
    nfold=5,
    early_stopping_rounds = 20,
    seed=myseed)
