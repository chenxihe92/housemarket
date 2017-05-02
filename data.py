import numpy as np
import xgboost as xgb
import pandas as pd
import glob
from collections import Counter
#from math import isnan
from features_house import house_basic

filedir = '/Users/hechenxi/Desktop/kaggle/'

econ = pd.read_csv(filedir + 'macro.csv', sep=",", encoding='utf-8')
train = pd.read_csv(filedir + 'train.csv', sep=",", encoding='utf-8')
test = pd.read_csv(filedir + 'test.csv', sep=",", encoding='utf-8')

for i in range(10):
    print house_basic(train, i+1)

#x = train['timestamp'].iloc[0]
#y = train['build_year'].iloc[0]

#print x[0:4], float((x[0:4]))

#"""
for i in train.columns:
    print i
    #print train[i].iloc[0:4]
    #print train[i].iloc[0] != train[i].iloc[0]
    #print "\n\n"

print "shape train", np.shape(train)
print "shape econ", np.shape(econ)

#print train.head(5)
print "floor", train[u'floor'].unique().tolist()
print "material", train.material.unique().tolist()
print "state", train.state.unique().tolist()
print "product_type", train.product_type.unique().tolist(), Counter(train.product_type.tolist())
print "sub_area", train.sub_area.unique().tolist()
#print train.sub_area.unique().tolist()
#"""
#print econ.columns
#print econ.head(5)
#print econ.gdp_quart.iloc[0], type(econ.gdp_quart.iloc[0]) #nan <type 'numpy.float64'>

#x = econ.gdp_quart.iloc[0]

#print x == x, x is None

#print econ.timestamp.iloc[0], type(econ.timestamp.iloc[0]) 2010-01-01 <type 'unicode'>
#print econ.timestamp.unique().tolist()
for i in test.columns:
    print i, test[i].iloc[0]