import numpy as np
import xgboost as xgb
import pandas as pd
from glob import glob
from collections import Counter
from itertools import izip
#import itertools

def isnan(a):

    if a != a:
        return True
    else:
        return False


def house_basic(df, id):

    rlt = {}
    df_tmp = df[df.id == id]

    rlt['price_doc'] = df_tmp['price_doc'].iloc[0]
    rlt['full_sq'] = df_tmp['full_sq'].iloc[0]
    rlt['life_sq'] = -1 if isnan(df_tmp['life_sq'].iloc[0]) else df_tmp['life_sq'].iloc[0]
    rlt['floor'] = -1 if isnan(df_tmp['floor'].iloc[0]) else df_tmp['floor'].iloc[0]
    rlt['max_floor'] = -1 if isnan(df_tmp['max_floor'].iloc[0]) else df_tmp['max_floor'].iloc[0]
    rlt['material'] = -1 if isnan(df_tmp['material'].iloc[0]) else df_tmp['material'].iloc[0]
    # build_year
    if isnan(df_tmp['build_year'].iloc[0]):
        rlt['build_year_totransaction'] = -1
    else:
        rlt['build_year_totransaction'] = (df_tmp['timestamp'].iloc[0][0:4]) - (df_tmp['build_year'].iloc[0])

    rlt['num_room'] = -1 if isnan(df_tmp['num_room'].iloc[0]) else df_tmp['num_room'].iloc[0]
    rlt['kitch_sq'] = -1 if isnan(df_tmp['kitch_sq'].iloc[0]) else df_tmp['kitch_sq'].iloc[0]
    rlt['state'] = -1 if isnan(df_tmp['state'].iloc[0]) else df_tmp['state'].iloc[0]

    # product_type
    rlt['product_type_Investment'] = 0
    rlt['product_type_OwnerOccupier'] = 0
    if df_tmp['product_type'].iloc[0] != df_tmp['product_type'].iloc[0]:
        rlt['product_type_Investment'] = -1
        rlt['product_type_OwnerOccupier'] = -1
    else:
        if df_tmp['product_type'].iloc[0] == u'Investment':
            rlt['product_type_Investment'] = 1
        if df_tmp['product_type'].iloc[0] == u'OwnerOccupier':
            rlt['product_type_Investment'] = 1

    return rlt


def house_same_area(df, area, tranc_year, id):

    rlt = {}
    df_tmp = df[(df.sub_area == area) & (df.id != id)]
    df_tmp['year'] = [i[0:4] if not isnan(i) else 3000 for i in df_tmp.timestamp.tolist()]

    price_current_year = []
    price_within_3years = []
    price_within_5years = []

    #for i,j in izip

    return 0