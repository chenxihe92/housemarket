import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
#from sklearn.model_selection import cross_val_score, train_test_split

filedir = '/Users/hechenxi/Desktop/kaggle/'
outputdir = '/Users/hechenxi/Desktop/kaggle/output/'

macro = pd.read_csv(filedir + 'macro.csv', sep=",", encoding='utf-8')
total = pd.read_csv(filedir + 'train.csv', sep=",", encoding='utf-8')
test = pd.read_csv(filedir + 'test.csv', sep=",", encoding='utf-8')

df_total = pd.merge(total, macro, on='timestamp', how='left')
df_total.drop('id', axis = 1, inplace = True)
df_total['price_doc'] = np.log1p(df_total['price_doc'])

df_test = pd.merge(test, macro, on='timestamp', how='left')
df_test.drop('id', axis = 1, inplace = True)
df_all = pd.concat([df_total,df_test], keys = ['total','test'])


def missingPattern(df):
    numGroup = list(df._get_numeric_data().columns)
    catGroup = list(set(df.columns) - set(numGroup))
    print 'Total categorical/numerical variables are %s/%s' % (len(catGroup), len(numGroup))

    # missing data
    n = df.shape[0]
    count = df.isnull().sum()
    percent = 1.0 * count / n
    dtype = df.dtypes
    # correlation
    missing_data = pd.concat([count, percent, dtype], axis=1, keys=['Count', 'Percent', 'Type'])
    missing_data.sort_values('Count', ascending=False, inplace=True)
    missing_data = missing_data[missing_data['Count'] > 0]
    print 'Total missing columns is %s' % len(missing_data)

    return numGroup, catGroup, missing_data

numGroup, catGroup, missing_data = missingPattern(df_all)

high_missing_data = missing_data[missing_data['Percent'] > 0.5]

import operator
def getCorr(df, numGroup, eps, *verbose):
    corr = df[numGroup].corr()
#     plt.figure(figsize=(8, 6))
#     plt.pcolor(corr, cmap=plt.cm.Blues)
#     plt.show()
    corr.sort_values(["price_doc"], ascending = False, inplace = True)
    highCorrList = list(corr.price_doc[abs(corr.price_doc)>eps].index)
    if verbose:
        print "Find most important features relative to target"
        print corr.price_doc[abs(corr.price_doc)>eps]
    return corr, highCorrList
corr, highCorrList = getCorr(df_all.ix['total',:], numGroup, 0.4, True)


for i in high_missing_data.index:
    df_all.drop(i, axis=1, inplace=True)

print 'all: ', df_all.shape


basic_missing = list((set(missing_data.index) - set(high_missing_data.index)) & set(total.columns))
macro_missing = list((set(missing_data.index) - set(high_missing_data.index)) & set(macro.columns))
print 'missing in basic: ', len(basic_missing)
print 'missing in macro: ', len(macro_missing)

df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all['year'] = df_all.timestamp.dt.year
df_all['year_month'] = df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100

# life_sq and full_sq are highly related to price_doc
# life_sq <= full_sq and full_sq has no missing value

# life_sq or full_sq <= 5
df_all['life_sq'][df_all['life_sq']<=5] = df_all['full_sq'][df_all['life_sq']<=5]
df_all['full_sq'][df_all['full_sq']<=5] = df_all['life_sq'][df_all['full_sq']<=5]


# # life_sq or full_sq > 200
df_all['life_sq'].ix['total'][1084] = 28.1
df_all['life_sq'].ix['total'][4385] = 42.6
df_all['life_sq'].ix['total'][9237] = 30.1
df_all['life_sq'].ix['total'][9256] = 45.8
df_all['life_sq'].ix['total'][9646] = 80.2
df_all['life_sq'].ix['total'][13546] = 74.78
df_all['life_sq'].ix['total'][13629] = 25.9
df_all['life_sq'].ix['total'][21080] = 34.9
df_all['life_sq'].ix['total'][26342] = 43.5

df_all['life_sq'].ix['test'][601] = 74.2
df_all['life_sq'].ix['test'][1896] = 36.1
df_all['life_sq'].ix['test'][2031] = 23.7
df_all['life_sq'].ix['test'][2791] = 86.9
df_all['life_sq'].ix['test'][5187] = 28.3

df_all['full_sq'].ix['total'][1478] = 35.3
df_all['full_sq'].ix['total'][1610] = 39.4
df_all['full_sq'].ix['total'][2425] = 41.2
df_all['full_sq'].ix['total'][2780] = 72.9
df_all['full_sq'].ix['total'][3527] = 53.3
df_all['full_sq'].ix['total'][5944] = 63.4
df_all['full_sq'].ix['total'][7207] = 46.1


# life_sq > full_sq
df_all['life_sq'][df_all.life_sq > df_all.full_sq] = df_all['full_sq'][df_all.life_sq > df_all.full_sq]

# kitch_sq > full_sq

df_all['kitch_sq'][df_all.kitch_sq > df_all.full_sq] = \
            df_all['full_sq'][df_all.kitch_sq > df_all.full_sq] - df_all['life_sq'][df_all.kitch_sq > df_all.full_sq]


# else
# floor > max_floor
df_all['max_floor'][df_all.floor > df_all.max_floor] = \
        df_all['floor'][df_all.floor > df_all.max_floor] + df_all['max_floor'][df_all.floor > df_all.max_floor]


# fill the missing value in train and test
def basicmissingFill(df):
    # num variables
    # pre-processing
    n = df.shape[0]

    df_all['life_sq'][df_all.life_sq.isnull()] = df_all['full_sq'][df_all.life_sq.isnull()]

    df['state'] = df['state'].replace({33: 3})
    df['build_year'][df['build_year'] == 20052009] = 2005
    df['build_year'][df['build_year'] == 4965] = float('nan')
    df['build_year'][df['build_year'] == 0] = float('nan')
    df['build_year'][df['build_year'] == 1] = float('nan')
    df['build_year'][df['build_year'] == 3] = float('nan')
    df['build_year'][df['build_year'] == 71] = float('nan')
    df['build_year'][df['build_year'] == 20] = 2000
    df['build_year'][df['build_year'] == 215] = 2015
    df['build_year'].ix['total'][13117] = 1970

    # zero-filling count feature
    zero_fil = ['build_count_brick', 'build_count_block', 'build_count_mix', 'build_count_before_1920', \
                'build_count_1921-1945', 'build_count_1946-1970', 'build_count_1971-1995', 'build_count_after_1995', \
                'build_count_monolith', 'build_count_slag', 'build_count_wood', 'build_count_panel',
                'build_count_frame', \
                'build_count_foam', 'preschool_quota']
    for i in zero_fil:
        df[i] = df[i].fillna(0)

    # mode-filling: count feature and ID
    mode_fil = ['state', 'ID_railroad_station_walk', 'build_year', 'material', 'num_room']
    for i in mode_fil:
        df[i] = df[i].fillna(df[i].mode()[0])

        # mean-filling
    mean_fil = ['cafe_avg_price_500', 'cafe_avg_price_1000', 'cafe_avg_price_1500', 'cafe_avg_price_2000', \
                'cafe_avg_price_3000', 'cafe_avg_price_5000', 'cafe_sum_500_max_price_avg',
                'cafe_sum_500_min_price_avg', \
                'cafe_sum_1000_max_price_avg', 'cafe_sum_1000_min_price_avg', 'cafe_sum_1500_max_price_avg', \
                'cafe_sum_1500_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_sum_2000_min_price_avg', \
                'cafe_sum_3000_max_price_avg', 'cafe_sum_3000_min_price_avg', 'cafe_sum_5000_max_price_avg', \
                'cafe_sum_5000_min_price_avg', 'railroad_station_walk_min', 'railroad_station_walk_km', \
                'school_quota', 'raion_build_count_with_material_info', 'prom_part_5000', \
                'raion_build_count_with_builddate_info', 'green_part_2000', 'metro_km_walk', 'metro_min_walk', \
                'hospital_beds_raion']
    for i in mean_fil:
        grouped = df[['year', i]].groupby('year')
        df[i] = grouped.transform(lambda x: x.fillna(x.mean()))

    # exception: 'kitch_sq','floor','max_floor'
    df['kitch_sq'][df.kitch_sq.isnull()] = df['full_sq'][df.kitch_sq.isnull()] - df['life_sq'][df.kitch_sq.isnull()]
    df['floor'] = df['floor'].fillna(df['floor'].mean())
    df['max_floor'][df.max_floor.isnull()] = df['floor'][df.max_floor.isnull()]

    # ================
    # Cat. variables
    df['product_type'] = df['product_type'].fillna(df['product_type'].mode()[0])

    return df


df_all = basicmissingFill(df_all)


macro_missing_obj = []
for i in macro_missing:
    if df_all[i].dtype == object:
        grouped = df_all[['year',i]].groupby(['year',i])
        print grouped.agg(len)
        macro_missing_obj.append(i)
        print missing_data.ix[i]
        print '\n'
# consider to drop macro_missing_obj
for i in macro_missing_obj:
    df_all.drop(i, axis = 1, inplace = True)
    macro_missing.remove(i)

print 'macro missing features count: ', len(macro_missing)
print 'df_all shape: ', df_all.shape


def macromissingFill(df):
    for i in macro_missing:
        fill2014 = np.nanmean(df[i][df['year'] == 2014])
        fill2015 = np.nanmean(df[i][df['year'] == 2015])
        # income_per_cap: the only macro_missing feature which is not agg by year
        if ~np.isnan(fill2015):
            df[i] = df[i].fillna(fill2015)
        else:
            df[i] = df[i].fillna(fill2014)

    return df


df_all = macromissingFill(df_all)
print 'macro_missing filling finished: ', df_all[macro_missing].isnull().sum().sum() == 0

# year and housing sq
df_all['used_yr'] = df_all['year'] - df_all['build_year']
df_all['used_yr'][df_all['used_yr'] < 0] = 0
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_life_sq'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
# fillna
df_all['rel_floor'] = df_all['rel_floor'].fillna(df_all['rel_floor'].mean())
df_all['rel_life_sq'] = df_all['rel_life_sq'].fillna(df_all['rel_life_sq'].mean())
df_all['rel_kitch_sq'] = df_all['rel_kitch_sq'].fillna(df_all['rel_kitch_sq'].mean())

numGroup = list(df_all._get_numeric_data().columns)
corr, highCorrList = getCorr(df_all.ix['total',:], numGroup, 0.15)

cntGroup = [i for i in numGroup if re.match(r'\w+_count+',i)]
raionGroup = [i for i in numGroup if re.match(r'\w+_raion+',i)]
kmGroup = [i for i in numGroup if re.match(r'\w+_km+',i)]
minGroup = ['metro_min_avto', 'metro_min_walk', 'public_transport_station_min_walk', 'railroad_station_avto_min',\
            'railroad_station_walk_min']


def min_km_trans(df, group):
    group = list(group)
    newFeats = []
    for i in group:
        df['fsq_'+i+'_inv1'] = df['full_sq'] / (df[i] + 0.1)
        df['fsq_'+i+'_inv5'] = df['full_sq'] / (df[i] + 0.5)
        df['fsq_'+i+'_inv10'] = df['full_sq'] / (df[i] + 1.0)
        df['fsq_'+i+'_invlg1'] = df['full_sq'] / (np.log1p(df[i]) + 0.1)
        df['fsq_'+i+'_invlg5'] = df['full_sq'] / (np.log1p(df[i]) + 0.5)
        df['fsq_'+i+'_invlg10'] = df['full_sq'] / (np.log1p(df[i]) + 1.0)
        # df['full_sq_'+i] = df['full_sq'] / (np.log1p(df[i]) + 0.1)
        newFeats += ['fsq_'+i+'_inv1','fsq_'+i+'_inv5','fsq_'+i+'_inv10','fsq_'+i+'_invlg1',\
                     'fsq_'+i+'_invlg5','fsq_'+i+'_invlg10']
    group.extend(newFeats)
    return df, group

df_all, kmFeats = min_km_trans(df_all, kmGroup)
df_all, minFeats = min_km_trans(df_all, minGroup)

cntFeats = list(set(cntGroup) & set(highCorrList))
extFeats = ['full_sq','life_sq','kitch_sq','floor','max_floor','num_room','build_year',\
            'used_yr','rel_life_sq','rel_kitch_sq','rel_floor',\
           'ppi','cpi','price_doc']
print len(highCorrList),len(cntFeats),len(kmFeats)/7,len(minFeats)/7,len(extFeats)
# print highCorrList
numFeats = cntFeats + kmFeats + minFeats + extFeats

drop_list = ['timestamp','year','year_month']
for i in drop_list:
    df_all.drop(i, axis = 1, inplace = True)

numGroup,catGroup,_ = missingPattern(df_all)

# self-define numGroup
numGroup = numFeats

df_total_num = df_all.ix['total',numGroup]
df_test_num = df_all.ix['test',numGroup]
df_test_num.drop('price_doc', axis = 1, inplace = True)
df_test_num['id'] = test['id']
df_total_cat = df_all.ix['total',catGroup]
df_test_cat = df_all.ix['test',catGroup]

# one-hot encoding for categorical variables
df_concat_cat = pd.concat([df_total_cat,df_test_cat],keys = ['total','test'])
df_total_cat = pd.get_dummies(df_concat_cat).ix['total',:]
df_test_cat = pd.get_dummies(df_concat_cat).ix['test',:]

# save df_total_num, df_total_cat, df_test_num, df_test_cat
df_total_num.to_csv('df_total_num.csv',index = False) # save to .csv file
df_total_cat.to_csv('df_total_cat.csv',index = False) # save to .csv file
df_test_num.to_csv('df_test_num.csv',index = False) # save to .csv file
df_test_cat.to_csv('df_test_cat.csv',index = False) # save to .csv file

# read prepared data from csv
df_total_num = pd.read_csv('df_total_num.csv')
df_total_cat = pd.read_csv('df_total_cat.csv')
df_test_num = pd.read_csv('df_test_num.csv')
df_test_cat = pd.read_csv('df_test_cat.csv')
Ytotal = df_total_num['price_doc'].as_matrix()
testId = list(df_test_num['id'])
df_total_num.drop('price_doc', axis = 1, inplace = True)
df_test_num.drop('id', axis = 1, inplace = True)

