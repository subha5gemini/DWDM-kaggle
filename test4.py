import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
#import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

#import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

prop = pd.read_csv('./input/properties_2016.csv')
train = pd.read_csv("./input/train_2016_v2.csv")

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

"""
    if dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
"""
df_train = train.merge(prop, how='left', on='parcelid')

df_train = df_train.drop(['parcelid', 'transactiondate'], axis=1)
#x_test = prop.drop(['parcelid'], axis=1)


missing_df = df_train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by='missing_count')

dropcols = ['finishedsquarefeet12','finishedsquarefeet13', 'finishedsquarefeet15','finishedsquarefeet6']
dropcols.append('finishedsquarefeet50')

dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')

#index = df_train.hashottuborspa.isnull()
#df_train.loc[index,'hashottuborspa'] = "None"

dropcols.append('pooltypeid10')
dropcols.append('propertycountylandusecode')
dropcols.append('propertyzoningdesc')

index = df_train.pooltypeid2.isnull()
df_train.loc[index,'pooltypeid2'] = 0

index = df_train.pooltypeid7.isnull()
df_train.loc[index,'pooltypeid7'] = 0

index = df_train.poolcnt.isnull()
df_train.loc[index,'poolcnt'] = 0

poolsizesum_median = df_train.loc[df_train['poolcnt'] > 0, 'poolsizesum'].median()
df_train.loc[(df_train['poolcnt'] > 0) & (df_train['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median

df_train.loc[(df_train['poolcnt'] == 0), 'poolsizesum'] = 0

#df_train['fireplaceflag']= "No"
#df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']= "Yes"

index = df_train.fireplacecnt.isnull()
df_train.loc[index,'fireplacecnt'] = 0

#index = df_train.taxdelinquencyflag.isnull()
#df_train.loc[index,'taxdelinquencyflag'] = "None"

index = df_train.garagecarcnt.isnull()
df_train.loc[index,'garagecarcnt'] = 0

index = df_train.garagetotalsqft.isnull()
df_train.loc[index,'garagetotalsqft'] = 0

df_train['airconditioningtypeid'].value_counts()
index = df_train.airconditioningtypeid.isnull()
df_train.loc[index,'airconditioningtypeid'] = 1

index = df_train.heatingorsystemtypeid.isnull()
df_train.loc[index,'heatingorsystemtypeid'] = 2

index = df_train.threequarterbathnbr.isnull()
df_train.loc[index,'threequarterbathnbr'] = 1

missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols = dropcols + missingvaluescols
df_train = df_train.drop(dropcols, axis=1)

dropcols.append('parcelid')

x_test = prop.drop(dropcols, axis=1)

#df_train.loc[(df_train['finishedfloor1squarefeet'].isnull()) & (df_train['numberofstories']==1),'finishedfloor1squarefeet'] = df_train.loc[(df_train['finishedfloor1squarefeet'].isnull()) & (df_train['numberofstories']==1),'calculatedfinishedsquarefeet']

#droprows = df_train.loc[df_train['calculatedfinishedsquarefeet']<df_train['finishedfloor1squarefeet']].index
#df_train = df_train.drop(droprows)

#df_train.loc[df_train.taxvaluedollarcnt.isnull(),'taxvaluedollarcnt'] = df_train.loc[df_train.taxvaluedollarcnt.isnull(),'taxamount']
#df_train.loc[df_train.landtaxvaluedollarcnt.isnull(),'landtaxvaluedollarcnt'] = df_train.loc[df_train.landtaxvaluedollarcnt.isnull(),'taxamount']

#========================================== Cleaning Done ======================#

#========================================== XGBoost Starting ===================#
import xgboost as xgb
"""
properties = pd.read_csv('./input/test_properties_2016.csv')
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))
"""
#x_test = prop.drop(['parcelid'], axis=1)

df_train = df_train[ df_train.logerror > -0.4 ]
df_train = df_train[ df_train.logerror < 0.42 ]
x_train = df_train.drop(['logerror'], axis=1)
y_train = df_train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

"""
print x_train
print "==================================="
print y_train
"""

xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold=5,
                   num_boost_round=500,
                   early_stopping_rounds=5,
                   verbose_eval=10,
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
<<<<<<< HEAD
=======

>>>>>>> e17e5eb9ddea0adabb4cba3269dbcb7c77665422
