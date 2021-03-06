import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


prop = pd.read_csv('./input/properties_2016.csv')
train = pd.read_csv("./input/train_2016_v2.csv")

for c in prop.columns:
    prop[c] = prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
    if prop[c].dtype == np.float64:   
        prop[c] = prop[c].astype(np.float32)


df_train = train.merge(prop, how='left', on='parcelid')
df_train = df_train.drop(['parcelid', 'transactiondate'], axis=1)

dropcols = []

#Remove columns with missing value % greater than 97
missingvalues_prop = ((df_train == -1).sum()/len(df_train)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols += missingvaluescols

#print dropcols
#Drop unnecessary columns
dropcols += ['finishedsquarefeet12', 'finishedsquarefeet15','finishedsquarefeet50','fullbathcnt','propertycountylandusecode','propertyzoningdesc']

bathroomcnt_mode = df_train.loc[df_train['bathroomcnt'] > 0, 'bathroomcnt'].mode()[0]
df_train.loc[(df_train['bathroomcnt'] == -1),'bathroomcnt'] = bathroomcnt_mode

dropcols += ['calculatedbathnbr', 'fullbathcnt']

bedroomcnt_mode = df_train.loc[df_train['bedroomcnt'] > 0, 'bedroomcnt'].mode()[0]
df_train.loc[(df_train['bedroomcnt'] == -1),'bedroomcnt'] = bedroomcnt_mode

# ff1sqf_mean = df_train.loc[df_train['finishedfloor1squarefeet'] > 0, 'finishedfloor1squarefeet'].mean()
# df_train.loc[(df_train['finishedfloor1squarefeet'] == -1),'finishedfloor1squarefeet'] = ff1sqf_mean

dropcols += ['finishedsquarefeet50']

# cfsqf_mean = df_train.loc[df_train['calculatedfinishedsquarefeet'] > 0, 'calculatedfinishedsquarefeet'].mean()
# df_train.loc[(df_train['calculatedfinishedsquarefeet'] == -1),'calculatedfinishedsquarefeet'] = cfsqf_mean

df_train.loc[(df_train['fireplacecnt'] == -1),'fireplacecnt'] = 0

df_train.loc[(df_train['garagecarcnt'] == -1), 'garagecarcnt'] = 0
df_train.loc[(df_train['garagetotalsqft'] > 0) & (df_train['garagecarcnt'] == -1), 'garagecarcnt'] = 1


df_train.loc[(df_train['garagecarcnt'] == 0), 'garagetotalsqft'] = 0
df_train.loc[(df_train['garagetotalsqft'] == -1), 'garagetotalsqft'] = 0

df_train.loc[(df_train['poolcnt'] == -1), 'poolcnt'] = 0
df_train.loc[(df_train['poolsizesum'] > 0) & (df_train['poolcnt'] == -1), 'poolcnt'] = 1

dropcols += ['censustractandblock','regionidneighborhood']

index = (df_train['roomcnt'] == -1)
df_train.loc[index,'roomcnt'] = 1

# index = df_train['assessmentyear'] == -1
# df_train.loc[index,'assessmentyear'] = 2015

# index = df_train['taxdelinquencyflag'] == -1
# df_train.loc[index,'taxdelinquencyflag'] = 'Y'

df_train = df_train.drop(dropcols, axis=1)
dropcols.append('parcelid')
x_test = prop.drop(dropcols, axis=1)

#========================================== Cleaning Done ======================#

#========================================== XGBoost Starting ===================#
import xgboost as xgb


df_train = df_train[ df_train.logerror > -0.4 ]
df_train = df_train[ df_train.logerror < 0.4 ]
x_train = df_train.drop(['logerror'], axis=1)
y_train = df_train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.75,
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
                   nfold = 20,
                   num_boost_round = 500,
                   early_stopping_rounds = 20,
                   verbose_eval = 10,
                   show_stdv = False
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

