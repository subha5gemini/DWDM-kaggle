import numpy as np
import pandas as pd
from datetime import datetime
import gc
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

prop = pd.read_csv('../kaggle/input/properties_2016.csv')
train = pd.read_csv("../kaggle/input/train_2016_v2.csv")

for c in prop.columns:
    if prop[c].dtype == np.float64:   
        prop[c] = prop[c].astype(np.float32)


df_train = train.merge(prop, how='left', on='parcelid')
df_train = df_train.drop(['parcelid', 'transactiondate'], axis=1)

dropcols = []

#Remove columns with missing value % greater than 97
missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols += missingvaluescols

#Drop unnecessary columns
dropcols += ['finishedsquarefeet12', 'finishedsquarefeet15','finishedsquarefeet50','fullbathcnt','propertycountylandusecode','propertyzoningdesc']
dropcols += ['finishedsquarefeet50', 'censustractandblock','regionidneighborhood', 'calculatedbathnbr', 'fullbathcnt']

bathroomcnt_mode = df_train.loc[df_train['bathroomcnt'] > 0, 'bathroomcnt'].mode()[0]
index = df_train.bathroomcnt.isnull()
df_train.loc[index,'bathroomcnt'] = bathroomcnt_mode

bedroomcnt_mode = df_train.loc[df_train['bedroomcnt'] > 0, 'bedroomcnt'].mode()[0]
index = df_train.bedroomcnt.isnull()
df_train.loc[index,'bedroomcnt'] = bedroomcnt_mode

index = df_train.fireplacecnt.isnull()
df_train.loc[index,'fireplacecnt'] = 0

index = df_train.garagecarcnt.isnull()
df_train.loc[index,'garagecarcnt'] = 0

index = df_train.garagecarcnt.isnull()
df_train.loc[(df_train['garagetotalsqft'] > 0) & index, 'garagecarcnt'] = 1


df_train.loc[(df_train['garagecarcnt'] == 0), 'garagetotalsqft'] = 0

index = df_train.garagetotalsqft.isnull()
df_train.loc[index, 'garagetotalsqft'] = 0

index = df_train.poolcnt.isnull()
df_train.loc[index, 'poolcnt'] = 0
df_train.loc[(df_train['poolsizesum'] > 0) & (df_train['poolcnt'] == 0), 'poolcnt'] = 1


index = df_train.roomcnt.isnull()
df_train.loc[index,'roomcnt'] = 1

dropcols = list(set(dropcols))
df_train = df_train.drop(dropcols, axis=1)
dropcols.append('parcelid')
x_test = prop.drop(dropcols, axis=1)

#Fill missing values
df_train.fillna(-1, inplace = True)
x_test.fillna(-1, inplace = True)

#========================================== Cleaning Done ======================#

df_train = df_train[ df_train.logerror > -0.4 ]
df_train = df_train[ df_train.logerror < 0.4 ]
x_train = df_train.drop(['logerror'], axis=1)
y_train = df_train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)


#========================================== RandomForest =======================#

model = RandomForestRegressor()
model.fit(x_train,y_train)
rf = model.predict(x_test)

#========================================== Decision Tree ======================#
model = DecisionTreeRegressor(max_depth=6)
model.fit(x_train,y_train)
dt = model.predict(x_test)

#========================================== AdaBoost ===========================#
rng = np.random.RandomState(1)
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=500, random_state=rng)
model.fit(x_train,y_train)
ada = model.predict(x_test)

#========================================== XGBoost Starting ===================# O/P: 0.0646140

xgb_params = {
    'eta': 0.01,
    'max_depth': 6,
    'subsample': 0.75,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold = 10,
                   num_boost_round = 500,
                   early_stopping_rounds = 20,
                   verbose_eval = 10,
                   show_stdv = False
                  )
num_boost_rounds = 250

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
xgb1 = model.predict(dtest)
gc.collect()

#====================================== XGBoost Again =========================#
xgb_params = {
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold = 10,
                   num_boost_round = 500,
                   early_stopping_rounds = 20,
                   verbose_eval = 10,
                   show_stdv = False
                  )
num_boost_rounds = 250

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
xgb2 = model.predict(dtest)

total_pred = 0.5*(0.3*xgb1 + 0.7*xgb2) + 0.2*rf + 0.2*ada + 0.1*dt

y_pred = list()
for i,predict in enumerate(total_pred):
    y_pred.append(str(round(predict,4)))
y_pred = np.array(y_pred)

output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
