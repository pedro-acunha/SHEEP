from sklearn.metrics import r2_score,accuracy_score
import numpy as np
import pandas as pd
import sklearn


import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

from sklearn.model_selection import RepeatedKFold
from cuml.model_selection import train_test_split
from sklearn.model_selection import train_test_split as skl_train_test_split

from cuml import preprocessing
import cuml

from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.neighbors import KNeighborsRegressor
from cuml.ensemble import RandomForestRegressor as curfc


cuml.set_global_output_type('numpy')
import cudf
import numpy as np
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

import cupy
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (20,20)

def metric_scores_rgr(x,y):
  """Computes the NMAD, bias, outlier fraction for the regression tasks
  x: ground-truth data
  y: predicted data
  """
    met = np.abs(pd.Series(y-x))
    f_out = met/(1+x.astype(np.float32))
    nmad=1.48*np.median(f_out)
    bias = np.median(f_out)
    y_outlier = pd.Series(np.where(f_out > 0.15, 'outlier', 'not outlier'))
    print("Outliers: \n", y_outlier.value_counts())
    print("\n Bias: \n", bias)
    print("\n NMAD score: \n", nmad)
    print('\n R2 Test: \n', r2_score(x, y))
    pass
  
def plot_feature_importance(importance,names,model_type):
    """Plots feature importance
    importance: features importance output from model
    names: features names
    model_type: name of the model used to compute features importance
  """
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(20,20))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    
def create_rgr_plot(y_test, z_pred, var, title):
"""Plots ground-truth vs predictions
    y_test: ground-truth data
    z_pred: predictions data
    var: variable to be compared
    title: plot title
  """    cmap="Paired"
    plt.figure()
    a=plt.scatter(y_test[var], pd.Series(z_pred), c=y_test['class'],
             cmap=cmap,  edgecolors=(0, 0, 0), s=60, alpha = 1)
    plt.plot([y_test[var].min(),y_test[var].max()], [y_test[var].min(),y_test[var].max()], 'g--', lw=2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(handles=a.legend_elements()[0], labels=['GALAXY','QSO','STAR'])
    plt.xlabel('1+z, Measured', fontsize=22)
    plt.ylabel('1+z, Predicted', fontsize=22)
    plt.title( title, fontsize=22)
    
    return plt
  
def ml_model(model, X_train, y_train, X_test, y_test, var):
""" Fit model, predicts feature, print scores
    model: model to be used (XGBoost, CatBoost, LightGBM, etc)
    X_train: Training features
    y_train: Training target
    X_test: Testing features
    y_test: Testing target
    var: name of the target in the dataset
  """    
    try:
        model.fit(X_train.astype(np.float32),y_train[var])
        pred = model.predict(X_test.astype(np.float32))
        test_y = y_test[var].reset_index(drop=True)
        print(metric_scores_rgr(test_y,pred))
    except:
        model.fit(X_train.to_pandas().astype(np.float32),y_train[var].to_pandas())
        pred = model.predict(X_test.to_pandas().astype(np.float32))
        test_y = y_test[var].to_pandas().reset_index(drop=True)
        print(metric_scores_rgr(test_y,pred))
    pass
    
    try:
        features_importances = model.feature_importances_
    except:
        features_importances = []

    return pred, features_importances

def kf_oof(clf,X_train,y_train,X_test, var,nkf):
""" Predictions using k-fold cross-validation
    clf: model to be used (XGBoost, CatBoost, LightGBM, etc)
    X_train: Training features
    y_train: Training target
    X_test: Testing features
    var: Feature to be predicted
    nkf: number of partitions/folds 
  """   
    kf= RepeatedKFold(n_splits=5,random_state=42)
    kf.get_n_splits(features)

    oof_predictions = np.zeros(len(X_train))
    oof_test_predictions = np.zeros(len(X_test))
    i=0
    
    for train_index, holdout_index in kf.split(X_train, y_train[var]):
        i+=1

        clf.fit(X_train.iloc[train_index], y_train[var].iloc[train_index])
    
        y_pred = clf.predict(X_train.iloc[holdout_index])
        oof_predictions[holdout_index] = y_pred
    
        y_test_pred = clf.predict(X_test)
        oof_test_predictions += y_test_pred
        print("For cycle loop:",i)
        

    oof_test_predictions = oof_test_predictions/i
    return oof_predictions, oof_test_predictions
  
# Read data
df = pd.read_parquet('Yourfilename.parquet')
  
# Create features
features = df.columns.to_list()
features.remove('class')
features.remove('z')
features.remove('specObjID')

targets = ['z']

# Train Test Split
X_train, X_test, y_train, y_test = skl_train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.3, 
                                                    shuffle =True, 
                                                    random_state=24)

#Spectroscopic redshift estimation

## Individual Analysis
### XGBoost
xgb_clf = xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24)

xgb_pred, xgb_feat_importance = ml_model(xgb_clf, X_train, y_train, X_test, y_test, 'z')

create_rgr_plot(y_test, xgb_pred, 'z', 'XGB, predicted spec_z')

plot_feature_importance(xgb_feat_importance,features,'XGBoost')

### CatBoost

train_pool = Pool(X_train, y_train['z'], feature_names=features)
test_pool = Pool(X_test, y_test['z'], feature_names=features)

cb_model = CatBoostRegressor(iterations=1500, max_depth=10, task_type="GPU",random_seed=0, verbose=0)
                          
cb_model.fit(train_pool)
z_cb_pred = cb_model.predict(X_test.astype(np.float32))

metric_scores_rgr(y_test['z'].reset_index(drop=True), z_cb_pred) #Metrics

plot_feature_importance(cb_model.get_feature_importance(),features,'CatBoost')

create_rgr_plot(y_test, xgb_pred, 'z', 'CatBoost, predicted spec_z')

### LightGBM

lgb_clf = lgb.LGBMRegressor(n_estimators=1500,random_seed=0, verbose=0)
                          
lgb_pred, lgb_feat_importance = ml_model(lgb_clf, X_train, y_train, X_test, y_test, 'z')

plot_feature_importance(lgb_feat_importance,features,'LightGBM')

create_rgr_plot(y_test, lgb_pred, 'z', 'LightGBM, predicted spec_z')

## OOF Predictions

models = {'xgb':xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24),
        'cb':CatBoostRegressor(iterations=1500, task_type="GPU",random_seed=0, verbose=0),
        'lgb':lgb.LGBMRegressor(n_estimators=1500,random_seed=0, verbose=0)}

train_meta = pd.DataFrame(index=X_train.index,columns=[*models])
test_meta = pd.DataFrame(index=X_test.index,columns=[*models])

for i, model_name in enumerate([*models]):
    if i==0: print('Starting OOF predictions for all models. \n')
    train_meta[model_name], test_meta[model_name] = kf_oof(models[model_name],X_train,y_train,X_test,'z', nkf=5)
    print('Predictions for '+str(model_name)+' done!')
    
test_predictions = pd.DataFrame(test_meta['average'], index=X_test.index)
train_predictions = pd.DataFrame(train_meta['average'], index=X_train.index)

oof_z_spec_pred = pd.concat([train_predictions,test_predictions],axis=0).sort_index()

oof_z_spec_pred.to_csv('./Data/oof_z_pred_v1.csv', index=True)

# Photo_z predictions correction
df_z = pd.read_csv('./Data/oof_z_pred_v1.csv')

df = pd.merge(df,df_z, left_index=True, right_index=True) # Merge photometric predictions with initial data

## Create features
features = df.columns.to_list()
features.remove('class')
features.remove('z')
features.remove('specObjID')

targets = ['class','z','specObjID']

X_train, X_test, y_train, y_test = skl_train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.3, 
                                                    shuffle =True, 
                                                    random_state=24)

models = {'xgb':xgb.XGBRegressor(n_estimators=1500, n_jobs=-1,tree_method='gpu_hist', random_state=24),
        'cb':CatBoostRegressor(iterations=1500, task_type="GPU",random_seed=0, verbose=0),
        'lgb':lgb.LGBMRegressor(n_estimators=1500,random_seed=0, verbose=0)}

train_meta = pd.DataFrame(index=X_train.index,columns=[*models])
test_meta = pd.DataFrame(index=X_test.index,columns=[*models])

for i, model_name in enumerate([*models]):
    if i==0: print('Starting OOF predictions for all models. \n')
    train_meta[model_name], test_meta[model_name] = kf_oof(models[model_name],X_train,y_train,X_test,'z', nkf=5)
    print('Predictions for '+str(model_name)+' done!')
    
# let's take a look at the features for training our meta learner:
test_meta['average'] = test_meta[[*models]].mean(axis=1)
train_meta['average'] = train_meta[[*models]].mean(axis=1)

test_predictions = pd.DataFrame(test_meta['average'], index=X_test.index)
train_predictions = pd.DataFrame(train_meta['average'], index=X_train.index)

oof_z_spec_pred = pd.concat([train_predictions,test_predictions],axis=0).sort_index()

oof_z_spec_pred.to_csv('./Data/oof_z_pred_v2.csv', index=True)
