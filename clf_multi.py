from sklearn.metrics import r2_score,accuracy_score, confusion_matrix, classification_report,precision_recall_fscore_support
import numpy as np
import pandas as pd
import sklearn
from flaml import AutoML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm, CatBoostClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler
cuml.set_global_output_type('numpy')
import cudf
import numpy as np
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
import cupy
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rcParams["figure.figsize"] = (20,20)
plt.rcParams.update({'font.size': 22})
import seaborn as sns

# Outlier detection

def flag_cat_outliers(x, var1, var2):
  """Identifies catastrophic outliers and flags them
    # x (DataFrame): Data to verify outliers
    # var1 (string): ground-truth variable
    # var2 (string): predicted variable
    """
    met = np.abs(pd.Series(x[var2]-x[var1]))
    f_out = met/(1+x[var1].astype(np.float32))
    y_outlier = pd.Series(np.where(f_out > 0.15, 0, 1)).reindex(x.index)
    return y_outlier
  
  def act_learn_outlier(model,x_train, y_train, x_test, y_test):
    """Dection of outliers
     model: ML model to be used
     x_train: training features
     y_train: training target
     x_test: testing features
     y_test: testing target
    """
    model.fit(x_train, y_train['flag_outlier'])
    print('Outlier model fit done!')
    outliers = pd.DataFrame(model.predict(x_test), index=x_test.index, columns=['flag'])
    print(metric_scores(y_test['flag_outlier'],outliers['flag']))
    print('Outlier predictions completed!')
    indices = outliers[outliers['flag']==0].index.to_list()
    x_test = x_test.drop(indices)
    y_test = y_test.drop(indices)
    print('Potential catastrophic outliers exterminated!')
    return x_test, y_test

def clf_act_learn (model_clf, model_out,df, features, targets):
  """Multi-class model with outlier detection and removal
     model_clf: ML model to be used for classification
     model_clf: ML model to be used for outlier detection
     features: list of features names
     features: list of targets names
     x_test: testing features
     y_test: testing target
    """
    X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.3, 
                                                    shuffle =True, 
                                                    random_state=0)
    print('Train Test Split finished.')
    model_clf.fit(X_train.astype(np.float32),y_train['class'].astype(np.int32))
    print('Model fit done!')
    pred = model_clf.predict(X_test.astype(np.float32))
    print('Model predictions completed!')
    test_y = y_test['class'].reset_index(drop=True)
    print(metric_scores(test_y,pred))
    
    print('Active learning mode activated!')
    X_test_al, y_test_al = act_learn_outlier(model_out,X_train, y_train, X_test, y_test)
    print('Active learning mode completed!')
    new_pred = model_clf.predict(X_test_al.astype(np.float32))
    print('New predictions!')
    y_test_al = y_test_al['class'].reset_index(drop=True)
    print(metric_scores(y_test_al,new_pred))
    
    return X_train, X_test, y_train, y_test, pred, new_pred, X_test_al, y_test_al

  
# Read dataframe
df = pd.read_pickle('./Yourfilehere.pkl')

# Train Test split
features = df.columns.to_list()
features.remove('class')
features.remove('z')


targets = ['class']
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.5, 
                                                    shuffle =True, 
                                                    random_state=0)

# Models optimised using FLAML
## XGBoost
xgb_clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.5241738682334621, colsample_bynode=1,
              colsample_bytree=0.671168787703373, gamma=0, gpu_id=-1,
              grow_policy='lossguide', importance_type='gain',
              interaction_constraints='', learning_rate=0.07438542233856403,
              max_delta_step=0, max_depth=0, max_leaves=605,
              min_child_weight=45.13563288392657, missing=-9999.0,
              monotone_constraints='()', n_estimators=150, n_jobs=-1,
              num_parallel_tree=1, objective='multi:softprob', random_state=0,
              reg_alpha=0.0009765625, reg_lambda=0.6112274550663784,
              scale_pos_weight=None, subsample=0.8977673953174077,
              use_label_encoder=False,validate_parameters=1, verbosity=0,
              tree_method='gpu_hist')

## CatBoost
cb_clf = CatBoostClassifier(early_stopping_rounds= 11, learning_rate= 0.18865783582261322,
                            n_estimators= 131,verbose=0, task_type="GPU")

## LightGBM
lgb_clf = lgb.LGBMClassifier(colsample_bytree=0.9102721130426018,
               learning_rate=0.10233023968742658, max_bin=1023,
               min_child_samples=9, n_estimators=914, num_leaves=26,
               reg_alpha=0.002021902215198225, reg_lambda=22.68747159832061,
               verbose=-1)

## Computing the metrics and feature importances
xgb_pred,xgb_pred_proba, xgb_feat_importance = ml_model(xgb_clf, X_train, y_train, X_test, y_test, 'class')
cb_pred, cb_pred_proba,cb_feat_importance = ml_model(cb_clf, X_train, y_train, X_test, y_test, 'class')
lgb_pred, lgb_pred_proba, lgb_feat_importance = ml_model(lgb_clf, X_train, y_train, X_test, y_test, 'class')

plot_feature_importance(xgb_feat_importance,features,'XGBoost', 'feat_import_xgb_multiclass.png')
plot_feature_importance(cb_feat_importance,features,'CatBoost', 'feat_import_cb_multiclass.png')
plot_feature_importance(lgb_feat_importance,features,'LightGBM', 'feat_import_lgb_multiclass.png')


## Combining te results into a dataframe
multiclass_pred = pd.DataFrame({'xgb': xgb_pred, 'cb': pd.Series(map(lambda x: x[0], cb_pred), index=y_test.index), 'lgb': lgb_pred}, index=y_test.index)
multiclass_pred['hard_vote']=np.round(multiclass_pred.mean(axis=1), decimals=0).astype('int32')

## Comparing classification metrics
metric_scores(y_test['class'],multiclass_pred['hard_vote'])

## Save predictions
multiclass_pred.to_csv('./Data/Clf_multiclass.csv', index=True)

# Outliers dection mode
  
## Flag outliers for training
df['flag_outlier']= flag_cat_outliers(df, 'z', 'imp_z')

targets.append('flag_outlier')

X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.5, 
                                                    shuffle =True, 
                                                    random_state=0)

## Models optimised using FLAML
xgb_out = xgb.XGBClassifier(n_estimators= 4, max_leaves= 4, min_child_weight= 0.9999999999999993,
                            learning_rate= 0.09999999999999995, subsample= 1.0, colsample_bylevel= 1.0,
                            colsample_bytree= 1.0, reg_alpha= 0.0009765625, reg_lambda= 1.0,
                            tree_method='gpu_hist', validate_parameters=1, verbosity=None)

xgb_clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.5241738682334621, colsample_bynode=1,
              colsample_bytree=0.671168787703373, gamma=0, gpu_id=-1,
              grow_policy='lossguide', importance_type='gain',
              interaction_constraints='', learning_rate=0.07438542233856403,
              max_delta_step=0, max_depth=0, max_leaves=605,
              min_child_weight=45.13563288392657, missing=-9999.0,
              monotone_constraints='()', n_estimators=150, n_jobs=-1,
              num_parallel_tree=1, objective='multi:softprob', random_state=0,
              reg_alpha=0.0009765625, reg_lambda=0.6112274550663784,
              scale_pos_weight=None, subsample=0.8977673953174077,
              use_label_encoder=False,validate_parameters=1, verbosity=0,
              tree_method='gpu_hist')

X_train, X_test, y_train, y_test, pred, new_pred, X_test_al, y_test_al = clf_act_learn (xgb_clf, xgb_out, df, features, targets)

#Save files here.
