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
