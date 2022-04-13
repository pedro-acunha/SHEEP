from sklearn.metrics import r2_score,accuracy_score, confusion_matrix, classification_report,precision_recall_fscore_support
import numpy as np
import pandas as pd
from collections import Counter
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

def flaml_opt_clf(estimator, X_train, y_train):
  """FLAML optimiser
    # estimator: estimator name -> ['cb', 'xgb', 'lgb'], if other then auto mode
    # X_train: training features
    # y_train: training targets
    """
    model = AutoML()
    
    if estimator == ['cb']:
        
        settings = {
        "time_budget": 4200,  # total running time in seconds
        "estimator_list": ['catboost'],  # list of ML learners; we tune xgboost in this example
        "n_jobs": -1,
        "task": 'classification',  # task type    
        "log_file_name": 'clf.log'  # flaml log file
        }
    
    
    elif estimator == ['xgb']:
        
        settings = {
        "time_budget": 4200,  # total running time in seconds
        "estimator_list": ['xgboost'],  # list of ML learners; we tune xgboost in this example
        "n_jobs": -1,
        #"average": 'macro',
        #"tree_method":'gpu_hist',
        "task": 'classification',  # task type    
        "log_file_name": 'clf.log'  # flaml log file
        }
        
    elif estimator == ['lgb']:
        
        settings = {
        "time_budget": 4200,  # total running time in seconds
        "estimator_list": ['lgbm'],  # list of ML learners; we tune xgboost in this example
        "n_jobs": -1,
        "task": 'classification',  # task type    
        "log_file_name": 'clf.log'  # flaml log file
        }
    else:
        settings = {
        "time_budget": 4200,  # total running time in seconds
        "n_jobs": -1,
        "task": 'classification',  # task type    
        "log_file_name": 'clf.log'  # flaml log file
        }
        
        
    model.fit(X_train=X_train, y_train=y_train, **settings)

    print('Best estimator:', model.best_config_per_estimator)
    return model

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
        pred_proba = model.predict_proba(X_test.astype(np.float32))
        test_y = y_test[var].reset_index(drop=True)
        print(metric_scores(test_y,pred))
    except:
        
        model.fit(X_train.to_pandas().astype(np.float32),y_train[var].to_pandas())
        pred = model.predict(X_test.to_pandas().astype(np.float32))
        pred_proba = model.predict_proba(X_test.to_pandas().astype(np.float32))
        test_y = y_test[var].to_pandas().reset_index(drop=True)
        print(metric_scores(test_y,pred))
    pass
    
    try:
        features_importances = model.feature_importances_
    except:
        features_importances = []

    return pred, pred_proba, features_importances

# Read dataframe
df = pd.read_pickle('./Yourfilehere.pkl')

# Set targets
df['is_galaxy'] = np.where(df['class']==0,0,1)
df['is_qso'] = np.where(df['class']==1,0,1)
df['is_star'] = np.where(df['class']==2,0,1)

# Set scale_pos_weight variable for XGBoost (use if necessary)
counter_gal = Counter(df['is_galaxy'])
counter_qso = Counter(df['is_qso'])
counter_star = Counter(df['is_star'])

scale_pos_weight_gal = np.round(counter_gal[0]/counter_gal[1])
scale_pos_weight_qso = np.round(counter_qso[1]/counter_qso[0])
scale_pos_weight_star = np.round(counter_star[1]/counter_star[0])

# Train Test split
features = df.columns.to_list()
features.remove('class')
features.remove('z')
features.remove('is_galaxy')
features.remove('is_qso')
features.remove('is_star')
features.remove('flag_outlier')

targets = ['class','is_galaxy','is_qso','is_star']

X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[targets], 
                                                    test_size=0.5, 
                                                    shuffle =True, 
                                                    random_state=0)

# Galaxy vs all (FLAML optimised models)
#XGBoost
xgb_gal = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.47730610136374474, colsample_bynode=1,
              colsample_bytree=0.9643171868050568, gamma=0, gpu_id=-1,
              grow_policy='lossguide', importance_type='gain',
              interaction_constraints='', learning_rate=0.03967786833707455,
              max_delta_step=0, max_depth=0, max_leaves=379,
              min_child_weight=1.5227128632589952, missing=-9999.0,
              monotone_constraints='()', n_estimators=592, n_jobs=-1,
              num_parallel_tree=1, random_state=0,
              reg_alpha=0.007264415841269545, reg_lambda=0.03829654048452607,
              scale_pos_weight=1, subsample=0.9487880180671172,
              tree_method='gpu_hist', use_label_encoder=False,
              validate_parameters=1, verbosity=0)

xgb_pred_gal, xgb_pred_proba_gal, xgb_feat_importance_gal = ml_model(xgb_gal, X_train, y_train, X_test, y_test, 'is_galaxy')

plot_feature_importance(xgb_feat_importance_gal,features,'XGBoost','feat_import_xgb_gal_vs_all.png')

#LightGBM
lgb_gal = lgb.LGBMClassifier(colsample_bytree=0.4778718200151245,
               learning_rate=0.04990970462089558, max_bin=31,
               min_child_samples=7, n_estimators=2024, num_leaves=65,
               reg_alpha=0.0012631653320853778, reg_lambda=0.06125445327792913,
               verbose=-1)

lgb_pred_gal, lgb_pred_proba_gal, lgb_feat_importance_gal = ml_model(lgb_gal, X_train, y_train, X_test, y_test, 'is_galaxy')

plot_feature_importance(lgb_feat_importance_gal,features,'LightGBM','feat_import_lgb_gal_vs_all.png')

#CatBoost
cb_gal = CatBoostClassifier(early_stopping_rounds= 10, learning_rate= 0.09356494712516784,
                            n_estimators= 290, task_type = 'GPU', verbose=0)

cb_pred_gal, cb_pred_proba_gal, cb_feat_importance_gal = ml_model(cb_gal, X_train, y_train, X_test, y_test, 'is_galaxy')

plot_feature_importance(cb_feat_importance_gal,features,'CatBoost','feat_import_cb_gal_vs_all.png')

# QSO vs all (FLAML optimised models)
# XGBoost
xgb_qso = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.41898802160182846, colsample_bynode=1,
              colsample_bytree=0.4972272561013618, gamma=0, gpu_id=-1,
              grow_policy='lossguide', importance_type='gain',
              interaction_constraints='', learning_rate=0.009359109522734632,
              max_delta_step=0, max_depth=0, max_leaves=2207,
              min_child_weight=19.88765593000535, missing=-9999.0,
              monotone_constraints='()', n_estimators=1733, n_jobs=-1,
              num_parallel_tree=1, random_state=0, reg_alpha=0.1461519451411134,
              reg_lambda=0.35169598803448954, scale_pos_weight=1,
              subsample=0.9462723804220237, tree_method='gpu_hist',
              use_label_encoder=False, validate_parameters=1, verbosity=0)

xgb_pred_qso, xgb_pred_proba_qso, xgb_feat_importance_qso = ml_model(xgb_qso, X_train, y_train, X_test, y_test, 'is_qso')

plot_feature_importance(xgb_feat_importance_qso,features,'XGBoost','feat_import_xgb_qso_vs_all.png')

#LightGBM
lgb_qso = lgb.LGBMClassifier(colsample_bytree=0.6017096634144301,
               learning_rate=0.036154450725483754, max_bin=1023,
               min_child_samples=8, n_estimators=692, num_leaves=519,
               reg_alpha=0.001975258376030875, reg_lambda=2.0822570450688955,
               verbose=-1)

lgb_pred_qso,lgb_pred_proba_qso, lgb_feat_importance_qso = ml_model(lgb_qso, X_train, y_train, X_test, y_test, 'is_qso')

plot_feature_importance(lgb_feat_importance_qso,features,'LightGBM','feat_import_lgb_qso_vs_all.png')

#CatBoost
cb_qso = CatBoostClassifier(early_stopping_rounds= 10, learning_rate= 0.1229120572728312,
                            n_estimators= 122, task_type = 'GPU', verbose=0)

cb_pred_qso, cb_pred_proba_qso, cb_feat_importance_qso = ml_model(cb_qso, X_train, y_train, X_test, y_test, 'is_qso')

plot_feature_importance(cb_feat_importance_qso,features,'CatBoost','feat_import_cb_qso_vs_all.png')

# Star vs all (FLAML optimised models)
# XGBoost
xgb_star = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=0, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.7540582998659452,
              max_delta_step=0, max_depth=11, min_child_weight=1,
              missing=-9999.0, monotone_constraints='()', n_estimators=1668,
              n_jobs=-1, num_parallel_tree=1, random_state=24, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=5.0, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

xgb_pred_star, xgb_pred_proba_star, xgb_feat_importance_star = ml_model(xgb_star, X_train, y_train, X_test, y_test, 'is_star')

plot_feature_importance(xgb_feat_importance_star,features,'XGBoost','feat_import_xgb_star_vs_all.png')

#LightGBM
lgb_star = lgb.LGBMClassifier(colsample_bytree=0.7547746254097104,
               learning_rate=0.0248246268850278, max_bin=1023,
               min_child_samples=11, n_estimators=4054, num_leaves=276,
               reg_alpha=0.025749981001891773, reg_lambda=0.6701686988213098,
               verbose=-1)

lgb_pred_star,lgb_pred_proba_star, lgb_feat_importance_star = ml_model(lgb_star, X_train, y_train, X_test, y_test, 'is_star')

plot_feature_importance(lgb_feat_importance_star,features,'LightGBM','feat_import_lgb_star_vs_all.png')

#CatBoost
cb_star = CatBoostClassifier(early_stopping_rounds= 10, learning_rate= 0.08740841428170476,
                             n_estimators= 167, task_type = 'GPU',verbose=0)

cb_pred_star,cb_pred_proba_star, cb_feat_importance_star = ml_model(cb_star, X_train, y_train, X_test, y_test, 'is_star')

plot_feature_importance(cb_feat_importance_star,features,'CatBoost','feat_import_cb_star_vs_all.png')

# Analysis of the predictions

predictions = pd.DataFrame({'is_gal': xgb_pred_gal, 'is_qso': xgb_pred_qso, 'is_star': xgb_pred_star}, index=y_test.index)
gal_pred_proba = pd.DataFrame({'xgb': pd.DataFrame(xgb_pred_proba_gal, index=y_test.index)[0], 'lgb': pd.DataFrame(lgb_pred_proba_gal, index=y_test.index)[0], 'cb': pd.DataFrame(cb_pred_proba_gal, index=y_test.index)[0]}, index=y_test.index)
qso_pred_proba = pd.DataFrame({'xgb': pd.DataFrame(xgb_pred_proba_qso, index=y_test.index)[0], 'lgb': pd.DataFrame(lgb_pred_proba_qso, index=y_test.index)[0], 'cb': pd.DataFrame(cb_pred_proba_qso, index=y_test.index)[0]}, index=y_test.index)
star_pred_proba = pd.DataFrame({'xgb': pd.DataFrame(xgb_pred_proba_star, index=y_test.index)[0], 'lgb': pd.DataFrame(lgb_pred_proba_star, index=y_test.index)[0], 'cb': pd.DataFrame(cb_pred_proba_star, index=y_test.index)[0]}, index=y_test.index)

predictions['equal_gal_qso']=np.where((predictions['is_gal']== 0) & (predictions['is_qso']==0), True, False)
predictions['equal_gal_star']=np.where((predictions['is_gal']== 0) & (predictions['is_star']==0), True, False)
predictions['equal_qso_star']=np.where((predictions['is_qso']== 0) & (predictions['is_star']==0), True, False)
predictions['equal_all']=np.where((predictions['is_gal']== 1) & (predictions['is_qso']== 1) & (predictions['is_star']==1), True, False)

#Check predictions
predictions[predictions['equal_all']==True]
predictions[predictions['equal_gal_star']==True]
predictions[predictions['equal_qso_star']==True]
predictions[predictions['equal_gal_qso']==True]

#Index from uncertain predictions
l_index = list(predictions['equal_gal_qso'][predictions['equal_gal_qso']==True].index) + list(predictions['equal_gal_star'][predictions['equal_gal_star']==True].index) + list(predictions['equal_qso_star'][predictions['equal_qso_star']==True].index) + list(predictions[predictions['equal_all']==True].index)

#Compilation of the one vs all approach
##Transforming binary classification into categorical
pred_gal = pd.Series(np.where((predictions['is_gal']== 0) & (predictions['is_qso']==1) & (predictions['is_star']==1), 'GALAXY', 'Other'), index = predictions.index)
pred_gal = pred_gal[pred_gal=='GALAXY']
pred_qso = pd.Series(np.where((predictions['is_gal']== 1) & (predictions['is_qso']==0) & (predictions['is_star']==1), 'QSO', 'Other'), index = predictions.index)
pred_qso = pred_qso[pred_qso=='QSO']
pred_star = pd.Series(np.where((predictions['is_gal']== 1) & (predictions['is_qso']==1) & (predictions['is_star']==0), 'STAR', 'Other'), index = predictions.index)
pred_star = pred_star[pred_star=='STAR']

pred_one_vs_all = pd.Series()
l = [pred_gal, pred_qso, pred_star]

#Append all predictions into one pandas Series
for i in range(len(l)):
    pred_one_vs_all = pred_one_vs_all.append(l[i])
    
le = sklearn.preprocessing.LabelEncoder()
pred_one_vs_all = pd.Series(le.fit_transform(pred_one_vs_all), index=pred_one_vs_all.index)
print('Labels [0,1,2]: ',le.inverse_transform([0,1,2]))
print('Data encoded.\n')

# For sanity check, run: len(y_test)-len(pred_one_vs_all)

# Validation of the one vs all predictions (without uncertain predictions)
one_index = pred_one_vs_all.index.to_list()

# FLAML optimised model
xgb_multi = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=0, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.28591333910945194,
              max_delta_step=0, max_depth=8, min_child_weight=1, missing=-9999.0,
              monotone_constraints='()', n_estimators=1301, n_jobs=-1,
              num_parallel_tree=1, objective='multi:softprob', random_state=24,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

xgb_pred_one,xgb_pred_one_proba, xgb_feat_importance_one = ml_model(xgb_multi, X_train, y_train, X_test.loc[one_index], y_test.loc[one_index], 'class')

metric_scores(y_test['class'].loc[one_index],pred_one_vs_all)

plot_feature_importance(xgb_feat_importance_one,features,'XGBoost','feat_import_xgb_onevsall_clear_preds.png')


# Meta-learner to correct uncertain predictions

one_predictions = pd.merge(gal_pred_proba, qso_pred_proba, left_index=True, right_index=True)
one_predictions = pd.merge(one_predictions, star_pred_proba, left_index=True, right_index=True)
one_predictions = pd.merge(one_predictions, y_test['class'], left_index=True, right_index=True)

X_train_one = one_predictions[['xgb_x', 'lgb_x', 'cb_x', 'xgb_y', 'lgb_y', 'cb_y', 'xgb', 'lgb', 'cb']].drop(l_index)
y_train_one = one_predictions[['class']].drop(l_index)
y_train_one = y_train_one['class']

X_test_one = one_predictions[['xgb_x', 'lgb_x', 'cb_x', 'xgb_y', 'lgb_y', 'cb_y', 'xgb', 'lgb', 'cb']].loc[l_index]
y_test_one = one_predictions[['class']].loc[l_index]
y_test_one = y_test_one['class']
y_test_one = y_test_one[~y_test_one.index.duplicated(keep='first')]

meta_automl = flaml_opt_clf([], X_train_one, y_train_one) # FLAML optimisation

# LGBClassifier from the previous FLAML_AUTOML
meta_automl= lgb.LGBMClassifier(learning_rate=0.1032373221027049, max_bin=1023,
               min_child_samples=18, n_estimators=103, num_leaves=38,
               reg_alpha=0.019732117320210126, reg_lambda=0.24956851428840968,
               verbose=-1, random_state=0)

meta_automl.fit(X_train_one,y_train_one)

meta_preds = meta_automl.predict(X_test_one)
meta_preds = pd.Series(meta_preds, index=X_test_one.index)
meta_preds = meta_preds[~meta_preds.index.duplicated(keep='first')]

meta_preds_proba = meta_automl.predict_proba(X_test_one)

metric_scores(y_test_one,meta_preds)

meta_preds_proba = pd.DataFrame(meta_preds_proba)


#Compilation of the one vs all predictions and uncertain predictions

final_preds = pred_one_vs_all.append(pd.Series(meta_preds))

# For sanity check, run: len(y_test)-len(final_preds)

metric_scores(y_test['class'].reindex(final_preds.index),final_preds)

# Save results here.
