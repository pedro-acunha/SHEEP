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
    """Computes the NMAD, bias, outlier fraction for the regression tasks
  importance: feature importance data
  names: features name
  model_type: Name of the model for the plot title
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
    # Plot predictions vs ground-truth
    cmap="Paired"
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
