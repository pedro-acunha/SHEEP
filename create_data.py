from sklearn.metrics import r2_score,accuracy_score, confusion_matrix, classification_report,precision_recall_fscore_support
import numpy as np
import pandas as pd
import sklearn
from flaml import AutoML

from sklearn.model_selection import train_test_split, RepeatedKFold

import xgboost as xgb

import cudf
import numpy as np
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

def create_colours(data, features):
    """Create dataframe with colour-colour data. Return an Array and list of colours.

    Args:
        data (DataFrame): Dataframe with photometric data
        features (list): List of features from photometric data
    """
    N = len(data)
    F = len(features)
    n=0
    for i in np.linspace(1,F,F,dtype=int):
    	n = n + (i-1)

    df_features = np.zeros((N, n))
    y=0
    lista=[]
    for z in np.linspace(0,F,F,dtype=int):
        for x in np.linspace(1,F-1,F-1,dtype=int):
            if z!=x and z<x:
                df_features[:,y] = np.abs(data[features[z]] - data[features[x]])
                y+=1
                lista += [features[z]+'-'+features[x]]
            else:
                pass
    df_colours = pd.DataFrame(df_features,columns = lista)
    return df_colours

def create_colours_filter(data, features):
  """Create dataframe with colour-colour data with the same broad band filter. Return an Array and list of colours.

    Args:
        data (DataFrame): Dataframe with photometric data
        features (list): List of features from photometric data
    """
    # Function to create colours by broadband magnitude filter
    N = len(data)
    F = len(features)
        
    df_features = np.zeros((N, F-2))
    y=0
    lista=[]
    for z in np.linspace(0,F-1,F-1,dtype=int):
        for x in np.linspace(1,F-1,F-1,dtype=int):
            if (features[z]!=features[x]) and (features[z][-2:] == features[x][-2:]):
                df_features[:,y] = np.abs(data[features[z]] - data[features[x]])
                y+=1
                lista += [features[z]+'-'+features[x]]
            else:
                pass
    df_colours = pd.DataFrame(df_features,columns = lista)
    
    return df_colours
