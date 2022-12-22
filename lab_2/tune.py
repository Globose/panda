import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, get_scorer_names
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from data import encode_labels, load_csv, FILES

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import data


def rand():
    df = encode_labels(load_csv(FILES[2]).dropna())
    X = df.iloc[:,[2,3,4,5,6]]
    y = df.iloc[:,1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, test_size=0.3)

    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=15)]
    max_depth = [int(x) for x in np.linspace(4,60,num=14)]
    max_depth.append(None)
    min_samples_split = [2,3,4,5,6,8,10]
    min_samples_leaf = [1,2,3,4]
    
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split':min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    
    rfc = RandomForestClassifier()
    skf = StratifiedKFold(n_splits=5)
    rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, scoring='recall',
                                   n_iter=100, cv=skf,n_jobs=-1, refit='recall_score')
    rfc_random.fit(X_train,y_train)
    print(rfc_random.best_params_)


def main():
    """Main Function"""
    df = encode_labels(load_csv(FILES[2]).dropna())
    X = df.iloc[:,[2,3,4,5,6]]
    y = df.iloc[:,1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, test_size=0.3)    

    rfc = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        'min_samples_split': [2,3,4,5],
        'n_estimators': [105,110,115,120],
        'max_depth': [13,14,15],
        'min_samples_leaf': [2,3]
    }
    
    scorers = {
        'recall_score': make_scorer(recall_score),
    }
    
    refit_score = 'recall_score'
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(rfc, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train.values, y_train.values)
    y_pred = grid_search.predict(X_test.values)
    
    print("refit_score", refit_score)
    print(grid_search.best_params_)
    print("\n cmatrix:")
    print(pd.DataFrame(confusion_matrix(y_test,y_pred),
        columns=['pred_neg', 'pred_pos'], index=['neg','pos']))
    
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='mean_test_recall_score', ascending=False)
    df2 = results[['mean_test_recall_score', 'param_max_depth',
             'param_min_samples_split', 
             'param_n_estimators']].head()
    df2.to_csv('results.csv')

if __name__ == '__main__':
    rand()




    # imp = IterativeImputer(max_iter=10, random_state=0)
    # imp.fit(X_test)
    # X_test_array = abs(np.round(imp.transform(X_test)))
    # X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
    
    # imp.fit(X_train)
    # X_train_array = abs(np.round(imp.transform(X_train)))
    # X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)