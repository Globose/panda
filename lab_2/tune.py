import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import data



def main():
    """Main Function"""
    df = pd.read_csv(data.FILES[2]).dropna()
    df = data.encode_labels(df)

    targets = df['Survived']
    df.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, 
        targets, stratify=targets,train_size=0.7, test_size=0.3)
    
    rfc = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        'min_samples_split': [3,5],
        'n_estimators': [100,300],
        'max_depth': [3,5],
        'max_features': [3,5]
    }
    
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    
    refit_score = 'recall_score'
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(rfc, param_grid, 
        scoring=scorers, refit=refit_score, cv=skf,
        return_train_score=True, n_jobs=-1)
    
    grid_search.fit(X_train.values, y_train.values)
    y_pred = grid_search.predict(X_test.values)
    
    print("refit_score", refit_score)
    print(grid_search.best_params_)
    print("\n cmatrix:")
    print(pd.DataFrame(confusion_matrix(y_test,y_pred),
        columns=['pred_neg', 'pred_pos'], index=['neg','pos']))
    
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='mean_test_precision_score', ascending=False)
    df2 = results[['mean_test_precision_score', 'mean_test_recall_score',
             'mean_test_accuracy_score', 'param_max_depth',
             'param_max_features', 'param_min_samples_split', 
             'param_n_estimators']].head()
    df2.to_csv('results.csv')

if __name__ == '__main__':
    main()
    
    #TODO: imputation (missing values)