import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from data import encode_labels, load_csv, FILES


def main():
    """Main Function"""
    df = encode_labels(load_csv(FILES[2]))
    X = df.iloc[:,[2,3,4,5,6,7,8,9]]
    y = df.iloc[:,1]
        
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)
    
    imp = IterativeImputer(max_iter=15, initial_strategy='mean')
    imp.fit(X_train)
    X_2 = abs(np.round(imp.transform(X_train)))
    X_train = pd.DataFrame(X_2, index=X_train.index, columns=X_train.columns)
    
    imp = IterativeImputer(max_iter=15, initial_strategy='mean')
    imp.fit(X_test)
    X_2 = abs(np.round(imp.transform(X_test)))
    X_test = pd.DataFrame(X_2, index=X_test.index, columns=X_test.columns)
    
    
    rfc = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        'min_samples_split': [3,4,5,6,7,8],
        'n_estimators': [110],
        'max_depth': [12,13,14],
        'min_samples_leaf': [2,3,4,5]
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
    main()
