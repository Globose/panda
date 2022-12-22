from sklearn.model_selection import train_test_split
from models import eval_model
from data import FILES, load_csv, encode_labels, normalize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def classification(m_depth):
    df = encode_labels(load_csv(FILES[2]))
    X = df.iloc[:,[2,3,4,5,6,7,8,9]]
    y = df.iloc[:,1]
        
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)
    evals = []
    
    imp = IterativeImputer(max_iter=15, initial_strategy='mean')
    imp.fit(X_train)
    X_2 = abs(np.round(imp.transform(X_train)))
    X_train = pd.DataFrame(X_2, index=X_train.index, columns=X_train.columns)
    
    imp = IterativeImputer(max_iter=15, initial_strategy='mean')
    imp.fit(X_test)
    X_2 = abs(np.round(imp.transform(X_test)))
    X_test = pd.DataFrame(X_2, index=X_test.index, columns=X_test.columns)
    
    rforest = RandomForestClassifier()
    rforest.fit(X_train,y_train)
    y_pred = rforest.predict(X_test)
    
    evals.append(eval_model(y_test, y_pred, "rforest"))

    pred_proba = rforest.predict_proba(X_test)
    y_pred_proba = (pred_proba [:,1] >= 0.4).astype('int')    
    evals.append(eval_model(y_test,y_pred_proba, "proba"))
    
    return evals


def main():
    """Main function"""
    iterations = 30
    results = {}
    for i in range(iterations):
        models = classification(i+2)
        for m in models:
            for scores in m:
                if results.get(scores[0]) is None:
                    results[scores[0]] = [scores[1]]
                else:
                    results[scores[0]].append(scores[1])
    for key, lst in results.items():
        print(key, sum(lst)/len(lst))
            

if __name__ == '__main__':
    main()