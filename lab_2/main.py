from sklearn.model_selection import train_test_split
from models import decision_tree_model, random_forest_model, knn_model, eval_model
from data import FILES, load_csv, encode_labels, normalize
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix




def classification():
    df = encode_labels(load_csv(FILES[2]))
    # df = normalize(df)
    X = df.iloc[:,[2,3,4,5,6]]
    y = df.iloc[:,1]
        
    imp = IterativeImputer(max_iter=20, initial_strategy='median')
    imp.fit(X)
    X_2 = abs(np.round(imp.transform(X)))
    X = pd.DataFrame(X_2, index=X.index, columns=X.columns)
        
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)
    evals = []
    
    # dtree = decision_tree_model(X_train, y_train)
    # prediction = dtree.predict(X_test)
    # evals.append(eval_model(y_test, prediction, "dtree"))
    
    rforest = random_forest_model(X_train,y_train)
    y_pred = rforest.predict(X_test)
    evals.append(eval_model(y_test, y_pred, "rforest"))
    
    # print("\n cmatrix:")
    # print(pd.DataFrame(confusion_matrix(y_test,y_pred),
    #     columns=['pred_neg', 'pred_pos'], index=['neg','pos']))

    # knn = knn_model(X_train,y_train)
    # prediction = knn.predict(X_test)
    # evals.append(eval_model(y_test,prediction, "knn"))
    
    return evals


def tuner():
    df = encode_labels(load_csv(FILES[2]))
    # df = normalize(df)
    
    X = df.iloc[:,[2,3,4,5,6]]
    y = df.iloc[:,1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)
    evals = []
    
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(X_test)
    X_test_array = abs(np.round(imp.transform(X_test)))
    X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
    
    imp.fit(X_train)
    X_train_array = abs(np.round(imp.transform(X_train)))
    X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)

    for sam_spl in range(2,7):
        for m_depth in range(1,15):
            rforest = RandomForestClassifier(min_samples_split=sam_spl, n_estimators=80, max_depth=m_depth, max_features=3)
            rforest.fit(X_train,y_train)

            y_pred = rforest.predict(X_test)
            print(sam_spl, m_depth, recall_score(y_test,y_pred))

    # print(pd.DataFrame(confusion_matrix(y_test,y_pred),
    #     columns=['pred_neg', 'pred_pos'], index=['neg','pos']))
    #return evals    


def main():
    """Main function"""
    iterations = 1
    results = {}
    for _ in range(iterations):
        models = classification()
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