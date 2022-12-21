from sklearn.model_selection import train_test_split
from models import decision_tree_model, random_forest_model, knn_model, eval_model
from data import FILES, load_csv, encode_labels, normalize


def classification():
    df = encode_labels(load_csv(FILES[2]).dropna())
    df = normalize(df)
    
    X = df.iloc[:,[2,3,4,5,6,7,8,9]]
    y = df.iloc[:,1]
    
    data = train_test_split(X,y,train_size=0.7, test_size=0.3)
    evals = []
    
    dtree = decision_tree_model(data[0], data[2])
    prediction = dtree.predict(data[1])
    evals.append(eval_model(data[3], prediction, "dtree"))
    
    rforest = random_forest_model(data[0],data[2])
    prediction = rforest.predict(data[1])
    evals.append(eval_model(data[3], prediction, "rforest"))

    knn = knn_model(data[0],data[2])
    prediction = knn.predict(data[1])
    evals.append(eval_model(data[3],prediction, "knn"))
    
    return evals


def main():
    """Main function"""
    iterations = 20
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