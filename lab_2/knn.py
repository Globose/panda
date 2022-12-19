import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from decision_tree import generate_color_image
import knn_draw

FILES = ['lab_2/1-titanic-small.csv', 'lab_2/2-titanic2attr.csv', 'lab_2/3-titanic.csv', 'lab_2/4-test.csv']

def load_csv(path):
    """Loads csv file"""
    dataFrame = pd.read_csv(path, sep=",", encoding="utf-8")
    return dataFrame

def knn_model(X, y, dataFrame):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X,y)
    pred = model.predict(X)
    try:
        knn_draw.draw(dataFrame, X, y, model)
    except:
        print("failed to draw")
    return model, accuracy_score(y, pred)

def knn():
    """K nearest neighbour"""
    # Titanic-small
    dataFrame = load_csv(FILES[0])
    print(knn_model(dataFrame.iloc[:,2:4], dataFrame.iloc[:,1], dataFrame)[1])

    # Titanic-2
    dataFrame_2 = load_csv(FILES[1])
    dataFrame_2 = dataFrame_2.dropna()
    print(knn_model(dataFrame_2.iloc[:,2:4], dataFrame_2.iloc[:,1], dataFrame)[1])

    # Titanic-3 full
    dataFrame_3 = load_csv(FILES[2])
    dataFrame_3 = dataFrame_3.dropna()
    
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = LabelEncoder()
    for l in labels:
        dataFrame_3[l] = le.fit_transform(dataFrame_3[l])
    
    print(knn_model(dataFrame_3.iloc[:,2:10], dataFrame_3.iloc[:,1], dataFrame)[1])


def classification():
    dataFrame = load_csv(FILES[2]).dropna()
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = LabelEncoder()
    for l in labels:
        dataFrame[l] = le.fit_transform(dataFrame[l])
    
    X = dataFrame.iloc[:,[2,3,4,5,6,7,8,9]]
    y = dataFrame.iloc[:,1]

    data = train_test_split(X,y,train_size=0.7, test_size=0.3)
    dtree = DecisionTreeClassifier()
    dtree.fit(data[0],data[2])
    prediction = dtree.predict(data[1])
    
    hm = {}
    hm["dtree_acc"] = accuracy_score(data[3],prediction)
    hm["dtree_pre"] = precision_score(data[3],prediction)
    hm["dtree_f1"] = f1_score(data[3],prediction)
    hm["dtree_rec"] = recall_score(data[3],prediction)
    
    random_forest = RandomForestClassifier()
    random_forest.fit(data[0], data[2])
    prediction = random_forest.predict(data[1])
    
    hm["rfor_acc"] = accuracy_score(data[3],prediction)
    hm["rfor_pre"] = precision_score(data[3],prediction)
    hm["rfor_f1"] = f1_score(data[3],prediction)
    hm["rfor_rec"] = recall_score(data[3],prediction)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X,y)
    prediction = model.predict(X)
    hm["knn_acc"] = accuracy_score(data[3],prediction)
    hm["knn_pre"] = precision_score(data[3],prediction)
    hm["knn_f1"] = f1_score(data[3],prediction)
    hm["knn_rec"] = recall_score(data[3],prediction)
    return hm

def dataFrame_properties():
    dataFrame = load_csv(FILES[2]).dropna()
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = LabelEncoder()
    for l in labels:
        dataFrame[l] = le.fit_transform(dataFrame[l])
    
    print("Shape \n", dataFrame.shape, "\n")
    print("dtypes \n", dataFrame.dtypes, "\n")

    df_fem  = dataFrame[dataFrame['Sex']==0]
    df_male  = dataFrame[dataFrame['Sex']==1]
    
    print("Female passengers: ", len(df_fem))
    print("Survived: ", df_fem['Survived'].sum())
    print("Percentage: ", df_fem['Survived'].sum()/len(df_fem), "\n")
    
    print("Male passengers: ", len(df_male))
    print("Survived: ", df_male['Survived'].sum())
    print("Percentage: ", df_male['Survived'].sum()/len(df_male), "\n")


def main():
    """Main function"""
    # knn()
    # dataFrame_properties()
    # classification()
    results = {}
    for z in range(40):
        hm = classification()
        for m in hm:
            if results.get(m) is None:
                results[m] = [hm.get(m)]
            else:
                results[m].append(hm.get(m))
    for key, lst in results.items():
        print(key, sum(lst)/len(lst))
            

if __name__ == '__main__':
    main()