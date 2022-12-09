from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from matplotlib.colors import ListedColormap
import sklearn.preprocessing as skpre

def load_csv(path):
    """Loads csv file"""
    dataFrame = pd.read_csv(path, sep=",", encoding="utf-8")
    return dataFrame

def knn(dataFrame):
    """K nearest neighbour"""
    X = dataFrame.iloc[:,2:4]
    y = dataFrame.iloc[:,1]
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X,y)
    
    op = knn.predict(X)
    acc = sklearn.metrics.accuracy_score(y, op)
    
    dataFrame_2 = load_csv("lab_2/2-titanic2attr.csv")
    dataFrame_2 = dataFrame_2.dropna()
    
    X_2 = dataFrame_2.iloc[:,2:4]
    y_2 = dataFrame_2.iloc[:,1]
    
    knn_2 = KNeighborsClassifier(n_neighbors=3)
    knn_2.fit(X_2,y_2)
    
    op_2 = knn_2.predict(X_2)
    acc_2 = sklearn.metrics.accuracy_score(y_2,op_2)
    # print(acc_2)
    # draw(dataFrame, X_2, y_2, knn)
    

    dataFrame_3 = load_csv("lab_2/3-titanic.csv")
    dataFrame_3 = dataFrame_3.dropna()
    
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = skpre.LabelEncoder()
    for l in labels:
        dataFrame_3[l] = le.fit_transform(dataFrame_3[l])

    X_3 = dataFrame_3.iloc[:,2:10]
    y_3 = dataFrame_3.iloc[:,1]
    
    print(X_3)
    
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    knn_3.fit(X_3,y_3)
    
    op_3 = knn_3.predict(X_3)
    acc_3 = sklearn.metrics.accuracy_score(y_3, op_3)
    
    print(acc_3)
    #draw(dataFrame_3, X_3, y_3, knn_3)
    
def upg2():
    df = load_csv("lab_2/3-titanic.csv")
    
    labels = ['Ticket', 'Sex', 'Embarked', 'Fare']
    le = skpre.LabelEncoder()
    for l in labels:
        df[l] = le.fit_transform(df[l])
    
    # print(df.shape)
    # print(df.dtypes)
    print(df.describe())
    # print(df['Sex'].value_counts())
    
    df_fem  = df[df['Sex']==0]
    df_male  = df[df['Sex']==1]
    
    print(df_fem['Survived'].value_counts())
    print(df_male['Survived'].value_counts())
    
    print(df_fem.describe())
    print(df_male.describe())
    
def draw(dataFrame, X, y, knn):
    cmap_light = ListedColormap(["orange", "cyan"])
    cmap_bold = ["darkorange", "c"]

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        knn, X, cmap=cmap_light,
        ax=ax, response_method = "predict",
        plot_method="pcolormesh",
        xlabel=dataFrame.columns[2],
        ylabel=dataFrame.columns[3],
        shading="auto"
    )
    
    sns.scatterplot(
        x=X.iloc[:, 0],
        y=X.iloc[:, 1],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
   
    plt.title(
        "Classification"
    )
    plt.show()

def main():
    """Main function"""
    #dataFrame = load_csv("lab_2/1-titanic-small.csv")
    #knn(dataFrame)
    upg2()


if __name__ == '__main__':
    main()