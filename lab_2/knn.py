from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

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

    d = {'Age':[2,2], 'SibSp':[1,0]}
    df = pd.DataFrame(data=d)
    op = knn.predict(df)
    print(op)

    # print(dataFrame.columns[1])
    
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
    dataFrame = load_csv("lab_2/1-titanic-small.csv")
    knn(dataFrame)

if __name__ == '__main__':
    main()