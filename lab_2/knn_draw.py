from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

def draw(dataFrame, X, y, knn):
    cmap_light = ListedColormap(["orange", "cyan"])
    cmap_bold = ["darkorange", "c"]

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(knn, X, cmap=cmap_light,ax=ax, 
        response_method = "predict",plot_method="pcolormesh",
        xlabel=dataFrame.columns[2],ylabel=dataFrame.columns[3],shading="auto")
    
    sns.scatterplot(x=X.iloc[:, 0],y=X.iloc[:, 1],hue=y,palette=cmap_bold,alpha=1.0,edgecolor="black")
   
    plt.title("Classification")
    plt.show()