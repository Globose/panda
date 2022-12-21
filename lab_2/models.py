import pydotplus
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from data import FILES, load_csv, encode_labels, normalize


### Eval
def eval_model(correct, prediction, model_name):
    """Evaluates models"""
    scores = []
    scores.append((model_name+"_acc", accuracy_score(correct,prediction)))
    scores.append((model_name+"_pre", precision_score(correct,prediction)))
    scores.append((model_name+"_f1", f1_score(correct,prediction)))
    scores.append((model_name+"_rec", recall_score(correct,prediction)))
    return scores


### Random Forest
def random_forest_model(X,y):
    rforest = RandomForestClassifier()
    rforest.fit(X,y)
    return rforest


### Decision Tree
def generate_color_image(dtree, X):
    """Generates a colored image of a decision tree"""
    dot_data = export_graphviz(dtree, feature_names=X.columns, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    colors =  ('springgreen', 'tomato', 'white')

    for node in nodes:
        if node.get_name() not in ('node', 'edge', '"\\n"'):
            values = dtree.tree_.value[int(node.get_name())][0]
            if max(values) == sum(values):    
                node.set_fillcolor(colors[np.argmax(values)])
            else:
                node.set_fillcolor(colors[-1])
    graph.write_png('lab_2/titanic.png')


def decision_tree_model(X, y):
    """Creates a DecisionTreeClassifier"""
    dtree = DecisionTreeClassifier()
    dtree.fit(X,y)
    return dtree


### KNN
def draw_knn(dataFrame, X, y, knn):
    """Draws knn model in a window"""
    cmap_light = ListedColormap(["orange", "cyan"])
    cmap_bold = ["darkorange", "c"]

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(knn, X, cmap=cmap_light,ax=ax, 
        response_method = "predict",plot_method="pcolormesh",
        xlabel=dataFrame.columns[2],ylabel=dataFrame.columns[3],shading="auto")
    
    sns.scatterplot(x=X.iloc[:, 0],y=X.iloc[:, 1],hue=y,palette=cmap_bold,alpha=1.0,edgecolor="black")
    plt.title("Classification")
    plt.show()


def knn_model(X,y):
    """Generates knn-model"""
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X,y)
    return model


def knn_from_file(file, cols_x, col_y, norm=False, enc_lbl = False, plot=False):
    """Creates a knn model and measures accuracy"""
    df = load_csv(file).dropna()  
    if enc_lbl:
        df = encode_labels(df)
    if norm:
        df = normalize(df, limit=10)
    
    X = df.iloc[:,cols_x]
    y = df.iloc[:,col_y]
    model = knn_model(X,y)
    pred = model.predict(X)

    if plot:
        draw_knn(df, X, y, model)

    return model, accuracy_score(y, pred)


def knn_titanic():
    """K nearest neighbour"""
    print(knn_from_file(FILES[0], [2,3], 1, plot=True, norm=False)[1])
    print(knn_from_file(FILES[1], [2,3], 1, plot=True, norm=True)[1])
    print(knn_from_file(FILES[2], [2,3,4,5,6,7,8,9], 1, enc_lbl=True)[1])


def main():
    """Main function"""
    print("models.py")
    knn_titanic()

if __name__ == '__main__':
    main()
