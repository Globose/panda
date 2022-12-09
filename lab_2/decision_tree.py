import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
import pydotplus

def load_csv(path):
    """Loads csv file"""
    dataFrame = pd.read_csv(path, sep=",", encoding="utf-8")
    return dataFrame

def create_decision_tree(dataFrame):
    """Creates a desicion tree"""
    X = dataFrame.iloc[:,1:3]
    y = dataFrame.iloc[:,3]
    dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    dtree.fit(X,y)
    generate_color_image(dtree, X)


def generate_image(dtree, X):
    """Generates an image of a decision tree"""
    graph = Source(export_graphviz(dtree, out_file=None, feature_names=X.columns))
    graph.format = 'png'
    graph.render('grilla_tree', view=True)


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


def main():
    """Main function"""
    dataFrame = load_csv("lab_2/1-titanic-small.csv")
    create_decision_tree(dataFrame)

if __name__ == '__main__':
    main()




