import numpy as np
from sklearn.tree import export_graphviz
import pydotplus

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



