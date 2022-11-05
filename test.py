import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import pydotplus

df = pd.read_csv("Grilla.csv", dtype="category", sep=",", encoding="utf-8")

# for catindex, catname in enumerate(df.columns):
#     print("{0:10s} = {1}".format(catname, df[catname].cat.categories))

df_coded = pd.DataFrame()
# print()
for catname in (df.columns):
    df_coded[catname] = df[catname].astype('category').cat.codes
    
# print(df_coded)

X = df_coded.iloc[:,1:5]
y = df_coded.iloc[:,5]

print(df_coded)
# print()
# print(y)

#dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
#dtree.fit(X,y)

# graph = Source(export_graphviz(dtree, out_file=None, feature_names=X.columns))
# graph.format = 'png'
# graph.render('grilla_tree', view=False)

# Generarea snyggare färglagd trädstruktur (vita noder, negativa löv som röda och positiva lov som gröna) till PNG-fil.

# dot_data = export_graphviz(dtree, feature_names=X.columns, out_file=None, filled=True, rounded=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# nodes = graph.get_node_list()
# colors =  ('springgreen', 'tomato', 'white')

# for node in nodes:
#     if node.get_name() not in ('node', 'edge', '"\\n"'):
#         values = dtree.tree_.value[int(node.get_name())][0]
#         #color only nodes where only one class is present
#         if max(values) == sum(values):    
#             node.set_fillcolor(colors[np.argmax(values)])
#         #mixed nodes get the default color
#         else:
#             node.set_fillcolor(colors[-1])
# graph.write_png('grilla_tree_color.png')

