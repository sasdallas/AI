import os

import pydotplus as pydotplus
import sklearn

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)



from sklearn.model_selection import train_test_split


train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.40,random_state =42)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4], [3.2, 0.9], [2.9, 1.9], [2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C=75)

Classifier_LR.fit(X, y)


min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0



mesh_step_size = 0.02

"""
x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
plt.figure()
plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
plt.show()
"""





print("Now what? just gonna continue.... awkward")


print("ok")



#duhduhduhduhduh codeman

import pydotplus
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn import model_selection as cross_validate
import collections

X=[[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],[128, 32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],[171,36],[116, 25],[196,25]]
Y= ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman','Man','Woman', 'Man','Woman','Woman','Woman','Man','Woman','Woman','Man']
data_feature_names = ['height','length of hair']
X_train, X_test, Y_train, Y_test = cross_validate.train_test_split(X, Y, test_size=0.40, random_state=5)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)


prediction = clf.predict([[133,37]])
print(prediction)

dot_data = tree.export_graphviz(clf,feature_names=data_feature_names,out_file=None,filled= True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('orange', 'yellow')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():edges[edge.get_source()].append(int(edge.get_destination( )))
for edge in edges: edges[edge].sort()
for i in range(2):dest = graph.get_node(str(edges[edge][i]))[0]
dest.set_fillcolor(colors[i])
graph.write_png('Decisiontree16.png')
