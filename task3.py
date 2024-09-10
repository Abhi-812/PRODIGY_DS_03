from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os


iris=load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.Series(iris.target)

X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.3,random_state=42)

clf=DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")

print("\n confusion metrix:")
print(confusion_matrix(y_test,y_pred))

print("\n classification report:")
print(classification_report(y_test,y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(13,10))
plot_tree(clf, filled=True, feature_names=iris.feature_names,class_names=iris.target_names,rounded=True)
plt.show()