
"""Iris_Classification_MLP_Classifier
Original file is located at
    https://colab.research.google.com/drive/18FjznL1b8IfBx8pejr7nbAljzBdlB0os - Jyoti Prajapati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from sklearn import datasets

#Load Iris dataset
iris = datasets.load_iris()
print(iris.DESCR)

"""Print Target names"""

print(iris.target_names)

print(iris.data)

print(iris.target)

#DataFrame
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['target'] =iris.target
df.head

df.shape

df.describe().transpose()

#Split for tranining and Testing
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test= train_test_split(X,y)

X_train.shape

X_test.shape

"""#MLP Classifier Function"""

#Initialization '
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

#Fitting the training data to the network
mlp.fit(X_train, y_train)

#Predicting y for X_test
y_pred = mlp.predict(X_test)

#Confusion matrix
confusion_matrix(y_pred, y_test)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements
print("Accuracy of MLPClassifier :", accuracy(confusion_matrix(y_pred, y_test)))

"""#Error Plots"""

plt.ylabel('cost')
plt.xlabel=('iteration')
plt.title('Learning Rate='+str(0.001))
plt.plot(mlp.loss_curve_)
plt.show()
