import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn import tree

def importdata():
    balance_data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-' +
        'databases/balance-scale/balance-scale.data',
        sep=',', header=None)

    #print("Dataset Length: ", len(balance_data))
    #print("Dataset Shape: ", balance_data.shape)
    #print("Dataset: ", balance_data.head())
    
    return balance_data


def splitdata(mydata):
    X = mydata.values[:,1:15]
    Y = mydata.values[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
    return X,Y,X_train,X_test,Y_train,Y_test

def gini_train(X_train, X_test, Y_train):
    clf_gini= DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5 )
    clf_gini.fit(X_train,Y_train)
    return clf_gini

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    print("Report : ",
          classification_report(y_test, y_pred))

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()


if __name__ == "__main__":
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdata(data)

    clf_gini = gini_train(X_train, X_test, y_train)
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    plot_decision_tree(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])
    
mydata = importdata()

