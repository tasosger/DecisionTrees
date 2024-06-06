import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

def importdata():
    balance_data = pd.read_csv(
            #'https://archive.ics.uci.edu/ml/machine-learning-' +
            #'databases/balance-scale/balance-scale.data',
    file_path,
        sep=',', header=0)

   
    
    return balance_data


def splitdata(mydata):
    X = mydata.iloc[:, 1:-1]  
    Y = mydata.iloc[:, -1] 
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
    return X,Y,X_train,X_test,Y_train,Y_test

def gini_train(X_train, X_test, Y_train):
    clf_gini= DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5 )
    clf_gini.fit(X_train,Y_train)
    return clf_gini
def preprocess_data(data):
    if data is not None:
     
        label_encoders = {}
        categorical_columns = ['Gender', 'Dependents','Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        
       
        data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        
        data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)  
        
      
        
    return data, label_encoders


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
   
    mydata = importdata()
    
 
    preprocessed_data, _ = preprocess_data(mydata)
    
   
    X, Y, X_train, X_test, y_train, y_test = splitdata(preprocessed_data)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    clf_gini = gini_train(X_train, X_test, y_train,class_weight)
    clf_entropy = entropy_train(X_train, X_test, y_train,class_weight)
    
    y_pred_gini = prediction(X_test, clf_gini)
    y_pred_entropy = prediction(X_test, clf_entropy)
  
    cal_accuracy(y_test, y_pred_gini)
    cal_accuracy(y_test,y_pred_entropy)

    plot_decision_tree(clf_gini, ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], ['denied', 'approved'])
    plot_decision_tree(clf_entropy, ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], ['denied', 'approved'])