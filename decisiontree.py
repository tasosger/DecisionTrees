import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def importdata():
    file_path = r'./train.csv'  # Provide the correct path to your CSV file
    
    balance_data = pd.read_csv(
            #'https://archive.ics.uci.edu/ml/machine-learning-' +
            #'databases/balance-scale/balance-scale.data',
    file_path,
        sep=',', header=0)
    train_data= pd.read_csv(
        #'https://archive.ics.uci.edu/ml/machine-learning-' +
        #'databases/balance-scale/balance-scale.data',
    file_path2,
        sep=',', header=0)
    #balance_data.append(train_data,ignore_indes=True)
        

    #print("Dataset Length: ", len(balance_data))
    #print("Dataset Shape: ", balance_data.shape)
    #print("Dataset: ", balance_data.head())
    
    return balance_data


def splitdata(mydata):
    X = mydata.iloc[:, 1:-1]  # Selecting columns from 2nd to 2nd from the end
    Y = mydata.iloc[:, -1] 
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
    return X,Y,X_train,X_test,Y_train,Y_test

def gini_train(X_train, X_test, Y_train):
    clf_gini= DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=20,min_samples_leaf=10 )
    clf_gini.fit(X_train,Y_train)
    return clf_gini
def entropy_train(X_train, X_test, Y_train):
    clf_gini= DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=20,min_samples_leaf=10)
    clf_gini.fit(X_train,Y_train)
    return clf_gini
def preprocess_data(data):
    if data is not None:
        # Handle '3+' in the 'Education' column
        
        # Convert categorical columns to numeric values using LabelEncoder
        label_encoders = {}
        categorical_columns = ['Gender', 'Dependents','Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        
        # Convert 'Education' column to numeric after handling '3+'
        
        
        # Handle missing values (example: fill with mean for numeric columns)
        # Assuming LoanAmount and Loan_Amount_Term are numeric columns with missing values
        data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        # Assuming Credit_History is also a numeric column with missing values
        data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)  # Fill with mode
        
        # Optionally, you might want to convert other columns to appropriate data types
        
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
    # Import data
    mydata = importdata()
    
    # Preprocess data
    preprocessed_data, _ = preprocess_data(mydata)
    
    # Split data into train and test sets
    X, Y, X_train, X_test, y_train, y_test = splitdata(preprocessed_data)
    
    # Train the decision tree classifier using entropy criterion
    clf_gini = gini_train(X_train, X_test, y_train)
    
    # Make predictions on the test set
    y_pred_gini = prediction(X_test, clf_gini)
    
    # Calculate accuracy
    cal_accuracy(y_test, y_pred_gini)
    
    # Plot the decision tree
    plot_decision_tree(clf_gini, ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], ['denied', 'approved'])
