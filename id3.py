import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute
        self.value = value
        self.result = result
        self.children = {}

def import_data():
    data = pd.read_csv(r'C:\Users\Admin\Documents\Programming\New folder\DecisionTrees\DecisionTrees\loan.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    attributes = list(range(X.shape[1]))
    return data, X, y, attributes

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, splits):
    total_entropy = entropy(y)
    weighted_entropy = 0
    for split in splits:
        split_entropy = entropy(split)
        weighted_entropy += (len(split) / len(y) * split_entropy)
    return total_entropy - weighted_entropy

def id3(X, y, attributes):
    if len(set(y)) == 1:
        return Node(result=y[0])
    if len(attributes) == 0:
        return Node(result=np.argmax(np.bincount(y)))
    best_attribute = None
    max_gain = -1
    for attribute in attributes:
        splits = []
        for value in np.unique(X[:, attribute]):
            splits.append(y[X[:, attribute] == value])
        gain = information_gain(y, splits)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    if max_gain == 0:
        return Node(result=np.argmax(np.bincount(y)))
    node = Node(attribute=best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    for value in np.unique(X[:, best_attribute]):
        idx = X[:, best_attribute] == value
        node.children[value] = id3(X[idx], y[idx], remaining_attributes)
    return node

def predict(node, sample, default_class):
    if node.result is not None:
        return node.result
    child_node = node.children.get(sample[node.attribute])
    if child_node is not None:
        return predict(child_node, sample, default_class)
    return default_class

def calculate_accuracy(tree, X_test, y_test, default_class):
    predictions = [predict(tree, sample, default_class) for sample in X_test]
    return accuracy_score(y_test, predictions), confusion_matrix(y_test, predictions)

data, X, y, attributes = import_data()

# Encode target labels with value between 0 and n_classes-1.
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
root = id3(X_train, y_train, attributes)
default_class = np.argmax(np.bincount(y_train))  # Use the most common class in training data as default
accuracy, conf_matrix = calculate_accuracy(root, X_test, y_test, default_class)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)