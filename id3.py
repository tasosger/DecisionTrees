import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image, display
import matplotlib.pyplot as plt

class Node:
    def __init__(self,attribute=None,value=None,result=None):
        self.attribute = attribute  
        self.value = value          
        self.result = result        
        self.children = {} 

def import_data():
    data = pd.read_csv(r'C:\Users\Admin\Documents\Programming\New folder\DecisionTrees\DecisionTrees\loan.csv')

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    attributes = list(range(X.shape[1]))
    return data,X,y,attributes

def entropy(y):
    _,counts = np.unique(y,return_counts=True)
    probabilities = counts/len(y)
    return -np.sum(probabilities*np.log2(probabilities))

def information_gain(y, splits):
    total_entropy = entropy(y)
    weighted_entropy = 0
    for split in splits:
        split_entropy =  entropy(split)
        weighted_entropy+=(len(split)/len(y)*split_entropy)
    return total_entropy - weighted_entropy

def id3(X,y,attributes):
    if len(set(y))==1:
        return Node(result=y[0])
    if len(attributes)==0:
        return Node(result=np.argmax(np.bicount(y)))
    best_attribute = None
    max_gain = -1

    for attribute in attributes:
        splits = []
        for value in np.unique(X[:,attribute]):
            splits.append(y[X[:,attribute]==value])
        gain = information_gain(y,splits)
        if gain>max_gain:
            max_gain=gain
            best_attribute = attribute
    if max_gain ==0:
        return Node(result=np.argmax(np.bicount(y)))
    
    node = Node(attribute=best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    for value in np.unique(X[:,best_attribute]):
        idx = X[:,best_attribute]==value
        node.children[value] = id3(X[idx], y[idx], remaining_attributes)
    return node

def plot_tree(node, depth=0):
    if node.result is not None:
        print('  ' * depth, 'Predict:', node.result)
    else:
        print('  ' * depth, 'Attribute:', node.attribute)
        for value, child in node.children.items():
            print('  ' * (depth+1), 'Value', value)
            plot_tree(child, depth + 1)





data,X,y,attributes = import_data()
feature_names = data.columns[:-1]
root = id3(X,y,attributes)
plot_tree(root)
plt.show()