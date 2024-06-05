import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image, display


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
        node.children = id3(X[idx],y[idx],remaining_attributes)
    return node


def visualize_tree(node, feature_names, depth=0):
    if node.result is not None:
        return f"Leaf: {node.result}"
    else:
        children_str = "\n".join(
            f"  {feature_names[node.attribute]} = {value} -> {visualize_tree(child, feature_names, depth + 1)}"
            for value, child in node.children.items()
        )
        return f"Node: {feature_names[node.attribute]}\n{children_str}"

def plot_tree_as_figure(node, feature_names):
    from sklearn.tree import plot_tree
    from sklearn import tree

   
    def traverse_tree(node, feature_names):
        if node.result is not None:
            return {
                "name": f"Leaf: {node.result}",
                "children": []
            }
        else:
            children = [
                traverse_tree(child, feature_names)
                for value, child in node.children.items()
            ]
            return {
                "name": feature_names[node.attribute],
                "children": children
            }
    
    tree_structure = traverse_tree(node, feature_names)
    

    class DecisionTree:
        def __init__(self, structure):
            self.structure = structure
        
        def plot(self, ax):
            self._plot_node(self.structure, ax, 0.5, 1, 0.25, 0)
        
        def _plot_node(self, node, ax, x, y, x_offset, depth):
            ax.text(x, y, node["name"], ha="center", va="center", bbox=dict(facecolor='white', edgecolor='black'))
            for i, child in enumerate(node["children"]):
                x_child = x + (i - len(node["children"]) / 2) * x_offset
                y_child = y - 0.2
                ax.plot([x, x_child], [y, y_child], 'k-')
                self._plot_node(child, ax, x_child, y_child, x_offset / 2, depth + 1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()
    tree_plot = DecisionTree(tree_structure)
    tree_plot.plot(ax)
    plt.show()

data,X,y,attributes = import_data()
root = id3(X,y,attributes)
plot_tree_as_figure(root, data.columns[:-1])
