# Mastering-Decision-Trees-A-Hands-On-Approach
In the field of machine learning, decision trees are effective tools that provide clear and understandable answers to problems with regression and classification. We'll explore decision trees in this hands-on lab by building one from the ground up and using it to solve the crucial problem of determining whether a mushroom is toxic or edible. Now let's get started and roll up our sleeves!
 
1. Packages

Let's make sure we have the resources we need before setting out on this journey. For data manipulation, we'll be mostly using Python and a couple of its libraries, like NumPy.

code in PythonCopy

bring in numpy as np

2. Problem Statement

Our job is to create a decision tree classifier that, given a set of features, can reliably identify whether a mushroom is toxic or edible. Every mushroom in our dataset will be given a label indicating whether it is "edible" or "poisonous," making this a classic binary classification problem.
 
3. Set of data

We will be working with a dataset that includes a variety of mushroom attributes and the class labels (edible or poisonous) that correspond to those attributes.

4.1 Find the Entropy

An indicator of disorder or impurity in a dataset is entropy. Entropy is computed as follows for a binary classification problem:

where the proportions of positive and negative examples in the dataset S are, respectively, 1p1​ and 2p2​.
 
Task 1
Write a function to calculate entropy given the proportions of positive and negative instances.
def compute_entropy(y):
entropy = 0.
n_edible = np.count_nonzero(y)
n_poisonous = len(y) — n_edible
# Check for no variation in the labels
if n_edible == 0 or n_poisonous == 0:
return 0
# Compute the probability of each label
p_edible = n_edible / len(y)
p_poisonous = n_poisonous / len(y)
# Entropy formula
entropy = -(p_edible * np.log2(p_edible) + p_poisonous * np.log2(p_poisonous))
return entropy
 
 
4.2 Divided Dataset

By splitting the dataset, you can create subsets according to a particular attribute's values. The goal of this procedure is to make the final subsets as homogeneous as possible.
Task 2
Write a function to split the dataset based on a given attribute and its value.
python
def split_dataset(X, node_indices, feature)
left_indices = []
right_indices = []
for idx in node_indices:
# If the feature value is 1, add the index to left_indices
if X[idx, feature] == 1:
left_indices.append(idx)
# Else, add the index to right_indices
else:
right_indices.append(idx)
return left_indices, right_indices
4.3 Calculate Information Gain
 
Information gain calculates how well an attribute does its job of classifying the data. The difference between the entropy of the parent dataset and the weighted sum of the entropies of its child subsets is how it is computed.
 
Task 3
Write a function to calculate information gain given a dataset and its subsets.
def compute_information_gain(X, y, node_indices, feature):
# Split dataset
left_indices, right_indices = split_dataset(X, node_indices, feature)
# Some useful variables
X_node, y_node = X[node_indices], y[node_indices]
X_left, y_left = X[left_indices], y[left_indices]
X_right, y_right = X[right_indices], y[right_indices]
information_gain = 0
node_entropy = compute_entropy(y_node)
left_entropy = compute_entropy(y_left)
right_entropy = compute_entropy(y_right)
# Compute the weight of each child node
left_weight = len(y_left) / len(y_node)
right_weight = len(y_right) / len(y_node)
# Compute the information gain
information_gain = node_entropy — (left_weight * left_entropy + right_weight * right_entropy)
return information_gain
 
4.4 Get Best Split
To determine the best attribute to split on, we calculate the information gain for each attribute and choose the one with the highest gain.
Task 4
Write a function to find the best attribute to split on.
def get_best_split(X, y, node_indices):
num_features = X.shape[1]
best_feature = -1
best_information_gain = -np.inf
best_threshold = None
# Iterate over each feature
for feature_index in range(num_features):
# Get unique feature values
unique_values = np.unique(X[node_indices, feature_index])
# Iterate over potential thresholds
for threshold in unique_values:
# Compute information gain for this split
information_gain = compute_information_gain(X, y, node_indices, feature_index)
# Update best split if this one is better
if information_gain > best_information_gain:
best_information_gain = information_gain
best_feature = feature_index
best_threshold = threshold
return best_feature
