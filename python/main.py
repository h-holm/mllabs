import monkdata as m
import dtree as d
from drawtree_qt5 import *
import random


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]



################################# Assignment 0 #################################
# Each one of the datasets has properties which makes them hard to learn.
# Motivate which of the three problems is most difficult for a decision tree
# algorithm to learn.
#
# I believe MONK-2 to be the most difficult for a decision tree algorithm to
# learn, since "ai = 1 for exactly two i of {1, 2, ..., 6} is difficult to
# express concisely with binary questions. Low information gain from each
# sub question. On the other hand, it has more training data.
#
# MONK-3 has the least amount of training data. Apart from that, it also has
# random noise added.
#
# MONK-2 has the lowest entropy.
################################################################################


################################# Assignment 1 #################################
# Calculate the entropy of the training datasets.

print("\n\nAssignment 1 - Calculate the entropy of each training dataset")

entropy_monk1 = d.entropy(m.monk1)    # Yields 1.0 because there's a 50/50 split.
entropy_monk2 = d.entropy(m.monk2)    # Yields 0.9571....
entropy_monk3 = d.entropy(m.monk3)    # yields 0.9998....
print()
print("monk1: ", entropy_monk1)
print("monk2: ", entropy_monk2)
print("monk3: ", entropy_monk3)
################################################################################


################################# Assignment 2 #################################
# Explain entropy for a uniform distribution and a non-uniform distribution,
# present some example distributions with high and low entropy.
#
# "(Shannon) entropy is a measure of uncertainty"
#   In the case of a uniform distribution, different outcomes have an equal
# probability of being picked. An example is a (non-weighted) die, which would
# therefore have an entropy of 1.
#   A non-uniform distribution is biased towards one/several outcomes, meaning
# that certain outcomes have a higher probability than others. In this case,
# the entropy is lower than 1. An example of a distribution with low(er) entropy
# is the probability distribution of students getting {A, B, C, D, E, F} as
# their final grade in a course. Certain grades are more common than others.
#   A 0 entropy distribution is one where P for a certain outcome equals 1. In
# this case, we already know the outcome.
################################################################################


################################# Assignment 3 #################################
# Use the function d.averageGain (defined in dtree.py) to calculate the expected
# information gain corresponding to each of the six attributes. Note that the
# attributes are represented as instances of the class Attribute (defined in
# monkdata.py) which you can access via m.attributes[0], ..., m.attributes[5].
# Based on the results, which attribute should be used for splitting the
# examples at the root node?


def get_avg_gain_dict(dataset):
    avg_gain_dict = dict()
    for i in range(len(m.attributes)):
        attribute_key = "A" + str(i+1)
        attribute = m.attributes[i]
        avg_gain = d.averageGain(dataset, attribute)
        avg_gain_dict[attribute_key] = avg_gain
    return avg_gain_dict


datasets_train = [m.monk1, m.monk2, m.monk3]
avg_gain_dict = dict()
for i, ds in enumerate(datasets_train):
    ds_key = "monk" + str(i+1)
    avg_gain_sub_dict = get_avg_gain_dict(ds)
    avg_gain_dict[ds_key] = avg_gain_sub_dict


print("\n\nAssignment 3 - Where do we find the highest average gain?\n")

# print(avg_gain_dict)
[print(name, ": ", max(monk_dict.items(), key=lambda item:item[1])) for name, monk_dict in avg_gain_dict.items()]
print()
# print(avg_gain_dict["monk1"])

# Answer:   A5 has the highest average gain in the monk1 dataset:
#           0.28703074971578435
#           A5 has the highest average gain in the monk2 dataset.
#           0.01727717693791797
#           A2 has the highest average gain in the monk3 dataset (A5 close 2nd).
#           0.29373617350838865
#           But on average, A5 yields the highest information gain.

#                       Information Gain
#   Dataset     a1      a2      a3      a4      a5      a6
#   MONK-1    0.0753  0.0058  0.0047  0.0263  0.2870  0.0007
#   MONK-2    0.0037  0.0025  0.0010  0.0157  0.0172  0.0062
#   MONK-3    0.0071  0.2937  0.0008  0.0029  0.2559  0.0071
################################################################################


################################# Assignment 4 #################################
# For splitting we choose the attribute that maximizes the information gain,
# Eq.3. Looking at Eq.3 how does the entropy of the subsets, Sk, look like when
# the information gain is maximized? How can we motivate using the information
# gain as a heuristic for picking an attribute for splitting? Think about
# reduction in entropy after the split and what the entropy implies.

# Answer: Entropy can be defined as uncertainty.
#
#         The information gain is maximized when the entropy of the subsets, Sk,
#         is minimized. After the split, there should be less uncertainty (i.e.
#         lower entropy).
#
#         Motivation:
#         Information gain can be defined as
#                   IG = EntropyBeforeSplit - AverageEntropyAfterSplit.
#
#         In other words, IG is the reduction in entropy achieved after
#         splitting the data according to an attribute. The attribute providing
#         the greatest IG is the attribute causing the greatest reduction in
#         entropy.

################################################################################


################################# Assignment 5 #################################
# Split the monk1 data into subsets according to the d.selected attribute using
# the function d.select (again, defined in dtree.py) and compute the information
# gains for the nodes on the next level of the tree. Which attributes should be
# tested for these nodes?
#       For the monk1 data draw the decision tree up to the first two levels
# and assign the majority class of the subsets that resulted from the two splits
# to the leaf nodes. You can use the predefined function mostCommon
# (in dtree.py) to obtain the majority class for a dataset.
#       Now compare your results with that of a predefined routine for ID3. Use
# the function buildTree(data, m.attributes) to build the decision tree. If you
# pass a third, optional, parameter to buildTree, you can limit the depth of the
# generated tree.


def get_avg_gain_dict_exclude(dataset, exclude=[]):
    avg_gain_dict = dict()
    for i in range(len(m.attributes)):
        if i not in exclude:
            attribute_key = "A" + str(i+1)
            attribute = m.attributes[i]
            avg_gain = d.averageGain(dataset, attribute)
            avg_gain_dict[attribute_key] = avg_gain

    return avg_gain_dict


d.selected_attribute = "A5"
print("\nAssignment 5.1 a) - Split monk1 into subsets according to selected attribute {}\n".format(d.selected_attribute))
idx = int(d.selected_attribute[-1]) - 1
subset_A5_true = d.select(m.monk1, m.attributes[idx], True)
subset_A12346 = [x for x in m.monk1 if x not in subset_A5_true]

print("\nAssignment 5.1 b) - Where do we find the highest average gain?")
IG_dict_A12346 = get_avg_gain_dict_exclude(subset_A12346, exclude=[idx])
IG_dict_A12346 = sorted(IG_dict_A12346.items(), key=lambda kv: kv[1], reverse=True)
# print("\n", IG_dict_A12346)
[print(tuple[0], ":    ", tuple[1]) for tuple in IG_dict_A12346]
print()

d.selected_attribute = "A1"
print("\nAssignment 5.1 c) - Split into further subsets according to selected attribute {}\n".format(d.selected_attribute))
old_idx = idx
idx = int(d.selected_attribute[-1]) - 1
subset_A1_true = d.select(subset_A12346, m.attributes[idx], True)
subset_A2346 = [x for x in subset_A12346 if x not in subset_A1_true]

# print(d.buildTree(m.monk1, m.attributes, 3))
# drawTree(d.buildTree(m.monk1, m.attributes, 10))
print()

# Build the full decision trees for all three Monk datasets using buildTree.
# Then, use the function check to measure the performance of the decision tree
# on both the training and test datasets.
#       Compute the train and test set errors for the three Monk datasets for
# the full trees. Were your assumptions about the datasets correct? Explain the
# results you get for the training and test datasets.
t1 = d.buildTree(m.monk1, m.attributes)
t2 = d.buildTree(m.monk2, m.attributes)
t3 = d.buildTree(m.monk3, m.attributes)
print("Monk1 training\t", d.check(t1, m.monk1))
print("Monk2 training\t", d.check(t2, m.monk2))
print("Monk3 training\t", d.check(t3, m.monk3))
print("Monk1 test\t", d.check(t1, m.monk1test))
print("Monk2 test\t", d.check(t2, m.monk2test))
print("Monk3 test\t", d.check(t3, m.monk3test))
print()
################################################################################


################################# Assignment 5 #################################
print("\nAssignment 6 - Prune the trees")
monk1train, monk1val = partition(m.monk1, 0.6)
monk2train, monk2val = partition(m.monk2, 0.6)
monk3train, monk3val = partition(m.monk3, 0.6)
tree_1 = d.buildTree(monk1train, m.attributes)
tree_2 = d.buildTree(monk2train, m.attributes)
tree_3 = d.buildTree(monk3train, m.attributes)
trees = [tree_1, tree_2, tree_3]

print("Monk1 test\t", d.check(tree_1, m.monk1test))
print("Monk2 test\t", d.check(tree_2, m.monk2test))
print("Monk3 test\t", d.check(tree_3, m.monk3test))

# Recursive function for finding the best pruned tree.
# Input: the tree we want to prune and a validation dataset (val_ds).
def prune_tree(tree, val_ds, best_perf=0):
    perf = d.check(tree, val_ds)
    if perf > best_perf:
        best_perf = perf
    else:
        return perf, tree

    all_pruned = d.allPruned(tree)

    for t in all_pruned:
        tmp_perf, tmp_tree = prune_tree(t, val_ds, best_perf)
        if tmp_perf > best_perf:
            best_perf = tmp_perf
            tree = tmp_tree

    return best_perf, tree

best_performance, best_tree = prune_tree(tree_1, monk1val)
print(best_performance)
# print(best_performance, "\n\n", best_tree)


################################################################################
