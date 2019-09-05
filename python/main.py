import monkdata as m
from dtree import *


################################# Assignment 0 #################################
# Each one of the datasets has properties which makes them hard to learn.
# Motivate which of the three problems is most difficult for a decision tree
# algorithm to learn.
#
# I believe MONK-2 to be the most difficult for a decision tree algorithm to
# learn, since "ai = 1 for exactly two i of {1, 2, ..., 6} is difficult to
# express concisely with binary questions. Low information gain from each
# sub question. On the other hand, it has more training data...
#
# MONK-3 has the least amount of training data. Apart from that, it also has
# random noise added.
#
# Then again, MONK-2 has the lowest entropy.
################################################################################

################################# Assignment 1 #################################
# Calculate the entropy of the training datasets.

entropy_monk1 = entropy(m.monk1)    # Yields 1.0 because there's a 50/50 split.
entropy_monk2 = entropy(m.monk2)    # Yields 0.9571....
entropy_monk3 = entropy(m.monk3)    # yields 0.9998....
print()
print(entropy_monk1)
print(entropy_monk2)
print(entropy_monk3)
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
# Use the function averageGain (defined in dtree.py) to calculate the expected
# information gain corresponding to each of the six attributes. Note that the
# attributes are represented as instances of the class Attribute (defined in
# monkdata.py) which you can access via m.attributes[0], ..., m.attributes[5].
# Based on the results, which attribute should be used for splitting the
# examples at the root node?

datasets_train = [m.monk1, m.monk2, m.monk3]
num_of_attributes = len(m.attributes)
avg_gain_dict = dict()
for i, ds in enumerate(datasets_train):
    ds_key = "monk" + str(i+1)
    avg_gain_sub_dict = dict()
    for j in range(num_of_attributes):
        attribute_key = "A" + str(j+1)
        attribute = m.attributes[j]
        avg_gain_sub_dict[attribute_key] = averageGain(ds, attribute)
    avg_gain_dict[ds_key] = avg_gain_sub_dict

print()
[print(x, '\n') for x in avg_gain_dict.values()]
# [print(*x, sep='\n') for x in [list(avg_gain_dict[y].values()) for y in avg_gain_dict.keys()]]

# Answer:   A5 has the highest average gain in the monk1 dataset:
#           0.28703074971578435
#           A5 has the highest average gain in the monk2 dataset.
#           0.01727717693791797
#           A2 has the highest average gain in the monk3 dataset (A5 close 2nd).
#           0.29373617350838865
################################################################################

################################# Assignment 3 #################################
# For splitting we choose the attribute that maximizes the information gain,
# Eq.3. Looking at Eq.3 how does the entropy of the subsets, Sk, look like when
# the information gain is maximized? How can we motivate using the information
# gain as a heuristic for picking an attribute for splitting? Think about
# reduction in entropy after the split and what the entropy implies.

# Answer: The information gain is maximized when the entropy of the subsets, Sk,
#         is minimized.
#               Motivation: 

################################################################################






















################################################################################
