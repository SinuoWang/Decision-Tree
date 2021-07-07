from __future__ import print_function
import math
import sys

class Node:
    def __init__(self, leaf=False):
        self.attr = 0
        self.splitval = 0
        self.leaf = leaf
        self.label = None
        self.right = None


def get_labelFreq_dict(data):
     Y=[sample[1] for sample in data]
     #a dictionary relate(Ys) labels with its freq
     labels_freq_dict = {}
     for label in Y:
         if label not in labels_freq_dict:
             labels_freq_dict[label]=1
         else:
            labels_freq_dict[label]+=1
     return labels_freq_dict


def entropy(data):
    labels_freq_dict = get_labelFreq_dict(data)
    entropy = 0.0
    for label, freq in labels_freq_dict.items():
        prob = float(freq) / float(len(data))
        entropy += (-prob * math.log(prob, 2))
    return entropy


def split_data(attr,splitval,data):
    left = []
    right = []
    for sample in data:
        x = sample[0]
        if x[attr] <= splitval:
            left.append(sample)
        elif  x[attr] > splitval:
            right.append(sample)
    return left, right



def choose_split(data):
    max_gain = 0
    best_attr = 0
    best_splitval=0
    for attr in range(len(data[0][0])):
        data.sort(key= lambda x: x[0][attr])
        for sample_idx in range(len(data)-1):
                splitval = 0.5 * (data[sample_idx][0][attr] + data[sample_idx+1][0][attr])
                # prob_left = float(sample_idx+1)/float(len(data))
                # prob_right = float(1.0-prob_left)
                left, right = split_data(attr, splitval,data)
                prob_left = float(len(left))/float(len(data))
                prob_right = float(len(right))/float(len(data))
                # if len(left)!=0 and len(right)!=0:
                gain = entropy(data)-(prob_left*entropy(left)+prob_right*entropy(right))
                if gain > max_gain:
                    best_attr = attr
                    best_splitval = splitval
                    max_gain = gain
    # print("maxgain = " + str(max_gain))
    # print("splitval = " + str(best_splitval))
    return best_attr, best_splitval


def find_label(data):
    labels_freq_dict = get_labelFreq_dict(data)
    listObj = zip(labels_freq_dict.keys(), labels_freq_dict.values())
    listt = list(listObj)
    max_freq = 0
    unique_label=None
    for label, freq in listt:
        if len(data)==0:
            unique_label= None
            break
        if freq > max_freq:
            unique_label = label
            max_freq = freq
        elif freq == max_freq:
            unique_label=None
    return unique_label


def dtl(data, minleaf):
    # print(len(data))fail
    if (len(data) == 0):
        return None;
    labels_freq_dict = get_labelFreq_dict(data)
    X = [tuple(sample[0]) for sample in data]
    same_x=all(element == X[0] for element in X)
    if (len(data) <= minleaf) or len(labels_freq_dict) == 1 or same_x:
        leaf_node = Node()
        leaf_node.leaf = True
        leaf_node.label = find_label(data)
        return leaf_node
    attr, splitval = choose_split(data)
    n = Node()
    n.label = find_label(data)
    n.attr = attr
    n.splitval = splitval
    left_data, right_data = split_data(attr, splitval,data)
    # print("left = " + str(len(left_data)))
    # print("right = " + str(len(right_data)))
    n.left = dtl(left_data, minleaf)
    n.right = dtl(right_data, minleaf)
    return n


def predict_DTL(n, x):
    while not n.leaf:
        # print(str(n.splitval))
        if x[n.attr] <= n.splitval:
            nextN=n.left
        else:
            nextN=n.right
        if (nextN == None):
            break
        n = nextN
    return n.label


train_data = []
test_data = []
train_file = open(sys.argv[1])
test_file = open(sys.argv[2])
minleaf = int(sys.argv[3])

train_file.readline()
test_file.readline()

line=train_file.readline()
while line:
    sample = [float(val) for val in line.strip().split()]
    label = sample[11]
    sample.pop(11)
    train_data.append((sample, label))
    line=train_file.readline()
    if line == "":
        break

line=test_file.readline()
while line:
    sample = [float(val) for val in line.strip().split()]
    test_data.append(sample)
    line=test_file.readline()
    if line == "":
        break


decisionTree = dtl(train_data, minleaf)
for sample in test_data:
    if predict_DTL(decisionTree, sample)==None:
        print('unkonwn')
    else:
        print(int(predict_DTL(decisionTree, sample)))
