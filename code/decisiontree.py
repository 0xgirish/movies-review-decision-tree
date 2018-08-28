import numpy as np
from math import log
import sys

from preprocess.data import Leaf, DecisionNode

LEAFS = 0
NODES = 0


# entropy calculation
def Entropy(y):
    if len(y) == 0:
        return 0
    selection_array = (y[:] == 1)
    p_plus = len(y[selection_array]) / len(y)
    p_minus = 1 - p_plus

    if p_plus == 0 or p_plus == 1:
        return 0
    else:
        return -(p_plus * log(p_plus, 2) + p_minus * log(p_minus, 2))


# get weighted entropy for branches
def weightedEntropy(x, y, ind):
    selection_array = x[:, ind] > 0
    set_ay = y[selection_array]
    set_by = y[x[:, ind] <= 0]
    entropy_ay = Entropy(set_ay)
    entropy_by = Entropy(set_by)
    return (len(set_ay) * entropy_ay + len(set_by) * entropy_by) / len(y)

# partition tree according to minimum entropy of two partitioned branches
def partition(x, y, attr):
    attr_length = attr
    min_entropy = -1
    index = 0
    for i in range(attr_length):
        entropy = weightedEntropy(x, y, i)


        if len(y[x[:, i] > 0]) == 0 or len(y[x[:, i] <= 0]) == 0:
            continue

        if entropy < min_entropy or min_entropy == -1:
            min_entropy = entropy
            index = i

    return min_entropy, index

# print number of nodes
def numNode():
    global NODES
    NODES += 1
    sys.stdout.write("\t\t\t[No of Leafs, Nodes: (%d, %d)]   \r" % (LEAFS, NODES) )
    sys.stdout.flush()

# print number of leafs and nodes
def numLeaf():
    global LEAFS
    global NODES
    LEAFS += 1
    NODES += 1
    sys.stdout.write("\t\t\t[No of Leafs, Nodes: (%d, %d)]   \r" % (LEAFS, NODES) )
    sys.stdout.flush()

# id3 algorithm to train tree
def idThree(x, y, attr, mu=0, leaf=False):
    if leaf:
        global LEAFS
        global NODES
        LEAFS = 0
        NODES = 0
    # print(len(y))
    entropy = Entropy(y)
    if len(y[y[:] == 1]) == 0 or len(y[y[:] == 0]) == 0 or entropy <= mu:
        result = 1 if len(y[y[:] == 1]) > len(y[y[:] != 1]) else 0
        numLeaf()
        return Leaf(result)
    # print('.', end="")
    _, index = partition(x, y, attr)
    # print(entropy, index)

    selection_array = x[:, index] > 0
    true_x = x[selection_array]
    true_y = y[selection_array]
    false_x = x[x[:, index] <= 0]
    false_y = y[x[:, index] <= 0]

    if len(true_y) == 0 or len(false_y) == 0:
        result = 1 if len(y[y[:] == 1]) > len(y[y[:] != 1]) else 0
        numLeaf()
        return Leaf(result)

    positive_branch = idThree(true_x, true_y, attr, mu)
    negative_branch = idThree(false_x,false_y, attr, mu)

    numNode()
    return DecisionNode(positive_branch, negative_branch, index)

# predict result for one instance
def predict(x, root):
    if isinstance(root, Leaf):
        return root.result

    if x[root.index] > 0:
        return predict(x, root.pbranch)
    else:
        return predict(x, root.nbranch)

# get accuracy for decision tree
def accuraccy(root, X, Y):
    p_y = []
    for x in X:
        py = predict(x, root)
        p_y.append(py)

    n = len(Y)
    right_predictions = 0
    for i in range(n):
        if p_y[i] == int(Y[i]):
            right_predictions += 1

    # print("accuraccy . . .")
    return  100 * right_predictions / n

# get accuracy for random forest
def majorityAccuracy(root, LX, Y, n):
    m = len(Y)
    p_y = []
    for j in range(m):
        one = 0
        for i in range(n):
            if predict(LX[i][0][j], root[i]) == 1:
                one += 1
        result = 1 if one > n - one else 0
        p_y.append(result)

    right_predictions = 0
    for i in range(m):
        if p_y[i] == int(Y[i]):
            right_predictions += 1

    return 100 * right_predictions / m

# get depth of the tree
def Depth(root):
    if isinstance(root, Leaf):
        return 0
    left = Depth(root.pbranch)
    right = Depth(root.nbranch)
    if left > right:
        return left + 1
    else:
        return right + 1

# count number of nodes in tree
def Nodes(root):
    if isinstance(root, Leaf):
        return 1
    nodes = 1 + Nodes(root.pbranch) + Nodes(root.nbranch)
    return nodes

# make y noisy
def getNoisy(y, error):
    n = len(y)
    isSelected = []
    for _ in range(n):
        isSelected.append(False)
    requiredChange = int(n * error / 100)
    while requiredChange > 0:
        index = np.random.randint(0, n)
        if not isSelected[index]:
            if y[index] == 1:
                y[index] = 0
            else:
                y[index] = 1
            isSelected[index] = True
            requiredChange -= 1

# noise study form given noise in data
def noiseStudy(x, y, tx, ty, noise, attr=5000):
    getNoisy(y, noise)
    print("Training tree with noise {}%".format(noise))
    root = idThree(x, y, attr, 0, True)
    print("Number of nodes = {}".format(Nodes(root)))
    print("Depth of the tree = {}".format(Depth(root)))
    print("Accuracy over train set = {}".format(accuraccy(root, x, y)))
    print("Accuracy over test set = {}".format(accuraccy(root, tx, ty)))
    print()



def REP(node, validx, validy):
    for i in range(len(validy)):
        classify(node, validx[i], validy[i])
    _ , node = prune(node)
    return node

def classify(node, vx, vy):
    node.Total += 1
    if vy == 1:
        node.pos += 1
    if not isinstance(node, Leaf):
        if vx[node.index] > 0:
            classify(node.pbranch, vx, vy)
        else:
            classify(node.nbranch, vx, vy)

def prune(node):
    if isinstance(node, Leaf):
        val = node.pos if node.result != 1 else node.Total - node.pos
        return val, node
    errp, node.pbranch = prune(node.pbranch)
    errn, node.nbranch = prune(node.nbranch)
    error = errp + errn
    if error < min(node.pos, node.Total - node.pos):
        return error, node
    else:
        pos, total = node.pos, node.Total
        result = 1 if pos > total - pos else 0
        node = Leaf(result, pos, total)
        return total - pos if result == 1 else pos, node