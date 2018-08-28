from preprocess.preprocess import process, cleanData, processAttr, processArgv, getAttrTree, getAttrForest,  getXY
from preprocess.config import PATH
from decisiontree import idThree, predict, Depth, REP, Nodes, accuraccy, majorityAccuracy, noiseStudy

import numpy as np
import sys
from copy import deepcopy


def main():
    expNo = processArgv(sys.argv)
    if expNo < 2 or expNo >5:
        print("Please provide experiment no[2-5] to run")
        exit(-1)

    print("Would you like to pre-process data or use pre-processed data file")
    print("1. To use pre-processed data file")
    print("2. To pre-process data")
    print("Enter option[1/2] :  ", end=" ")
    option = input()
    if option.isnumeric():
        option = True if int(option) == 2 else False
    else:
        print("Invalid option")
        option = True

    if expNo == 2:
        attr, train, test = getAttrTree(option)
        x, y = getXY(train)
        
        t_x, t_y = getXY(test)

        attr_length = len(attr)
        print("number of attributes = {}".format(attr_length))
        print("Training and Test set size = {}".format(len(y)))
        print("initiating decision tree classifier . . .")

        root = idThree( x, y, attr_length)

        accrcy = accuraccy(root, t_x, t_y)
        print("Training completed")
        print("Training set accuracy = {}".format(accuraccy(root, x, y)))
        print("Test set accuracy = {}".format(accrcy))
        print("Number of nodes = {}".format(Nodes(root)))
        print("Depth of the tree = {}".format(Depth(root)))
        print()
        muList = [0.2, 0.5, 0.8, 0.9]
        print("Effect of early stoping")
        for mu in muList:
            print("Early stoping criteria entropy <= {}".format(mu))
            root = idThree(x, y, attr_length, mu, True)
            print("Completed")
            print("Training set accuracy = {}".format(accuraccy(root, x, y)))
            print("Test set accuracy = {}".format(accuraccy(root, t_x, t_y)))
            print("Number of nodes = {}".format(Nodes(root)))
            print("Depth of the tree = {}".format(Depth(root)))
            print()
        
    elif expNo == 4:
        attr, train, test = getAttrTree(option)
        x, y = getXY(train)
        
        t_x, t_y = getXY(test)

        attr_length = len(attr)
        print("number of attributes = {}".format(attr_length))
        print("initiating decision tree classifier . . .")

        root = idThree( x, y, attr_length)

        accrcy = accuraccy(root, t_x, t_y)
        print("Training completed")
        print("Training set accuracy = {}".format(accuraccy(root, x, y)))
        print("Test set accuracy before pruning = {}".format(accrcy))
        print("Number of nodes = {}".format(Nodes(root)))
        print("Depth of the tree = {}".format(Depth(root)))
        print("Starting pruning . . .")
        # prev = deepcopy(root)
        root = REP(root, t_x, t_y)
        # root = REP(root, t_x, t_y)

        print()
        accrcy = accuraccy(root, t_x, t_y)
        print("Pruining completed")
        print("Training set accuracy = {}".format(accuraccy(root, x, y)))
        print("Test set accuracy  = {}".format(accrcy))
        print("Number of nodes = {}".format(Nodes(root)))
        print("Depth of the tree = {}".format(Depth(root)))
    elif expNo == 3:
        attr, train, test = getAttrTree(option)
        x, y = getXY(train)
        
        t_x, t_y = getXY(test)

        attr_length = len(attr)
        print("number of attributes = {}".format(attr_length))
        print("Training and Test set size = {}".format(len(y)))
        Noise = [0.5, 1, 5, 10, 20]
        print("Noise study . . .\n")
        for noise in Noise:
            noiseStudy(deepcopy(x), deepcopy(y), t_x, t_y, noise, attr_length)
    elif expNo == 5:
        n = 25
        print("Number of attribute per tree = 2000")
        attr, trainA, testA = getAttrForest(n, option, n * 2000)
        print("Intiating random forest . . .")
        root = []

        Y = trainA[0][1]
        # print(len(trainA[0][0]), len(testA[0][0]))
        # exit(-1)
        TY = testA[0][1]
        for i in range(n):
            node = idThree(trainA[i][0], Y, len(attr[i]), True)
            node = REP(node, testA[i][0], TY)
            root.append(node)
            print("tree[{}] complete".format(i))

        for i in range(1, n+1):
            if i == 3 or i == 1 or i % 5 == 0:
                print("Accuraccy for random forest (total trees = {}) = {}".format(i, majorityAccuracy(root, testA, TY, i)))

if __name__=='__main__':
    main()
