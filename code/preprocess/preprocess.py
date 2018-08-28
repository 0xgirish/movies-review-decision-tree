from datetime import datetime
from inspect import getframeinfo, currentframe
from copy import deepcopy
import numpy as np
import logging
import pickle

from .config import PATH, LOGGING_FILE
from .data import dataClass

logging.basicConfig(filename=LOGGING_FILE, level=logging.INFO)
filename = getframeinfo(currentframe()).filename.split('/')[-1]


# get random reviews with size m
def data_sample(file, m):
    lines = []
    lnum = 0
    with open(file, 'r') as f:
        for line in f.readlines():
            ln = line[:-1].split(' ')
            lines.append([lnum, int(int(ln[0]) >= 7), ' '.join(ln[1:])])
            lnum += 1
    lnum = len(lines)
    # print(lnum)
    # print(lines[0])
    pos_review = np.array(lines[:int(lnum/2)-1])
    neg_review = np.array(lines[int(lnum/2)+1:])
    # print(pos_review[0])
    # print(neg_review[0])
    del lines
    np.random.shuffle(pos_review)
    np.random.shuffle(neg_review)
    # print(pos_review[0])
    # print(neg_review[0])
    review = np.append(pos_review[:int(m/2)], neg_review[:int(m/2)], axis=0)
    np.random.shuffle(review)
    return review

# get top n/2 positive attributes and n/2 negative attributes
def vocabEr(n):
    file_str = "data/imdbEr.txt"
    weight = []
    line_no = 0
    with open(file_str, "r") as f:
        for line in f.readlines():
            weight.append([line_no, float(line[:-1])])
            line_no += 1

    weight = sorted(weight, key=lambda x:x[1], reverse=True)
    if n > len(weight):
        n = len(weight)-1
    attrp = np.array(weight[:int(n/2)])
    attrn = np.array(weight[-int(n/2):])
    weight = np.append(attrn, attrp, axis=0)
    np.random.shuffle(weight)
    
    with open('code/selected-features-indices.txt', 'w') as f:
        for w in weight:
            f.write("{}\n".format(int(w[0])))

    return weight


# process data for decision tree
def process(n, m):
    attributes = vocabEr(n)
    train = data_sample('data/train/labeledBow.feat', m)
    # train = data[:m]
    # validate = data[m:]
    test = data_sample('data/test/labeledBow.feat',  m)
    return attributes, train, test

# process data for random forest
def processAttr(m, n, t):
    attr_file = "data/imdbEr.txt"
    weight = []
    line_no = 0
    with open(attr_file,  "r") as f:
        for line in f.readlines():
            weight.append([line_no, float(line[:-1])])
            line_no += 1

    if m > len(weight):
        m = len(weight) - 1
    weight = np.array(weight)
    np.random.shuffle(weight)
    attr = []
    for i in range(n):
        attr.append(weight[i * int(m/n):(i+1)*int(m/n)])

    train = data_sample('data/train/labeledBow.feat', t)
    test = data_sample('data/test/labeledBow.feat', t)
    return attr, train, test

# get matrix according to attributes for training and test sets
def cleanData(data, attr):
    # data is numpy array data[0] = line no, data[1] = (0, 1), data[2] = 'review string'
    cleaned_data = {}
    attr_line = []
    rdata = []
    for a in attr:
        attr_line.append(int(a[0]))
        cleaned_data[int(a[0])] = 0

    for d in data:
        # line = int(d[0])
        tag = int(d[1])
        review = str(d[2]).split(' ')
        # logging.info(review)
        reviewc = deepcopy(cleaned_data)
        for r in review:
            a = [int(x) for x in r.split(':')]
            # print(a)
            if a[0] in attr_line:
                reviewc[a[0]] = a[1]
        review = []
        for _, v in reviewc.items():
            review.append(v)
        del reviewc
        dc = dataClass(tag, review)
        # logging.info(reviewc)
        rdata.append(dc)

    return rdata

# get command line arguments
def processArgv(argv):
    if len(argv) > 1:
        for arg in argv[1:]:
            if arg.isnumeric():
                return int(arg)
        else:
            return -1
    else:
        return -1

# get instance and target from data
def getXY(data):
    y = []
    x = []
    for d in data:
        y.append(d.tag)
        x.append(d.review)
    y = np.array(y)
    x = np.array(x)
    return [x, y]

# get attributes , training and test set for random forest 
# Either from saved file or process
def getAttrForest(n, isProcess, m=10000, t=1000):
    print("Getting data for random forest")
    try:
        if isProcess:
            raise NameError('PreProcess')
        dataFile = open('code/random_forest', 'rb')
        data = pickle.load(dataFile)
        dataFile.close()
        return data[0], data[1], data[2]
    except (IOError, EOFError, NameError, FileNotFoundError):
        print("Getting attributes for random forest")
        attr, train, test = processAttr(m, n, t)
        trainA = []
        testA = []
        for i in range(n):
            trainA.append(getXY(cleanData(train, attr[i])))
            testA.append(getXY(cleanData(test, attr[i])))
        data = [attr, trainA, testA]
        with open('code/random_forest', 'wb') as f:
            pickle.dump(data, f)
        return attr, trainA, testA

# get attributes, training and test sets for decision tree
# Either from saved file or process data
def getAttrTree(isProcess, n=5000, m=1000):
    print("Getting data for tree")
    try:
        if isProcess:
            raise NameError('PreProcessError')
        data = open('code/tree_data', 'rb')
        dataList = pickle.load(data)
        attr, train, test= dataList[0], dataList[1], dataList[2]
        data.close()
        return attr, train, test
    except (IOError, EOFError, NameError, FileNotFoundError):
        print("Processing data . . .")
        attr, train, test = process(5000, 1000)
        train = cleanData(train, attr)
        test = cleanData(test, attr)
        # valid = cleanData(valid, attr)
        data = [attr, train, test]
        with open('code/tree_data', 'wb') as f:
            pickle.dump(data, f)
        return attr, train, test