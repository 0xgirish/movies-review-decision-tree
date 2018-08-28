class dataClass:
    def __init__(self, tag, review=None):
        self.tag = tag
        self.review = review

# Leaf class for decision tree
class Leaf:
    def __init__(self, result, pos=0, Total=0):
        self.result = result
        self.pos = pos
        self.Total = Total

# Decision node class for the tree
class DecisionNode:
    def __init__(self, positive_branch, negative_branch, ind, total=0, pos=0):
        self.pbranch = positive_branch
        self.nbranch = negative_branch
        self.index = ind
        self.Total = total
        self.pos = pos