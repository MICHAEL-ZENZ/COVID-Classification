import numpy as np

def Precision(confusionMatrix, num_class):
    precision = []
    for c in range(num_class):
        precision.append(confusionMatrix[c,c]/confusionMatrix[c,:].sum())
    return np.array(precision)

def Recall(confusionMatrix, num_class):
    recall = []
    for c in range(num_class):
        recall.append(confusionMatrix[c,c]/confusionMatrix[:,c].sum())
    return np.array(recall)

def TPrecision(confusionMatrix):
    return confusionMatrix[0,0]/confusionMatrix[0,:].sum()

def TRecall(confusionMatrix):
    return confusionMatrix[0,0]/confusionMatrix[:,0].sum()

def Acc(confusionMatrix,num_class):
    correct = 0
    total= 0
    for c in range(num_class):
        correct += confusionMatrix[c,c]
        total += confusionMatrix[c,:].sum()
    return correct/total


def F1(precision,recall):
    return (2*precision*recall)/(precision+recall)