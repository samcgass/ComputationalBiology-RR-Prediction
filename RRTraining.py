# Sam Gass
# scg0040
# Computational Biology
# Prokect 4
# python3 coded in Microsoft Visual Studio Code for Windows 10
# Last Modified: March 31, 2020
#
# This python script uses gradient ascent for Logistic Regression to predict residue-residue contact of proteins

from os import listdir
from random import shuffle
from random import sample
from math import log
from math import exp
from copy import copy
from pickle import Pickler


class Sequence:
    def __init__(self, name):
        self.name = name
        self.features, self.rr = self.fillFeatures(
            self.name, self.fillContacts(self.name))

    def fillContacts(self, filename):
        path = '.\\rr\\' + filename + '.rr'
        rr = []
        with open(path, 'r') as rrFile:
            lines = rrFile.readlines()
        lines.pop(0)
        for i in range(len(lines)):
            lines[i] = lines[i].split()

        for line in lines:
            rr.append((int(line[0]), int(line[1])))
        return tuple(rr)

    #   Accesses the filename and returns a tuple of tuple of the features.

    def fillFeatures(self, filename, contacts):
        path = '.\\pssm\\' + filename + '.pssm'
        features = []
        with open(path, 'r') as pssmFile:
            lines = pssmFile.readlines()
        for i in range(3):
            lines.pop(0)
        for i in range(6):
            lines.pop(-1)
        for i in range(len(lines)):
            lines[i] = lines[i].split()

        rr = []
        empty = [-1] * 20
        for i in range(len(lines)):
            row = []
            if i - 2 < 0:
                row += empty
            else:
                for k in range(2, 22):
                    row.append(int(lines[i-2][k]))
            if i - 1 < 0:
                row += empty
            else:
                for k in range(2, 22):
                    row.append(int(lines[i-1][k]))
            for k in range(2, 22):
                row.append(int(lines[i][k]))
            if i + 1 >= len(lines):
                row += empty
            else:
                for k in range(2, 22):
                    row.append(int(lines[i+1][k]))
            if i + 2 >= len(lines):
                row += empty
            else:
                for k in range(2, 22):
                    row.append(int(lines[i+2][k]))
            j = i + 6
            while j < len(lines):
                jRow = []
                if j - 2 < 0:
                    jRow += empty
                else:
                    for k in range(2, 22):
                        jRow.append(int(lines[j-2][k]))
                if j - 1 < 0:
                    jRow += empty
                else:
                    for k in range(2, 22):
                        jRow.append(int(lines[j-1][k]))
                for k in range(2, 22):
                    jRow.append(int(lines[j][k]))
                if j + 1 >= len(lines):
                    jRow += empty
                else:
                    for k in range(2, 22):
                        jRow.append(int(lines[j+1][k]))
                if j + 2 >= len(lines):
                    jRow += empty
                else:
                    for k in range(2, 22):
                        jRow.append(int(lines[j+2][k]))
                # Since i starts at 0 but rr indexes start at 1, +1 to i and j
                if (i + 1, j + 1) in contacts:
                    rr.append(True)
                else:
                    rr.append(False)

                features.append(tuple(row + jRow))
                j += 1
        return tuple(features), tuple(rr)


class DataPoint:
    def __init__(self, feature, contact):
        self.feature = feature
        self.contact = contact

# -------------------------------------------------------------------------------


def getData():
    filenames = listdir('.\\rr')
    data = []
    for name in filenames:
        name = name[0:-3]
        m = Sequence(name)
        for i in range(len(m.rr)):
            d = DataPoint(m.features[i], m.rr[i])
            data.append(d)
    return data


def splitData(data, percent):
    positiveData = []
    negativeData = []
    for d in data:
        if d.contact:
            positiveData.append(d)
        else:
            negativeData.append(d)

    shuffle(positiveData)
    shuffle(negativeData)

    positiveTraining = []
    negativeTraining = []
    length = percent * len(positiveData)
    while length > 0:
        positiveTraining.append(positiveData.pop())
        negativeTraining.append(negativeData.pop())
        length -= 1

    training = positiveTraining + negativeTraining
    testing = positiveData + negativeData

    return training, testing

# ------------------------------------------------------------------------------


def linearClassification(data, naught, weights):
    sum = 0
    for i in range(len(data.feature)):
        sum += weights[i] * data.feature[i]
    return naught + sum


def logistic(x):
    if abs(x) > 709:
        print('overflow')
        return 1 / (1 + exp(709))
    return 1 / (1 + exp(x))


def gradientAscent(data, step=0.001, stop=0.01, sampleSize=100):
    numOfFeatures = len(data[0].feature)
    naught = 0
    newNaught = naught
    weights = [0] * numOfFeatures
    newWeights = copy(weights)
    avgChange = float("inf")
    iterations = 0
    while abs(avgChange) > stop and iterations < 2000:
        sampleData = sample(data, sampleSize)
        avgChange = 0
        #   w0
        change = 0
        for d in sampleData:
            change += int(d.contact) - (1 -
                                        logistic(linearClassification(d, naught, weights)))
        change *= step
        avgChange += change
        newNaught = naught + change
        #   wi
        for i in range(numOfFeatures):
            change = 0
            for d in sampleData:
                prob = 1 - logistic(linearClassification(d, naught, weights))
                term = int(d.contact) - prob
                change += d.feature[i] * term
            change *= step
            avgChange += change
            newWeights[i] = weights[i] + change
        #   update
        weights = copy(newWeights)
        naught = newNaught
        avgChange /= (numOfFeatures + 1)
        iterations += 1
    print('iterations: ', iterations)
    return weights, naught

# -------------------------------------------------------------------------------


def pickleModel(modelname, w, n):
    with open(modelname, "wb") as f:
        p = Pickler(f)
        p.dump(w)
        p.dump(n)

# -------------------------------------------------------------------------------


def testModel(data, weights, naught):
    predictions = []
    for d in data:
        prob = 1 - logistic(linearClassification(d, naught, weights))
        if prob > 0.5:
            predictions.append((prob, d.contact))
    predictions = sorted(predictions, key=lambda tup: tup[0], reverse=True)

    correct = 0
    incorrect = 0
    for i in range(int(len(predictions) / 10)):
        if predictions[i][1]:
            correct += 1
        else:
            incorrect += 1

    if correct == 0:
        L10 = 0.0
    else:
        L10 = correct / (incorrect + correct)

    correct = 0
    incorrect = 0
    for i in range(int(len(predictions) / 5)):
        if predictions[i][1]:
            correct += 1
        else:
            incorrect += 1

    if correct == 0:
        L5 = 0.0
    else:
        L5 = correct / (incorrect + correct)

    correct = 0
    incorrect = 0
    for i in range(int(len(predictions) / 2)):
        if predictions[i][1]:
            correct += 1
        else:
            incorrect += 1

    if correct == 0:
        L2 = 0.0
    else:
        L2 = correct / (incorrect + correct)

    print("Model Complete")
    print("______________________________")
    print("L10: ", L10)
    print("L5: ", L5)
    print("L2: ", L2)


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    dataSet = getData()
    percent = 0.75  # percent of the data that is for training
    trainingData, testingData = splitData(dataSet, percent)

    step = 0.0001
    stop = 0.00001
    sampleSize = 100
    weights, naught = gradientAscent(trainingData, step, stop, sampleSize)

    modelname = "RRmodel.pkl"
    pickleModel(modelname, weights, naught)

    testModel(testingData, weights, naught)
