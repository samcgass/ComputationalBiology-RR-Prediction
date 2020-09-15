# Sam Gass
# scg0040
# Computational Biology
# Prokect 4
# python3 coded in Microsoft Visual Studio Code for Windows 10
# Last Modified: March 31, 2020
#
# This python script

from sys import argv
from math import exp
from pickle import load


def openModel(modelname):
    try:
        with open(modelname, 'rb') as f:
            return load(f), load(f)
    except:
        print("Error opening model. Model must be a pkl file for RR classification.")
        exit()


def validateArgs():
    if (len(argv) < 3):
        print("Error. Insufficent arguments. Requires a model and pssm file.")
        exit()


def fileToMatrix(filename):
    features = []
    indices = []
    with open(filename, 'r') as pssmFile:
        lines = pssmFile.readlines()
    for i in range(3):
        lines.pop(0)
    for i in range(6):
        lines.pop(-1)
    for i in range(len(lines)):
        lines[i] = lines[i].split()

    empty = [-1] * 20
    sequence = []
    for i in range(len(lines)):
        sequence.append(lines[i][1])
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

            features.append(tuple(row + jRow))
            indices.append((i + 1, j + 1))
            j += 1
    return tuple(features), tuple(indices), tuple(sequence)


def predict(weights, naught, filename):
    data, indices, sequence = fileToMatrix(filename)
    predictions = []
    rrName = filename[:-5] + "_prediction.rr"
    with open(rrName, 'w') as f:
        for c in sequence:
            f.write(c)
        f.write('\n')
        for d in range(len(data)):
            sum = 0
            for i in range(len(data[0])):
                sum += (data[d][i] * weights[i])
            sum += naught
            if sum > 709:
                prob = 1 - (1 / (1 + exp(709)))
            else:
                prob = 1 - (1 / (1 + exp(sum)))
            if (prob) > 0.5:
                predictions.append((indices[d], prob))
        predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True)
        for p in predictions:
            f.write(str(p[0][0]) + ' ' + str(p[0][1]) +
                    ' 0 8 ' + str(p[1]) + '\n')


if __name__ == "__main__":
    validateArgs()
    weights, naught = openModel(argv[1])
    filename = argv[2]
    predict(weights, naught, filename)
    print('Classification complete. Results in ' +
          filename[0:-5] + '_prediction.rr')
