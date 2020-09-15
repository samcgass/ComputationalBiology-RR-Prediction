------------------
Usage Instructions
------------------
This program was create on Windows.
From the command line, RRPredict.py takes two command line arguements.
The first is the model as a pickle file, the second is the .pssm file to be predicted.
The model is provided, it is the RRmodel.pkl file.

To run the program, from the command line type:		python RRPredict.py RRmodel.pkl [filename].pssm
Note: the .pssm file should be in the same directory as RRPredict.py.

The program will create a file in the same directory with the name [filename]_prediction.rr
This file is in the same format as the .rr files and it contains the given model's predicted outputs.
However, the last column of the rr file produced with contain the probability that the model predicts for contact.

RRTraining.py is the program that creates the RRmodel.pkl file.
There is no need to run it as I have already run it on my machine and its output, RRmodel.pkl, is given.