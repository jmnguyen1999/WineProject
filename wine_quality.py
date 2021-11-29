#wine_quality.py
#Purpose:       Reads training data from 'winequality-red.csv' and 'winequality-white.csv' and trains a model using Naive-bayes and KNN. Using the model, it compares its predictions
#               to actual classifications to calculate accuracy, precision, f1 score, and recall. Does preprocessing of data before training model through looking at imbalance of data and
#               finding correspondence between attributes.

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Class Constants:
DATASETS = ['winequality-red.csv', 'winequality-white.csv']
NUM_ATTRIBUTES = 11
MAX_CORRELATION = 0.65
ATTRIBUTE_NAMES = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]


#Purpose:       Calculate one of these types: precision, recall, or f1 score, given the number of true positives, false positives, and false negatives. Used by evaluate().
def getCalc(type, tp, fp, fn):
    if type == "precision":
        try:
            return (tp / (tp+fp))
        except ZeroDivisionError:
            return 0
    elif type == "recall":
        try:
            return tp/(tp+fn)
        except ZeroDivisionError:
            return 0
    else:
        try:
            precision = getCalc("precision", tp, fp, fn)
            recall = getCalc("recall", tp, fp, fn)
            return (2*recall*precision)/(recall+precision)
        except ZeroDivisionError:
            return 0

#Purpose:   Used to evaluate predictions against test-values, then calculates and prints stats: accuracy, precision, recall, and F1
def evaluate(y_pred, y_test, algorithm):
    correct_classes = []            #track which specific classes were predicted correct for precision, recall, and F1
    incorrect_classes = []
    correct = 0
    total = 0

    #Count how many predictions were wrong/right:
    for i in range(len(y_pred)):
        total += 1
        if y_pred[i] == y_test[i]:
            correct += 1
            correct_classes.append(y_test[i])
        else:
            incorrect_classes.append(y_test[i])

    #Evaluate for: Accuracy, Precision, Recall, and F1 score:
    result = [[(correct/total)]]

    #For Precision, Recall, and F1, calculate for each class label:
    for classLabel in range(0, NUM_ATTRIBUTES):
        truePositives = correct_classes.count(classLabel)
        trueNegatives = correct - truePositives
        falsePositives = incorrect_classes.count(classLabel)
        falseNegatives = (total - correct) - falsePositives

       # print(str(truePositives) + "   " +  str(trueNegatives) + "   " + str(falsePositives) + "   " + str(falseNegatives))
        precision = getCalc("precision", truePositives, falsePositives, falseNegatives)
        recall = getCalc("recall", truePositives, falsePositives, falseNegatives)
        f1 = getCalc("F1", truePositives, falsePositives, falseNegatives)
        # print("For classification: " + str(classLabel) + " on dataset: " + str(ds) + ", using " + algorithm + ":")
        # print("\tAccuracy: %.4f" % (correct/total))
        # print("\tPrecision: %.4f" %precision)
        # print("\tRecall: %.4f" %recall)
        # print("\tF1 score: %.4f" %f1)

        result.append([precision, recall, f1])
    return result


#Purpose: Prints out the table of results given a 2d result matrix
def print_result(result):
    print("| Class | Accuracy | Precision | Recall |  F1  |")
    for row in range(len(result)):
        if row == 0:
            print("           %.4f" % result[row][0], end='')     #this is the accuracy and only has one elem!
        else:
            print("    " + str(row - 1) + "       ", end='')
            currList = result[row]
            for col in range(len(currList)):
                if col == 0:                                       #spacing to account for empty accuracy value!
                    print("          %.4f" % result[row][col], end='')
                else:
                    print("    %.4f" % result[row][col], end ='')
        print()



#For each dataset:
for ds in DATASETS:
    dbTraining = []
    X = []
    Y = []

    #1.)-------------------Loading Data-----------------------------------------------
    #Read file, and store each instance in dbTraining
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                rowSplit = row[0].split(";")
                dbTraining.append(rowSplit)

    #Separate attributes (X[]) from classifications (Y[]) for each instance
    for row in range(len(dbTraining)):
        data = []
        for i in range(NUM_ATTRIBUTES):
            data.append(float(dbTraining[row][i]))
        X.append(data)
        Y.append(float(dbTraining[row][NUM_ATTRIBUTES]))



    #2.) ---------Pre-processing--------------------------------
    #2a.) Find # of instances vs. class label:
    numOfObservations = []
    possibleClasses = []
    for i in range(0, 10):              #For each possible class label (0-10), count how many instances in Y[] have it
        possibleClasses.append(i)
        numOfObservations.append(Y.count(i))

    #Make plot
    plt.figure(figsize=[10, 6])
    plt.bar(possibleClasses, numOfObservations, color='red')
    plt.xlabel('Quality')
    plt.ylabel('Number of Observations')
    plt.title(ds)
    plt.show()

    #We have an imbalance of data...


    #2b.) Calculate correlations b/w attributes to see if we can do feauture selection!:
    # Re-arrange attributes in X[] to pass into np.corrcoef():
    transposedAttr = np.array(X).T.tolist()
    correlationList = np.corrcoef(transposedAttr)               #Creates the correlations

    #Use heatmap to plot the correlations:
    plt.figure(figsize=[19,10], facecolor='blue')
    sb.heatmap(correlationList, annot=True, xticklabels=ATTRIBUTE_NAMES, yticklabels=ATTRIBUTE_NAMES)

    #if correlations of features are too high (MAX_CORRELATION), drop them:
    rowsToDelete = []
    for col in range(len(correlationList)):
        for row in range(col):
            if abs(correlationList[col][row]) > MAX_CORRELATION and abs(correlationList[col][row]) < 1:
                rowsToDelete.append(row)

    #Delete the corresponding row in the transposed[][] (b/c every row represents an attribute):
    uniquerowsToDelete = np.unique(rowsToDelete)[::-1]
    for i in uniquerowsToDelete:
        del transposedAttr[i]

    #Update X w/ deleted rows:
    newX = np.array(transposedAttr).T.tolist()




    #3.)---------------------------------Machine Learning: Apply algorithms to train data + evaluate-------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(newX, Y, test_size=0.2, random_state=0)
    #3a.) Perform Naive Bayes:
    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)

    print("Naive Bayes for " + ds)
    print_result(evaluate(y_pred_gnb, y_test, "Naive Bayes"))

    #3b.) Perform KNN:
    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    y_pred_knn = knn.fit(X_train, y_train).predict(X_test)

    print("\nKNN for " + ds)
    print_result(evaluate(y_pred_knn, y_test, "KNN"))




