import pandas as pd
import requests
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import time
import numpy

def getErrors(test, prediction, npis, dataframe):
    prediction = prediction.tolist()
    typeOneErrors = [] # incorrectly predicted that they are Psychiatrist
    typeTwoErrors = [] # incorrectly predicted that they are not Psychiatrist
    for i in range(len(test)):
        if (test[i] != prediction[i]):
            if (test[i] == 0 and prediction == 1):
                typeOneErrors.append(npis[i])
            else:
                typeTwoErrors.append(npis[i])
    typeOneDF = dataframe[dataframe['npi'].isin(typeOneErrors)]
    typeTwoDF = dataframe[dataframe['npi'].isin(typeTwoErrors)]
    return typeOneDF, typeTwoDF

def nullAccuracy(Y_Test):
    oneCount = 0
    zeroCount = 0
    for x in Y_test:
        if x == 1:
            oneCount += 1
        else:
            zeroCount += 1
    oneAverage = (float((oneCount)) / (oneCount + zeroCount)) * 100
    zeroAverage = 1 - oneAverage
    nullAccuracy = max(oneAverage, zeroAverage)
    return nullAccuracy

def findK(data, labels):
    k = 1
    bestAccuracy = -1
    best_Y_test = -1
    best_predictions = -1
    for i in range(1, 11):
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, random_state=0)
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, Y_train)
    
        predictions = knn.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, predictions)
        print("Accuracy with k value of " + str(i) + ": " + str(accuracy))
        if (accuracy > bestAccuracy):
            k = i
            bestAccuracy = accuracy
            best_Y_test = Y_test
            best_predictions = predictions
    return k, bestAccuracy, best_Y_test, best_predictions   

def ROC_AUC(Y_Test, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Psychiatrist Prediction Classifier")
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()
    
    AUC = metrics.roc_auc_score(Y_test, predictions)
    print("AUC value is: " + str(AUC))

def getData():
    #start_time = time.time()

    numberOfEntries = "$limit=50000"
    selectClause = "$select=npi,total_claim_count,drug_name,specialty_desc"
    query = "https://data.cms.gov/resource/hffa-2yrd.json?" + selectClause + "&" + numberOfEntries 
    dataFrame = pd.read_json(query)
    
    crosstab = pd.crosstab([dataFrame["npi"], dataFrame["specialty_desc"]], dataFrame["drug_name"])
    drugList = crosstab.keys() # list of all drugs. 1 if doctor has prescribed it 0 if not
    npis = [] # array of npi's
    data = [] # each entry is an array of drugs prescribed
    labels = [] # each entry is either 1 for Psychiatrist, 0 if not.
    
    psychCount = 0
    nonPsychCount = 0
    
    # Slow loop
    for index, row in crosstab.iterrows():
        if (psychCount > 1313): 
            break
        if (index[1] == "Psychiatry"):
            labels.append(1)
            psychCount += 1
        else:
            if (nonPsychCount > 1313):
                continue
            labels.append(0)
            nonPsychCount += 1
        newArray = [] # holds drugs they prescribed
        for drug in drugList:
            newArray.append(row[drug])
        data.append(newArray)
        npis.append(index[0])

    #print("Number of Psychiatrists: " + str(psychCount) + ", Other Specialities: " + str(nonPsychCount))
    #print("--- %s seconds ---" % (time.time() - start_time))
    return data, labels, npis, dataFrame

if __name__ == "__main__":
    data, labels, npis, dataframe = getData()
    
    # get optimal k and data associated with it
    k, accuracy, Y_test, predictions = findK(data, labels)
    
    # compute type I and type II errors
    typeOneErrors, typeTwoErrors = getErrors(Y_test, predictions, npis, dataframe)

    nullAccuracy = nullAccuracy(Y_test)
    print(nullAccuracy)
    
    # plot ROC curve and compute AUC curve
    ROC_AUC(Y_test, predictions)

# Problem functions
# 4-fold cross-validation 
def crossvalidate(data, labels, k):
    quarter = int(math.floor(len(data) * .25))
    accuracy = 0
    knn = KNeighborsClassifier(n_neighbors = k)
    for x in range(1, 5):
        # partition data into 3/4 training, 1/4 testing
        trainingData = data[0:quarter * (x - 1)] + data[quarter * x:]
        trainingLabels = labels[0:quarter * (x - 1)] + labels[quarter * x:]
        testData = data[quarter * (x - 1): quarter * x]
        testLabels = labels[quarter * (x - 1): quarter * x]
        
        knn.fit(trainingData, trainingLabels)
        correct = 0.0
        wrong = 0.0
        for y in range(len(testData)):
            prediction = knn.predict([vals[y]])
            if (prediction == 1):
                print("testing")
            if (prediction == labels[y]):
                correct += 1
            else:
                wrong += 1
        accuracy += (correct / (correct + wrong)) * 100
    average = accuracy / 4;
    return average

def findK(data, labels):
    maxAccuracy = -1
    optimalK = -1
    for k in range(1, 10):
        accuracy = crossvalidate(data, labels, k)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy 
            optimalK = k
        print("Average accuracy for k value of " + str(k) + " is: " + str(accuracy))
    return optimalK
