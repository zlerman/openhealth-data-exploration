import pandas as pd
import requests
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def getData():
    numberOfEntries = "$limit=50000"
    selectClause = "$select=npi,total_claim_count,drug_name,specialty_desc"
    query = "https://data.cms.gov/resource/hffa-2yrd.json?" + selectClause + "&" + numberOfEntries 
    dataFrame = pd.read_json(query)
    
    crosstab = pd.crosstab([dataFrame["npi"], dataFrame["specialty_desc"]], dataFrame["drug_name"])
    drugList = crosstab.keys() # list of all drugs. 1 if doctor has prescribed it 0 if not
    npi = [] # array of npi's
    data = [] # each entry is an array of drugs prescribed
    labels = [] # each entry is either 1 for Psychiatrist, 0 if not.
    
    psychCount = 0
    nonPsychCount = 0
    
    # need more efficient way to handle this
    for index, row in crosstab.iterrows():
        if (index[1] == "Psychiatry"):
            labels.append("1")
            psychCount += 1
        else:
            if (nonPsychCount > 1314):
                continue
            labels.append("0")
            nonPsychCount += 1
        newArray = [] # holds drugs they prescribed
        for drug in drugList:
            newArray.append(row[drug])
        data.append(newArray)
        
        npi.append(index[0])
    return [npi, data, labels, dataFrame]

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

# finds mistakes that occur while using optimal k value
# returns them as dataframe 
def findMistakes(vals, labels, optimalK):
    # find incorrectly labeled providers' npi with K value found
    quarter = int(math.floor(len(vals) * .25))
    mistakes = [] # will hold the npi's of the inccorrect values
    knn = KNeighborsClassifier(n_neighbors = 1)
    trainingData = vals[quarter:]
    trainingLabels = labels[quarter:]
    testData = vals[0: quarter]
    testLabels = labels[0: quarter]
    npi = npi[0: quarter]
    knn.fit(trainingData, trainingLabels)
    for y in range(len(testData)):
            prediction = knn.predict([vals[y]])
            if (prediction != labels[y]):
                mistakes.append(npi[y])
    mistakeDataframe = dataframe[dataframe['npi'].isin(mistakes)]
    return mistakeDateframe

if __name__ == "__main__": 
    data = getData()
    npi = data[0]
    vals = data[1]
    labels = data[2]
    dataframe = data[3]
    #k = findK(vals, labels)    
    
    rocTrue = [] # hold true values
    rocPred = [] # hold predicted probabilities
    
    quarter = int(math.floor(len(vals) * .25))
    knn = KNeighborsClassifier(n_neighbors = 1)
    trainingData = np.array(vals[quarter:])
    trainingLabels = np.array(labels[quarter:])
    testData = np.array(vals[0: quarter])
    testLabels = np.array(labels[0: quarter])
    knn.fit(trainingData, trainingLabels)
    print(testData)
    for y in range(len(testData)):
            rocTrue.append(int(labels[y]))
            rocPred.append(knn.predict_proba(vals[y]).item(0))
            val = knn.predict_proba(vals[y])
            print val[0]
            
    y = np.array(rocTrue)
    scores = np.array(rocPred)
    fpr, tpr, thresholds = roc_curve(y, scores)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("ROC Curve For Psychiatrist Classifier")
    plt.grid(True)
    plt.show()
