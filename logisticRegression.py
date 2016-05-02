
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
    return [typeOneDF, typeTwoDF]

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

def getData():
    start_time = time.time()

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
    
    # need more efficient way to handle this
    for index, row in crosstab.iterrows():
        if (psychCount > 1313): # made it 10x faster, this loop is what takes so long
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

    print("Number of Psychiatrists: " + str(psychCount) + ", Other Specialities: " + str(nonPsychCount))
    print("--- %s seconds ---" % (time.time() - start_time))

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, random_state=0)
    
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    predictions = logreg.predict(X_test)
    
    errors = getErrors(Y_test, predictions, npis, dataFrame)
    typeOneErrors = errors[0]
    typeTwoErrors = errors[1]
   # print typeTwoErrors
    
    AUC = metrics.roc_auc_score(Y_test, predictions)
    print(AUC)
    
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Psychiatrist Prediction Classifier")
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    getData()
