import pandas as pd
import requests
import math
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

def getData():
    # Requires $limit clause to go over 1000 entries
    numberOfEntries = "$limit=1000"
    selectClause = "$select=npi,total_claim_count,drug_name,specialty_desc"
    query = "https://data.cms.gov/resource/hffa-2yrd.json?" + selectClause + "&" + numberOfEntries 
    dataFrame = pd.read_json(query)
    
    crosstab = pd.crosstab([dataFrame["npi"], dataFrame["specialty_desc"]], dataFrame["drug_name"])
    drugList = crosstab.keys() # list of all drugs. 1 if doctor has prescribed it 0 if not
   
    data = [] # each entry is an array of drugs prescribed
    labels = [] # each entry is either 1 for Psychiatrist, 0 if not.
    
    for index, row in crosstab.iterrows():
        newArray = [] # holds drugs they prescribed
        for drug in drugList:
            newArray.append(row[drug])
        data.append(newArray)
        if (index[1] == "Psychiatry"):
            labels.append("1")
        else:
            labels.append("0")
    return [data, labels]

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
    for k in range(1, 11):
        accuracy = crossvalidate(data, labels, k)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy 
            optimalK = k
        print("Average accuracy for k value of " + str(k) + " is: " + str(accuracy))
    return optimalK

# name of person, npi, specialty. Put into a dataframe. who looks like psychiatrist but isn't 

if __name__ == "__main__": 
    data = getData()
    vals = data[0]
    labels = data[1]
    k = findK(vals, labels)
