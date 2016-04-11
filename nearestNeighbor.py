import pandas as pd
import requests
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

def getData():
    # Get all Anesthesiologists
    clauseOne = "$select=npi,total_claim_count,drug_name,specialty_desc&specialty_desc='Anesthesiology'"
    query = "https://data.cms.gov/resource/hffa-2yrd.json?" + clauseOne
    dataFrameOne = pd.read_json(query)
    
    # Get all Psychiatrists
    clauseTwo = "$select=npi,total_claim_count,drug_name,specialty_desc&specialty_desc='Psychiatry'"
    query = "https://data.cms.gov/resource/hffa-2yrd.json?" + clauseTwo
    dataFrameTwo = pd.read_json(query) 
    
    frames = [dataFrameOne, dataFrameTwo]
    dataFrame = pd.concat(frames)
    
    test = pd.crosstab([dataFrame["npi"], dataFrame["specialty_desc"]], dataFrame["drug_name"])
    #distanceMatrix = distance.pdist(test, 'euclidean') unused currently
    keyArr = test.keys()
    data = [] # holds our array vectors
    labels = [] # holds classification of each 
    
    for index, row in test.iterrows():
        newArray = []    
        for x in keyArr:
            newArray.append(row[x])
        data.append(newArray)
        labels.append(index[1])
    return [data, labels]

if __name__ == "__main__": 
    # We have 
    knn = KNeighborsClassifier(n_neighbors = 3) 
    data = getData()
    vals = data[0]
    print(len(vals))
    labels = data[1]
    length = len(vals)
    # now partition the data into 3/4 for testing data and 1/4 for training
    trainingVals = []
    trainingLabels = []
    for x in range(1000):
        trainingVals.append(vals[x])
        trainingLabels.append(labels[x])
    print("")
    knn.fit(trainingVals,trainingLabels)
    
    correct = 0.0
    wrong = 0.0
    for x in range(1001,1620):
        prediction = knn.predict([vals[x]])
        if (prediction == labels[x]):
            correct += 1
        else:
            wrong += 1
        #print("Actual specialty: " + labels[x] + ", Nearest Neightbor prediction: " + str(prediction[0]))
    accuracy = (correct / (correct + wrong)) * 100
    print(accuracy)
    
    
