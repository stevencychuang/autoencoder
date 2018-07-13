import numpy as np

def splitData(data, ratio=[0.8, 0.1]):
    numInst = data.shape[0]
    data = np.random.shuffle(data)
    indTrain = round(numInst*ratio[0])
    indTest = round(numInst*sum(ratio))
    dataTrain = data[:indTrain]
    dataTest = data[indTrain:indTest]
    dataValid = data[:indTrain+indTest]
    return dataTrain, dataTest, dataValid 