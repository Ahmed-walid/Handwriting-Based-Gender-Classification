import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import csv


def write_data_txt(arr,path):

    file = open(path,'w') 
    for i,val in enumerate(arr):
        if i==len(arr)-1:
            file.write(str(val))
        else:
            file.write(str(val)+"\n") 
    file.close() 


def read_data_txt(file_name):
    rows = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            rows.append(line)
    data = np.asarray(rows).astype(int)
    return data


def read_data_csv(file_name):
    rows = []
    with open(file_name, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            rows.append(row[0])
    data = np.asarray(rows).astype(int)
    return data


def loadImages(path):
    dataset = []
    for filename in glob.glob(path+"/*.*", recursive=True):
        im=Image.open(filename)
        dataset.append(im)
    return dataset

def loadDataSet(dataSetPath,labelPath):
    dataset = []
    labels  = read_data_csv(labelPath)
    for filename in glob.glob(dataSetPath+"/*.*", recursive=True):
        im=Image.open(filename)
        dataset.append(im)
    return dataset , labels


def splitData(data,labels):
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size = 0.4 , random_state=50)
    testData , validationData , testLabels, validationLabels = train_test_split(testData,testLabels, test_size=0.5,random_state=50)
    return trainData,validationData,testData,trainLabels,validationLabels,testLabels

# def splitData(data,labels):
    #trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size = 0.15 , random_state=50)
    #testData , validationData , testLabels, validationLabels = train_test_split(testData,testLabels, test_size=0.5,random_state=50)
    # print(len(data))
    # print(len(trainData))
    # print(len(testData))
    # return trainData,[],testData,trainLabels,[],testLabels
    # return data,[],[],labels,[],[]   #full training