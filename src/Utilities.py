import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob
import pandas as pd
import csv
from skimage import io


def read_data(file_name):
    rows = []
    with open(file_name, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            rows.append(row[0])
    data = np.asarray(rows).astype(int)
    return data


def loadDataSet(dataSetPath,labelPath):
    dataset = []
    labels  = read_data(labelPath)
    for filename in glob.glob(dataSetPath+"/*.*", recursive=True):
        im=Image.open(filename)
        dataset.append(im)
    return dataset , labels


def splitData(data,labels):
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size = 0.4 , random_state=50)
    testData , validationData , testLabels, validationLabels = train_test_split(testData,testLabels, test_size=0.5,random_state=50)
    return trainData,validationData,testData,trainLabels,validationLabels,testLabels

