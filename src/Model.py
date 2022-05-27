from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from Utilities import *
from Features_Extraction import * 

DATASET_PATH = "../../dataset"
LABELS_PATH = "../../labels.csv"


dataset, labels = loadDataSet(DATASET_PATH,LABELS_PATH)
trainData,validationData,testData,trainLabels,validationLabels,testLabels = splitData(dataset,labels)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

cold = Cold(bordersize=3,sharpness_factor=10)
hinge = Hinge(bordersize=3,sharpness_factor=10)


hinge_train = []
cold_train = []

hinge_test = []
cold_test = []

for img in trainData:
    #cold_train.append(cold.get_cold_features(img))
    hinge_train.append(hinge.get_hinge_features(img))

    print("cold")

for img in testData:
    #cold_test.append(cold.get_cold_features(img))
    hinge_test.append(hinge.get_hinge_features(img))

#Train the model using the training sets y_pred=clf.predict(X_test)
print("Training...")
clf.fit(hinge_train,trainLabels)
print("Finished.")

y_pred=clf.predict(hinge_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(testLabels, y_pred))
