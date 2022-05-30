from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from Utilities import *
from joblib import dump, load
from Features_Extraction import * 
import time as time


#Variables
DATASET_PATH = "../../dataset"
LABELS_PATH = "../../labels.csv"
TRAIN = False

#Load and Split data
dataset, labels = loadDataSet(DATASET_PATH,LABELS_PATH)
trainData,validationData,testData,trainLabels,validationLabels,testLabels = splitData(dataset,labels)

#Classifiers:
#clf2 = KNeighborsClassifier(n_neighbors=10)
#clf3 = svm.LinearSVC() 

hinge_train = []
cold_train  = []
hog_train   = []
glcm_train  = []

hinge_test  = []
cold_test   = []
hog_test    = []
glcm_test   = []

hinge = Hinge(bordersize=3,sharpness_factor=10)

classifier = None
if TRAIN:

    for i,img in enumerate(trainData):
        hinge_train.append(hinge.get_hinge_features(img))
        glcm_train.append(get_glcm_features(img))
        print(i)

    classifier=RandomForestClassifier(n_estimators=100)

    time1 = time.time()
    classifier.fit(np.hstack((hinge_train,glcm_train)),trainLabels)
    time2 = time.time()

    print("Finished training in " + str(time2-time1))

    dump(classifier, 'model5.joblib')

else:
    classifier = load('model3.joblib')


time1 = time.time()

for img in testData:
    hinge_test.append(hinge.get_hinge_features(img))
    glcm_test.append(get_glcm_features(img))


y_pred=classifier.predict(np.hstack((hinge_test,glcm_test)))

time2 = time.time()

print("Finished Predicting in " + str(time2-time1))

# for i in range(0,len(y_pred)):
#     print(str(testLabels[i]) + " is classified as " + str(y_pred[i]))

print("Accuracy:",metrics.accuracy_score(testLabels, y_pred))
