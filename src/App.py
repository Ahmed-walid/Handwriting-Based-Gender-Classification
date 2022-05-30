from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from Utilities import *
from joblib import dump, load
from Features_Extraction import * 


#Variables
DATASET_PATH = "../../dataset"
LABELS_PATH = "../../labels.csv"
TRAIN = True

#Load and Split data
dataset, labels = loadDataSet(DATASET_PATH,LABELS_PATH)
trainData,validationData,testData,trainLabels,validationLabels,testLabels = splitData(dataset,labels)


#clf2 = KNeighborsClassifier(n_neighbors=10)
#clf3 = svm.LinearSVC() 


hinge = Hinge(bordersize=3,sharpness_factor=10)

hinge_train = []
cold_train  = []
hog_train   = []
glcm_train  = []

hinge_test  = []
cold_test   = []
hog_test    = []
glcm_test   = []



for img in trainData:
    #cold_train.append(cold.get_cold_features(img))
    #hog_train.append(HOG(img))
    hinge_train.append(hinge.get_hinge_features(img))
    glcm_train.append(get_glcm_features(img))


for img in testData:
    #cold_test.append(cold.get_cold_features(img))
    # hog_test.append(HOG(img))
    hinge_test.append(hinge.get_hinge_features(img))
    glcm_test.append(get_glcm_features(img))



# np.hstack((hinge_train,glcm_train))


classifier = None
if TRAIN:
    classifier=RandomForestClassifier(n_estimators=100)
    print("Training...")
    classifier.fit(np.hstack((hinge_train,glcm_train)),trainLabels)
    print("Finished.")
    dump(classifier, 'model.joblib')
else:
    classifier = load('model.joblib')


y_pred=classifier.predict(np.hstack((hinge_test,glcm_test)))

# for i in range(0,len(y_pred)):
#     print(str(testLabels[i]) + " is classified as " + str(y_pred[i]))

print("Accuracy:",metrics.accuracy_score(testLabels, y_pred))
