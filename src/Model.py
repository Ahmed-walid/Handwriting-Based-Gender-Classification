from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_hastie_10_2
from Utilities import *
from Features_Extraction import * 

"""

THIS CODE IS FOR TESTING - NOT A FINAL ONE

"""

DATASET_PATH = "../../dataset"
LABELS_PATH = "../../labels.csv"

dataset, labels = loadDataSet(DATASET_PATH,LABELS_PATH)
trainData,validationData,testData,trainLabels,validationLabels,testLabels = splitData(dataset,labels)

clf=RandomForestClassifier(n_estimators=100)
# clf = KNeighborsClassifier(n_neighbors=7)

cold = Cold(bordersize=3,sharpness_factor=10)
hinge = Hinge(bordersize=3,sharpness_factor=10)

hinge_train = []
cold_train  = []
hog_train = []
glcm_train = []

hinge_test  = []
cold_test   = []
hog_test = []
glcm_test = []

i=0
for img in trainData:
    #cold_train.append(cold.get_cold_features(img))
    #hinge_train.append(hinge.get_hinge_features(img))
    #hog_train.append(HOG(img))
    l = get_glcm_features(img)
    glcm_train.append(l)
    print(i)
    i+=1

for img in testData:
    #cold_test.append(cold.get_cold_features(img))
    # hinge_test.append(hinge.get_hinge_features(img))
    # hog_test.append(HOG(img))
    glcm_test.append(get_glcm_features(img))

#np.hstack((hinge_train,hog_train))

print(np.shape(glcm_train))

print("Training...")
clf.fit(glcm_train,trainLabels)
print("Finished.")

y_pred=clf.predict(glcm_test)
print("Accuracy:",metrics.accuracy_score(testLabels, y_pred))
