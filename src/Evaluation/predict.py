"""

EVALUATION SCRIPT


"""

import argparse
import sys
import os
sys.path.append(os.path.abspath('../'))
from Utilities import *
from joblib import dump, load
from Features_Extraction import * 
import time as time
from sklearn import metrics

OUT_DIR  = ""
TEST_DIR = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test')
    parser.add_argument('--out')
    args = parser.parse_args()
    OUT_DIR = args.out
    TEST_DIR = args.test

    testImages = loadImages(TEST_DIR)
    model = load('../model5.joblib')


    hinge = Hinge(bordersize=3,sharpness_factor=10)

    classification = []
    classificationTime = []

    for img in testImages:
        time1    = time.time()
        feature1 = hinge.get_hinge_features(img)
        feature2 = get_glcm_features(img)
        classification.append(model.predict([np.hstack((feature1,feature2))]))
        time2    = time.time()
        classificationTime.append(np.round(time2-time1,2))

   
    assert len(classification)==len(testImages) and len(classificationTime)==len(testImages)

    classificationTime[classificationTime==0] = 0.001
   

    classification = np.asarray(classification).flatten()
    classificationTime =np.asarray(classificationTime).flatten()
    
    write_data_txt(classification,OUT_DIR+"/results.txt")
    write_data_txt(classificationTime,OUT_DIR+"/times.txt")


    












