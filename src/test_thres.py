import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2


def removeShadow(img):
    
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm




img = cv2.imread('987.jpg',0)
img = removeShadow(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (7,7), 0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)


# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
dilate = cv2.dilate(thresh1, kernel, iterations=4)
cv2.imwrite("dilate1.jpg",dilate)

# Find contours and draw rectangle
cnts,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
#for i,c in enumerate(cnts):

for i,cnt in enumerate(cntsSorted):
    x,y,w,h = cv2.boundingRect(cnt)
    abbas = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    # cv2.imwrite("./abbas/"+str(np.random.randint(0,1000))+".jpg",abbas)
    cv2.imwrite("thres"+str(i)+".jpg",abbas)


