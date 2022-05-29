import imutils
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_text_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts , _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #for i,c in enumerate(cnts):
    c = cnts[-1]
    x,y,w,h = cv2.boundingRect(c)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    return image[y:y+h, x:x+w]

def preprocess_image( im, sharpness_factor = 10, bordersize = 3):

    
    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(sharpness_factor)
    # plt.imshow(im_s_1, cmap='gray')
    
    (width, height) = (im.width * 2, im.height * 2)
    im_s_1 = im_s_1.resize((width, height))
    #if self.show_images: plt.imshow(im_s_1, cmap='gray')
    image = np.array(im_s_1)
    image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255,255,255]
    )
    orig_image = image.copy()

    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(3,3),0)
   # if self.show_images: plt.imshow(image, cmap='gray')
    (thresh, bw_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   # if self.show_images: plt.imshow(bw_image, cmap='gray')
    return bw_image, orig_image



def get_contour_pixels(bw_image):

    contours, _= cv2.findContours(
        bw_image, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE
        ) 
    # contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    img2 = bw_image.copy()[:,:,np.newaxis]
    img2 = np.concatenate([img2, img2, img2], axis = 2)
    
    # if self.show_images:
    #     for cnt in contours : 
    #         cv2.drawContours(img2, [cnt], 0, (255, 0, 0), 1)  
            
    #     plt.imshow(img2, cmap='gray')
    return contours