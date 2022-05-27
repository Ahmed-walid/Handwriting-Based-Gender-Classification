import imutils
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(self, img_file, sharpness_factor = 10, bordersize = 3):
    im = Image.open(img_file)
    
    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(sharpness_factor)
    # plt.imshow(im_s_1, cmap='gray')
    
    (width, height) = (im.width * 2, im.height * 2)
    im_s_1 = im_s_1.resize((width, height))
    if self.show_images: plt.imshow(im_s_1, cmap='gray')
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
    if self.show_images: plt.imshow(image, cmap='gray')
    (thresh, bw_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if self.show_images: plt.imshow(bw_image, cmap='gray')
    return bw_image, orig_image



def get_contour_pixels(self, bw_image):
    contours, _= cv2.findContours(
        bw_image, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE
        ) 
    # contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    img2 = bw_image.copy()[:,:,np.newaxis]
    img2 = np.concatenate([img2, img2, img2], axis = 2)
    
    if self.show_images:
        for cnt in contours : 
            cv2.drawContours(img2, [cnt], 0, (255, 0, 0), 1)  
            
        plt.imshow(img2, cmap='gray')
    return contours