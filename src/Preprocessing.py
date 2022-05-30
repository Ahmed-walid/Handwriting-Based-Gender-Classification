from PIL import Image, ImageEnhance
import cv2
import numpy as np


def remove_shadow(img):
    
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

def get_text_area(image):
    
    image = np.array(image)
    image = remove_shadow(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)
    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts , _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    x,y,w,h = cv2.boundingRect(cntsSorted[-1])

    image = Image.fromarray(image[y:y+h, x:x+w].astype('uint8'), 'RGB')
    return image

def preprocess_image( im, sharpness_factor = 10, bordersize = 3):

    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(sharpness_factor)

    (width, height) = (im.width * 2, im.height * 2)
    im_s_1 = im_s_1.resize((width, height))

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
    (thresh, bw_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw_image, orig_image



def get_contour_pixels(bw_image):

    contours, _= cv2.findContours(
        bw_image, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE
        ) 

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    img2 = bw_image.copy()[:,:,np.newaxis]
    img2 = np.concatenate([img2, img2, img2], axis = 2)
    
    return contours



