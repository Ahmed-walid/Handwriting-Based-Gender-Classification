from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt


# #reading the image
# img = imread('F1.jpg')
# imshow(img)
# print(img.shape)


def HOGG(img):
    #resizing image 
    resized_img = resize(img, (128,64)) 
   
    #creating hog features 
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    # # Rescale histogram for better display 
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

    return fd