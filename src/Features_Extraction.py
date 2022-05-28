
import cv2
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import Preprocessing

N_RHO_BINS = 7
N_ANGLE_BINS = 12
N_BINS = N_RHO_BINS * N_ANGLE_BINS
BIN_SIZE = 360 // N_ANGLE_BINS
R_INNER = 5.0
R_OUTER = 35.0
K_S = np.arange(3, 8)

#####

N_ANGLE_BINS = 40
BIN_SIZE = 360 // N_ANGLE_BINS
LEG_LENGTH = 25


class Cold():
    def __init__(self, sharpness_factor,bordersize):
        self.sharpness_factor = sharpness_factor
        self.bordersize = bordersize
        
        
    def get_cold_features(self, img, approx_poly_factor = 0.01):
        bw_image,_ = Preprocessing.preprocess_image(img, self.sharpness_factor, self.bordersize)
        contours = Preprocessing.get_contour_pixels(bw_image)
        
        rho_bins_edges = np.log10(np.linspace(R_INNER, R_OUTER, N_RHO_BINS))
        feature_vectors = [] #np.zeros((len(K_S), N_BINS))
        
        # print([len(cnt) for cnt in contours])
        for j, k in enumerate(K_S):
            hist = np.zeros((N_RHO_BINS, N_ANGLE_BINS))
            for cnt in contours:
                epsilon = approx_poly_factor * cv2.arcLength(cnt,True)
                cnt = cv2.approxPolyDP(cnt,epsilon,True)
                n_pixels = len(cnt)
                
                point_1s = np.array([point[0] for point in cnt])
                x1s, y1s = point_1s[:, 0], point_1s[:, 1]
                point_2s = np.array([cnt[(i + k) % n_pixels][0] for i in range(n_pixels)])
                x2s, y2s = point_2s[:, 0], point_2s[:, 1]
                
                thetas = np.degrees(np.arctan2(y2s - y1s, x2s - x1s) + np.pi)
                rhos = np.sqrt((y2s - y1s) ** 2 + (x2s - x1s) ** 2)
                rhos_log_space = np.log10(rhos+0.001)
                
                quantized_rhos = np.zeros(rhos.shape, dtype=int)
                for i in range(N_RHO_BINS):
                    quantized_rhos += (rhos_log_space < rho_bins_edges[i])
                    
                for i, r_bin in enumerate(quantized_rhos):
                    theta_bin = int(thetas[i] // BIN_SIZE) % N_ANGLE_BINS
                    hist[r_bin - 1, theta_bin] += 1
                
            normalised_hist = hist / hist.sum()
            feature_vectors.append(normalised_hist.flatten())
            
        return np.asarray(feature_vectors).flatten()



class Hinge():
    def __init__(self, sharpness_factor,bordersize):
        self.sharpness_factor = sharpness_factor
        self.bordersize = bordersize
        
    def get_hinge_features(self, img_file):
    
        bw_image, _ = Preprocessing.preprocess_image(img_file, self.sharpness_factor, self.bordersize)
        contours = Preprocessing.get_contour_pixels(bw_image)
        
        hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))
            
        # print([len(cnt) for cnt in contours])
        for cnt in contours:
            n_pixels = len(cnt)
            if n_pixels <= LEG_LENGTH:
                continue
            
            points = np.array([point[0] for point in cnt])
            xs, ys = points[:, 0], points[:, 1]
            point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]
            
            phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
            phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
            
            indices = np.where(phi_2s > phi_1s)[0]
            
            for i in indices:
                phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
                phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
                hist[phi1, phi2] += 1
                
        normalised_hist = hist / np.sum(hist)
        feature_vector = normalised_hist[np.triu_indices_from(normalised_hist, k = 1)]
        
        return feature_vector



def HOG(img):

    gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    #resizing image 
    resized_img = np.resize(gray, (128,64)) 
   
    #creating hog features 
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)

    # # Rescale histogram for better display 
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 
    
    return fd