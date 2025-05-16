import cv2
import numpy as np
import pickle
from skimage import color, feature
from skimage.filters import gabor

# Data preprocessing functions

def hsv_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_saturation_threshold = 60
    saturation_mask = cv2.inRange(hsv_image[:, :, 1], lower_saturation_threshold, 255)
    kernel_size = (5, 5)
    smoothed_mask = cv2.GaussianBlur(saturation_mask, kernel_size, 0)
    _, leaf_mask = cv2.threshold(smoothed_mask, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    segmented_image = image.copy()
    segmented_image[closed_mask == 0] = [0, 0, 0]
    return segmented_image

def extract_lbp_glcm_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    glcm_props = []
    glcm = feature.graycomatrix((image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    glcm_props.append(feature.graycoprops(glcm, prop='dissimilarity'))
    glcm_props.append(feature.graycoprops(glcm, prop='contrast'))
    glcm_props.append(feature.graycoprops(glcm, prop='homogeneity'))
    glcm_props.append(feature.graycoprops(glcm, prop='energy'))
    glcm_props.append(feature.graycoprops(glcm, prop='correlation'))
    glcm_props = np.array(glcm_props).squeeze()

    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequency = [0.1, 0.5, 1.0]

    gabor_features = []
    for t in theta:
        for f in frequency:
            gabor_filter_real, _ = gabor(image, frequency=f, theta=t)
            gabor_features.append(np.mean(gabor_filter_real))
    gabor_features = np.array(gabor_features).squeeze()

    return lbp_hist, glcm_props, gabor_features

def calculate_color_moments(image):
    channels = cv2.split(image)
    color_moments = []
    for channel in channels:
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (variance ** (3/2) + 1e-6)
        color_moments.extend([mean, variance, skewness])
    return color_moments

# Load model and preprocessors once
with open("./Best_Model.pkl", "rb") as f:
    classifier = pickle.load(f)
with open("./pca.pkl", "rb") as f:
    pca = pickle.load(f)
with open("./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_image(image_path):
    image = cv2.imread(image_path)
    hsv = hsv_mask(image)
    image_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
    moments = calculate_color_moments(image_rgb)

    gray_image = color.rgb2gray(image)
    lbp_features, glcm_features, gabor_features = extract_lbp_glcm_features(gray_image)

    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if len(gradient_magnitude.shape) != 2:
        gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)

    slbp_features, sglcm_features, sgabor_features = extract_lbp_glcm_features(gradient_magnitude)

    features_arr = np.concatenate((
        lbp_features, glcm_features, gabor_features,
        slbp_features, sglcm_features, sgabor_features,
        moments
    ))

    X = pca.transform([features_arr])
    X = scaler.transform(X)
    Y = classifier.predict(X)

    return {"class": str(Y[0])}
