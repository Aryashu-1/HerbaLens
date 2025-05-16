from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pickle
from skimage import feature, color
from skimage.filters import gabor


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once globally
with open("./Best_Model.pkl", "rb") as file:
    classifier = pickle.load(file)
with open("./pca.pkl", "rb") as file:
    pca = pickle.load(file)
with open("./scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

def hsv_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_saturation_threshold = 60
    saturation_mask = cv2.inRange(hsv_image[:, :, 1], lower_saturation_threshold, 255)
    smoothed_mask = cv2.GaussianBlur(saturation_mask, (5, 5), 0)
    _, leaf_mask = cv2.threshold(smoothed_mask, 1, 255, cv2.THRESH_BINARY)
    closed_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    segmented_image = image.copy()
    segmented_image[closed_mask == 0] = [0, 0, 0]
    return segmented_image

def calculate_color_moments(image):
    channels = cv2.split(image)
    color_moments = []
    for channel in channels:
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (variance ** (3/2) + 1e-6)
        color_moments.extend([mean, variance, skewness])
    return color_moments

from skimage import color, feature
from skimage.filters import gabor

def extract_lbp_glcm_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    glcm = feature.graycomatrix((image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    glcm_props = [
        feature.graycoprops(glcm, 'dissimilarity'),
        feature.graycoprops(glcm, 'contrast'),
        feature.graycoprops(glcm, 'homogeneity'),
        feature.graycoprops(glcm, 'energy'),
        feature.graycoprops(glcm, 'correlation')
    ]
    glcm_props = np.squeeze(np.array(glcm_props))

    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.5, 1.0]:
            gabor_real, _ = gabor(image, frequency=freq, theta=theta)
            gabor_features.append(np.mean(gabor_real))

    return lbp_hist, glcm_props, np.array(gabor_features)

def predict_image(filepath):
    try:
        image = cv2.imread(filepath)
        if image is None:
            return {"error": "Invalid image"}

        hsv = hsv_mask(image)
        image_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
        moments = calculate_color_moments(image_rgb)

        gray = color.rgb2gray(image)
        lbp_features, glcm_features, gabor_features = extract_lbp_glcm_features(gray)

        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(gradient_magnitude.shape) != 2:
            gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)
        slbp_features, sglcm_features, sgabor_features = extract_lbp_glcm_features(gradient_magnitude)

        full_features = np.concatenate([
            lbp_features, glcm_features, gabor_features,
            slbp_features, sglcm_features, sgabor_features,
            moments
        ])

        transformed = scaler.transform(pca.transform([full_features]))
        prediction = classifier.predict(transformed)[0]

        return {
            "class": prediction,
           
        }

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
