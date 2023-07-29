import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import hog

# Set the paths to the test and train folders
dataset_path = "/content/drive/MyDrive/dataset"
train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")

# Create empty lists to store image features and corresponding labels
features = []
labels = []

# Load the images from the train folder, extract features, and assign labels
for category in os.listdir(train_folder):
    category_path = os.path.join(train_folder, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)
        features.append(hog(thresholded_image))
        labels.append(category)

# Convert the features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Perform label encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Perform feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create and train the SVM classifier
svm = SVC(kernel='linear')
svm.fit(features_scaled, labels_encoded)

# Iterate over the images in the test folder, make predictions, and print the result
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)
    test_feature = hog(thresholded_image)
    test_feature_scaled = scaler.transform([test_feature])
    predicted_label = svm.predict(test_feature_scaled)
    predicted_category = label_encoder.inverse_transform(predicted_label)[0]
    print(f"Image: {image_path} - Predicted Category: {predicted_category}")

