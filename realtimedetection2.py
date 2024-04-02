import cv2
from keras.models import model_from_json
import numpy as np
import os

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON_PATH = os.path.join(BASE_DIR, 'facialemotionmodel.json')
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'facialemotionmodel.h5')
CASCADE_CLASSIFIER_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Load model from JSON
json_file = open(MODEL_JSON_PATH, "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load model weights
model.load_weights(MODEL_WEIGHTS_PATH)

# Load cascade classifier
face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try: 
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass
