import base64
from imageio import imread
import numpy as np
import io

def convert_to_array(img_str):
    return imread(io.BytesIO(base64.b64decode(img_str)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).reshape(rgb.shape[0], rgb.shape[1], 1)

def predict_emotion(img_str):
    arr = rgb2gray(np.array(convert_to_array(img_str)))
    arr = arr.reshape(1,48,48,1)

    from keras.models import load_model
    model = load_model("model.h5")
    pred = np.argmax(model.predict(arr))
    labels = ['Angry', 'Disquist', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return labels[pred]
