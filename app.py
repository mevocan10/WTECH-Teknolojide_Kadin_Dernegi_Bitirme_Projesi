from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img



def getClassName(classNo):
    if classNo == 0:
        return 'Hız Limiti 20 km/h'
    elif classNo == 1:
        return 'Hız Limiti 30 km/h'
    elif classNo == 2:
        return 'Hız Limiti 50 km/h'
    elif classNo == 3:
        return 'Hız Limiti 60 km/h'
    elif classNo == 4:
        return 'Hız Limiti 70 km/h'
    elif classNo == 5:
        return 'Hız Limiti 80 km/h'
    elif classNo == 6:
        return 'Hız Limiti 80 km/h yolunun sonu'
    elif classNo == 7:
        return 'Hız Limiti 100 km/h'
    elif classNo == 8:
        return 'Hız Limiti 120 km/h'
    elif classNo == 9:
        return 'Taşıt Giremez'
    elif classNo == 10:
        return '3,5 tonun üzerindeki araçlara geçemez'
    elif classNo == 11:
        return 'Ana Yol-Tali Yol Kavşağı'
    elif classNo == 12:
        return 'Anayol'
    elif classNo == 13:
        return 'Yol ver'
    elif classNo == 14:
        return 'Dur'
    elif classNo == 15:
        return 'Taşıt Trafiğine Kapalı Yol'
    elif classNo == 16:
        return 'Kamyon Giremez'
    elif classNo == 17:
        return 'Giriş Yasak'
    elif classNo == 18:
        return 'Dikkat'
    elif classNo == 19:
        return 'Sola Tehlikeli Viraj'
    elif classNo == 20:
        return 'Sağa Tehlikeli Viraj'
    elif classNo == 21:
        return 'Tehlikeli Devamlı Virajlar'
    elif classNo == 22:
        return 'Kasisli Yol'
    elif classNo == 23:
        return 'Kaygan Yol'
    elif classNo == 24:
        return 'Sağdan Daralan Yol'
    elif classNo == 25:
        return 'Yol Çalışması'
    elif classNo == 26:
        return 'Trafik Lambası'
    elif classNo == 27:
        return 'Yaya Yolu'
    elif classNo == 28:
        return 'Okul Yolu'
    elif classNo == 29:
        return 'Bisiklet Yolu'
    elif classNo == 30:
        return 'Gizli Buzlanma Uyarısı'
    elif classNo == 31:
        return 'Vahşi Hayvanlar Çıkabilir'
    elif classNo == 32:
        return 'Tüm Hız ve Geçiş Sınırlarının sonu'
    elif classNo == 33:
        return 'Sağa Zorunlu Dönüş'
    elif classNo == 34:
        return 'Sola Zorunlu Dönüş'
    elif classNo == 35:
        return 'Mecburi İleri Yön'
    elif classNo == 36:
        return 'İleri ve Sağa Mecburi Yön'
    elif classNo == 37:
        return 'İleri ve Sola Mecburi Yön'
    elif classNo == 38:
        return 'Sağda Kal'
    elif classNo == 39:
        return 'Solda Kal'
    elif classNo == 40:
        return 'Döner Kavşak'
    elif classNo == 41:
        return 'Geçme Yasağı Sonu'
    elif classNo == 42:
        return '3,5 tonun üzerindeki araçlara geçemez yasağının sonu'


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    # probabilityValue =np.amax(predictions)
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)
