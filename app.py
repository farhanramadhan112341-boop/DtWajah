import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.utils import img_to_array

# === Inisialisasi ===
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("Emotion_Detection.h5")  # gunakan model baru Anda

# Label emosi
class_labels = ('Marah','Biasa','Takut','Bahagia','Netral','Sedih','Terkejut')

# === Fungsi deteksi emosi ===
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (224,224), interpolation=cv2.INTER_AREA)

        if np.sum([roi_color]) != 0:
            img_pixels = img_to_array(roi_color)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels / 255.0

            prediction = model.predict(img_pixels, verbose=0)[0]
            label = class_labels[prediction.argmax()]

            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

# === Streamlit UI ===
st.title("🎥 Deteksi Ekspresi Wajah Real-Time")
st.write("Ekspresi yang dikenali: **Marah, Biasa, Takut, Bahagia, Netral, Sedih, Terkejut**")

run = st.checkbox("Aktifkan Kamera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("⚠️ Kamera tidak terdeteksi.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = detect_emotion(frame)
    FRAME_WINDOW.image(frame)

camera.release()


