import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ===== Util: versi & info env =====
def env_info():
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
    except Exception:
        tf_ver = "TensorFlow not found"
    try:
        import keras
        k_ver = keras.__version__
    except Exception:
        k_ver = "Keras not found"
    return tf_ver, k_ver

# ===== Detector wajah =====
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===== Konfigurasi =====
MODEL_PATH = "Emotion_Detection.h5"   # ganti jika perlu
CLASS_LABELS = ('Marah','Biasa','Takut','Bahagia','Netral','Sedih','Terkejut')
INPUT_SIZE = (224, 224)  # samakan dengan input model kamu

# ===== Loader model yang robust + cache =====
@st.cache_resource(show_spinner=False)
def load_emotion_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

    errors = []
    # 1) Coba tf.keras
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        errors.append(("tf.keras.models.load_model", repr(e)))

    # 2) Coba Keras 3 API (safe_mode=False utk custom layers/ops)
    try:
        from keras.saving import load_model as k3_load_model
        return k3_load_model(path, compile=False, safe_mode=False)
    except Exception as e:
        errors.append(("keras.saving.load_model", repr(e)))

    # 3) Info HDF5 (jika file h5 tapi corrupt)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            _ = list(f.keys())  # sekadar memastikan bisa dibaca
    except Exception as e:
        errors.append(("h5py.File", repr(e)))

    # Gagal total -> lempar alasan lengkap
    details = "\n".join([f"- {src}: {msg}" for src, msg in errors])
    raise ValueError("Gagal memuat model.\n" + details)

def predict_emotion(model, face_bgr):
    roi = cv2.resize(face_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    x = roi.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    label = CLASS_LABELS[int(np.argmax(pred))]
    return label

def annotate_frame(model, frame_rgb):
    # OpenCV pakai BGR; model kita pakai BGR di preprocessing
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_bgr = frame_bgr[y:y+h, x:x+w]
        if face_bgr.size == 0:
            continue
        try:
            label = predict_emotion(model, face_bgr)
            cv2.putText(frame_bgr, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        except Exception as e:
            # Kalau ada error shape/ukuran input, tampilkan ringkas
            cv2.putText(frame_bgr, "Err", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            st.caption(f"Prediksi gagal: {type(e).__name__}: {e}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# ===== UI =====
st.title("🎥 Deteksi Ekspresi Wajah")
st.write("Ekspresi: **Marah, Biasa, Takut, Bahagia, Netral, Sedih, Terkejut**")

tf_ver, k_ver = env_info()
with st.expander("ℹ️ Environment info"):
    st.write(f"TensorFlow: `{tf_ver}` — Keras: `{k_ver}`")
    st.write(f"Model path: `{MODEL_PATH}` (exists: {os.path.exists(MODEL_PATH)})")

# Muat model dengan error detail
try:
    model = load_emotion_model(MODEL_PATH)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error("❌ Model gagal dimuat.")
    st.exception(e)
    st.stop()

mode = st.radio("Mode kamera", ["Foto (aman/Cloud)", "Streaming (Lokal)"], horizontal=True)

if mode == "Foto (aman/Cloud)":
    img_file = st.camera_input("Ambil gambar wajah")
    if img_file is not None:
        img = Image.open(img_file)
        frame_rgb = np.array(img)
        out_rgb = annotate_frame(model, frame_rgb)
        st.image(out_rgb, caption="Hasil Deteksi", use_column_width=True)

else:  # Streaming (Lokal)
    st.warning("Mode ini membutuhkan akses webcam lokal. Tidak didukung di Streamlit Cloud.")
    run = st.checkbox("Aktifkan Kamera")
    frame_slot = st.empty()

    if run:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            st.error("⚠️ Kamera tidak terdeteksi.")
        else:
            try:
                while run:
                    ok, frame_bgr = cam.read()
                    if not ok:
                        st.warning("Gagal membaca frame dari kamera.")
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    out_rgb = annotate_frame(model, frame_rgb)
                    frame_slot.image(out_rgb)
                    time.sleep(0.03)
            finally:
                cam.release()
    else:
        st.info("Centang ✅ **Aktifkan Kamera** untuk memulai.")

