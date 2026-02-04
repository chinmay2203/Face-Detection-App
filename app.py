import cv2
import numpy as np
import tensorflow as tf
from collections import deque

try:
    model = tf.keras.models.load_model('face_model.h5')
    print("Model Loaded! Strict Distance Optimization Active.")
except Exception as e:
    print(f"Error: {e}")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
history = deque(maxlen=20) 

def check_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))

    if len(faces) == 0:
        history.clear()

    for (x, y, w, h) in faces:
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        face_roi = frame[max(0, y-pad_h):y+h+pad_h, max(0, x-pad_w):x+w+pad_w]
        
        if face_roi.size == 0: continue


        img_prep = cv2.resize(face_roi, (96, 96))
        img_prep = img_prep.astype("float32") / 255.0
        img_prep = np.expand_dims(img_prep, axis=0)
        prediction = model.predict(img_prep, verbose=0)
        ai_score = prediction[0][0]

        t_score = check_texture(face_roi)
        dynamic_t_thresh = 6 + (w * 0.22)

        if w < 120:
            ai_threshold = 0.20  
            t_threshold = dynamic_t_thresh * 1.3 
        elif w < 180:
            ai_threshold = 0.30
            t_threshold = dynamic_t_thresh
        else:
            ai_threshold = 0.35 
            t_threshold = dynamic_t_thresh


        is_real = 1 if (ai_score < ai_threshold and t_score > t_threshold) else 0
        history.append(is_real)

        smooth_score = sum(history) / len(history)


        if smooth_score > 0.7: 
            label, color = "REAL PERSON", (0, 255, 0)
        else:
            label, color = "SPOOF / SCREEN", (0, 0, 255)


        text = f"{label} ({int(smooth_score*100)}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

        diag = f"W:{w} AI:{ai_score:.2f} T:{int(t_score)}/{int(t_threshold)}"
        cv2.putText(frame, diag, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imshow('Anti-Spoofing Pro v5 - Strict', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()