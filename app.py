import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region
        face_region = frame[y:y + h, x:x + w]
        
        # Convert to grayscale for eye detection
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(gray_face)
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(face_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Crop the eye region and preprocess it for prediction
            eye_region = face_region[ey:ey + eh, ex:ex + ew]
            img = cv2.resize(eye_region, (48, 48))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            
            # Predict drowsiness or non-drowsiness
            prediction = model.predict(img)

            if prediction < 0.5:
                cv2.putText(frame, 'Drowsy', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'NonDrowsy', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with rectangles around faces and eyes, and drowsiness status
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
