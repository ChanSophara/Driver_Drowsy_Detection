import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start the webcam
cap = cv2.VideoCapture(0)
closed_counter = 0
CONSECUTIVE_FRAMES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    alert_message = ""
    border_color = (0, 255, 0)  # Default green color for "Non-Drowsy"
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)

        if len(eyes) == 0:
            closed_counter += 1
            if closed_counter >= CONSECUTIVE_FRAMES:
                # Red border for "Drowsiness Alert"
                border_color = (0, 0, 255)
                alert_message = "DROWSINESS ALERT!"
        else:
            closed_counter = 0
            alert_message = "NON-DROWSY"
            border_color = (0, 255, 0)

    # Draw border around the frame
    frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 20)

    # Display the alert message
    if alert_message:
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(alert_message, font, font_scale, thickness)[0]
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        # Shift the text down by increasing the Y-coordinate
        text_y = int(frame.shape[0] - 50)  # Move the message closer to the bottom
        cv2.putText(frame, alert_message, (text_x, text_y), font, font_scale, border_color, thickness)

    # Show "Closed Score" at the bottom left corner
    score_text = f"Closed Score: {closed_counter}"
    cv2.putText(frame, score_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()