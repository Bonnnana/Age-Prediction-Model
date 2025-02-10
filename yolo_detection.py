import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np

model_yolo = YOLO('yolov8n-face.pt')
model_age = tf.keras.models.load_model('vgg16_age_recognition_final_256_128_4_20.h5')

age_predictions = {}


# Comparing box coordinates
def is_same_face(box1, box2, threshold=30):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    # Distance from the centers of the boxes
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold


def detect_faces_and_predict_age():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)

        current_faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # 0 = face, 0.5 = treshold
                if cls == 0 and conf > 0.5:
                    current_faces.append((x1, y1, x2, y2))

                    face_detected = False
                    for face_id, (stored_box, predicted_age) in age_predictions.items():
                        if is_same_face((x1, y1, x2, y2), stored_box):
                            face_detected = True
                            age = predicted_age
                            break

                    if not face_detected:
                        face = frame[y1:y2, x1:x2]

                        face_resized = cv2.resize(face, (224, 224))
                        face_normalized = face_resized / 255.0
                        face_input = np.expand_dims(face_normalized, axis=0)

                        age_prediction = model_age.predict(face_input)
                        age = int(age_prediction[0][0])

                        age_predictions[len(age_predictions)] = ((x1, y1, x2, y2), age)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 150), 2)
                    cv2.putText(frame, f'Age: {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('YOLO Face Detection and Age Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_faces_and_predict_age()
