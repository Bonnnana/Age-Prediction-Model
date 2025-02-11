import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np

model_yolo = YOLO('yolov8n-face.pt')
model_age = tf.keras.models.load_model('vgg16_age_recognition_final_256_128_4_20.h5')

LIGHT_BLUE = (255, 191, 0)


def is_same_face(box1, box2, threshold=30):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold


def detect_faces_and_predict_age_from_image(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Грешка: Сликата на патеката {image_path} не може да се вчита.")
        return

    results = model_yolo(frame)

    current_faces = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), LIGHT_BLUE, 2)
                cv2.putText(frame, f'Age: {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIGHT_BLUE, 2)

    cv2.imshow('YOLO Face Detection and Age Prediction', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(output_path, frame)
    print(f"Сликата е зачувана како {output_path}")


if __name__ == '__main__':
    image_files = ['results_on_images/img_1.png', 'results_on_images/img_2.png', 'results_on_images/img_3.png',
                   'results_on_images/img_4.png']

    for image_file in image_files:
        image_path = image_file
        output_path = image_file.replace('.png', '_result.png')
        age_predictions = {}
        detect_faces_and_predict_age_from_image(image_path, output_path)
