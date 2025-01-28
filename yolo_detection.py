import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')

# Function to process video frames and detect faces
def detect_faces_in_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if cls == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 150), 2)
                    cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('YOLO Face Detection', frame)

        # Quit video on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_faces_in_video()
