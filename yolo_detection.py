import cv2
from ultralytics import YOLO

# Load the YOLOv8n model (trained on face detection)
model = YOLO('yolov8n-face.pt')  # Ensure this file exists in your environment

# Function to process video frames and detect faces
def detect_faces_in_video():
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection
        results = model(frame)  # Get results from YOLO

        # Loop over detections and draw bounding boxes
        for result in results:
            # result.boxes contains the bounding boxes and other relevant data
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID

                if cls == 0:  # Assuming class '0' is for face
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 150), 2)
                    # Display the confidence score
                    cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show the frame with detections
        cv2.imshow('YOLO Face Detection', frame)

        # Quit video on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run face detection
if __name__ == '__main__':
    detect_faces_in_video()
