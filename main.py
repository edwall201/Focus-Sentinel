import os
# Hide PyTorch C++ warnings (like NNPACK) by setting log level to ERROR
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import cv2
from ultralytics import YOLO
import time 

def run_sentinel():
    # Yolo nano is suitable for raspberry pi
    model = YOLO('yolov8n.pt') 

    # open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Logic variables
    session_start = time.time()
    distraction_start = None
    total_distraction = 0
    is_distracted = False
    print("Press 'q' to end session.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "USER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # using existing cell phone detection
        results = model(frame, classes=[67], verbose=False, conf=0.5, imgsz=320)
        phone_detect = False
        
        # Check detected objects
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                if label == 'cell phone':
                    phone_detect = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "DISTRACTION!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Distraction Timer
        if phone_detect:
            if not is_distracted:
                distraction_start = time.time()
                is_distracted = True
        else:
            if is_distracted:
                duration = time.time() - distraction_start
                total_distraction += duration
                is_distracted = False

        # Visual 
        color = (0, 0, 255) if phone_detect else (0, 255, 0)
        status_text = "STATUS: DISTRACTED" if phone_detect   else "STATUS: FOCUSING"
        
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Total Distraction: {int(total_distraction)}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Focus Sentinel Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    # Account for ongoing distraction time if quits while distracted
    if is_distracted:
        total_distraction += time.time() - distraction_start
        
    print(f"\nSession Summary:")
    print(f"Total focus time: {int(time.time() - session_start - total_distraction)} seconds.")
    print(f"Total time spent on phone: {int(total_distraction)} seconds.")

if __name__ == "__main__":
    run_sentinel()