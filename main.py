import cv2
from ultralytics import YOLO
import time

def run_sentinel():
    # Load the smallest YOLOv8 model (Nano)
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(0)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Logic variables
    distraction_start_time = None
    total_distraction_time = 0
    is_distracted = False

    print("--- Focus Sentinel Active ---")
    print("Press 'q' to end session.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "USER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Run inference (detect only specific classes to save power)
        # Class 67 is 'cell phone' in COCO dataset
        results = model(frame, classes=[67], verbose=False, conf=0.5)
        
        phone_in_view = False
        
        # Check detected objects
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                
                if label == 'cell phone':
                    phone_in_view = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "DISTRACTION!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Distraction Timer Logic
        if phone_in_view:
            if not is_distracted:
                distraction_start_time = time.time()
                is_distracted = True
        else:
            if is_distracted:
                duration = time.time() - distraction_start_time
                total_distraction_time += duration
                is_distracted = False

        # Visual Feedback
        color = (0, 0, 255) if phone_in_view else (0, 255, 0)
        status_text = "STATUS: DISTRACTED" if phone_in_view else "STATUS: FOCUSING"
        
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Total Distraction: {int(total_distraction_time)}s", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Focus Sentinel Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Account for ongoing distraction time if user quits while distracted
    if is_distracted:
        total_distraction_time += time.time() - distraction_start_time
        
    print(f"\nSession Summary:")
    print(f"Total time spent on phone: {int(total_distraction_time)} seconds.")

if __name__ == "__main__":
    run_sentinel()