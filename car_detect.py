from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')

video_path = '/home/petr-mikeska/projects/parking-car-detection/kamera_parkoviste.mp4'
cap = cv2.VideoCapture(video_path) #vytvori capture objekt z videa "ctecka videa" 

if not cap.isOpened():
    raise FileNotFoundError(f"Nelze otevřít video: {video_path}")

cv2.namedWindow("Detekce aut", cv2.WINDOW_NORMAL)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-2:
    fps = 30.0

start_playback = time.time()
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "CAR", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Detekce aut", frame)

    frame_idx += 1
    target_time = start_playback + (frame_idx / fps)
    sleep_time = target_time - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()