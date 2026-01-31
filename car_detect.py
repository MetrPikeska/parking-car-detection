from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')

video_path = '/home/petr-mikeska/projects/parking-car-detection/kamera_parkoviste.mp4'
cap = cv2.VideoCapture(video_path)          #vytvori capture objekt z videa "ctecka videa"
print(f"Video otevreno: {video_path}") 

if not cap.isOpened():
    raise FileNotFoundError(f"Nelze otevřít video: {video_path}")

cv2.namedWindow("Detekce aut", cv2.WINDOW_NORMAL) #vytvori okno pro zobrazeni videa

fps = cap.get(cv2.CAP_PROP_FPS)             #ziskani fps z videa
if not fps or fps <= 1e-2:                  #osetreni pripadu, kdy fps neni dostupne
    fps = 30.0                              #predpokladana hodnota fps

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = '/home/petr-mikeska/projects/parking-car-detection/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

start_playback = time.time()                #cas, kdy zacalo prehravani videa
frame_idx = 0

while cap.isOpened():                       #hlavni smycka pro cteni a zpracovani snimku cap = ctecka videa
    ret, frame = cap.read()                 #cteni jednoho snimku z videa
    if not ret:
        break

    results = model(frame)                  #results obsahuje vysledky detekce

    for r in results:                       #prochazi vsechny detekce ve snimku r=results
        for box in r.boxes:                 #prochazi vsechny detekcni boxy v ramci jednoho snimku
            cls = int(box.cls[0])           #ziskani tridy detekce
            label = model.names[cls]        #ziskani jmena tridy podle indexu

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2) #vykresleni obdelniku kolem osoby
                cv2.putText(frame, "person", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if video_writer is not None:
        video_writer.write(frame)

    cv2.imshow("Detekce aut", frame)

    frame_idx += 1  
    target_time = start_playback + (frame_idx / fps)
    sleep_time = target_time - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        
        break



cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()