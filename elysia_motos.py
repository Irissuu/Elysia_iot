from ultralytics import YOLO
import cv2

model = YOLO("weights/motos_best.pt")  
cap = cv2.VideoCapture("motos2.mp4")     

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = model.predict(frame, conf=0.25, verbose=False)[0]

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        name = model.names[cls]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detecção (motos)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
