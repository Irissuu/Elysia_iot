from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/train/weights/best.pt")  

cap = cv2.VideoCapture("pl2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25)[0]

    vagas_ocupadas = 0
    vagas_livres = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        classe = int(box.cls)
        nome = model.names[classe]

        if nome == "occupied":
            color = (0, 0, 255)     
            vagas_ocupadas += 1
        elif nome == "vacant":
            color = (0, 255, 0)     
            vagas_livres += 1
        else:
            color = (255, 255, 255)  

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, nome, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total = vagas_livres + vagas_ocupadas
    contador_texto = f"{vagas_ocupadas}/{total} vagas ocupadas"
    cv2.putText(frame, contador_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Estacionamento Inteligente", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
