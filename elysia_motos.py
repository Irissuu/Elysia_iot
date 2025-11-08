from ultralytics import YOLO
import cv2
from db_oracle import log_event  

WEIGHTS = "weights/motos_best.pt"   
VIDEO   = "motos.mp4"              

CONF_TH = 0.25
IOU_TH  = 0.5

VEL_AREA_THRESH = 40.0   
HISTERESE_S     = 2.0   

RESIZE_TO = (1280, 720)  

last = {}         
state = {}        
below_since = {}   
_last_logged = {}  

def update_state(track_id, cx, cy, area, fps):
    if track_id not in last:
        last[track_id] = (cx, cy, area)
        state[track_id] = "PARADA"
        below_since[track_id] = 0
        return state[track_id], 0.0

    lx, ly, larea = last[track_id]
    last[track_id] = (cx, cy, area)

    if larea > 0:
        area_ratio = abs(area - larea) / larea
    else:
        area_ratio = 0.0

    vel = area_ratio * 1000.0 

    if vel > VEL_AREA_THRESH:
        state[track_id] = "EM_USO"
        below_since[track_id] = 0
    else:
        below_since[track_id] += 1
        if below_since[track_id] >= int(HISTERESE_S * fps):
            state[track_id] = "PARADA"

    return state[track_id], vel


model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

win_name = "Detecção (motos)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    if RESIZE_TO is not None:
        frame = cv2.resize(frame, RESIZE_TO)

    results = model.track(
        source=frame,
        conf=CONF_TH,
        iou=IOU_TH,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for b in results.boxes:
        if b.id is None:
            continue

        tid = int(b.id.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls.item()) if b.cls is not None else -1
        name = model.names.get(cls, "obj") if hasattr(model, "names") else "obj"

        if name.lower() not in ("motorbike", "motorcycle", "moto"):
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1, (x2 - x1) * (y2 - y1))

        est, vel = update_state(tid, cx, cy, area, fps)  

        if _last_logged.get(tid) != est:
            try:
                log_event(tid, est, {"vel_area": float(vel)})
                _last_logged[tid] = est
            except Exception as e:
                print("Falha ao gravar no Oracle:", e)

        color = (0, 255, 0) if est == "EM_USO" else (0, 0, 255)
        label = f"{est}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
