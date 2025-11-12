import cv2
import numpy as np
import time
from ultralytics import YOLO  # Asegúrate de tenerlo instalado

# ———————————————————
# 🧠 Filtro FIR Gaussiano 3x3
# ———————————————————
ker_gaus = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

# ———————————————————
# 🎯 Cargar modelo YOLOv8
# ———————————————————
model = YOLO("yolov8s.pt")  # Usa tu modelo

# ———————————————————
# 🎥 Cargar video
# ———————————————————
cap = cv2.VideoCapture("video1.mp4")
if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

# ———————————————————
# ⏱️ Medir rendimiento
# ———————————————————
contador = 0
tiempo_inicio = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ———————————————————
    # 🧪 Aplicar filtro Gaussiano
    # ———————————————————
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.filter2D(frame_gray, -1, ker_gaus)

    # Convertir de nuevo a BGR para que YOLO lo acepte (opcional)
    frame_input = cv2.cvtColor(frame_blur, cv2.COLOR_GRAY2BGR)

    # ———————————————————
    # 🚀 Detección con YOLO
    # ———————————————————
    results = model(frame_input)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.4 and cls == 0:  # persona en COCO
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS en pantalla
    contador += 1
    tiempo_actual = time.time()
    fps = contador / (tiempo_actual - tiempo_inicio)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar resultado
    cv2.imshow("YOLO con filtro Gaussiano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ———————————————————
# 📊 Resultados finales
# ———————————————————
tiempo_total = time.time() - tiempo_inicio
print(f"[CON GAUSSIANO]")
print(f"Frames procesados: {contador}")
print(f"Tiempo total: {tiempo_total:.2f}s")
print(f"FPS promedio: {contador / tiempo_total:.2f}")
cap.release()
cv2.destroyAllWindows()
