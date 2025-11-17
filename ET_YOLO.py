import numpy as np
import cv2
import time
from ultralytics import YOLO

def deteccion_yolo(q_entrada):
    model = YOLO("yolov8s.pt")  # Carga el modelo YOLOv8
    contador = 0
    tiempo_inicio = time.time()

    while True:
        v_entrada = q_entrada.get()
        if v_entrada is None:
            break

        resultados = model(v_entrada)

        for r in resultados:
            for caja in r.boxes:  # 🔁 Corregido: usar .boxes
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                conf = float(caja.conf[0])
                cls = int(caja.cls[0])

                if conf > 0.4 and cls == 0:  # clase 0 = persona (COCO)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.rectangle(v_entrada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(v_entrada, (cx, cy), 5, (255, 0, 0), -1)

        contador += 1
        fps = contador / (time.time() - tiempo_inicio)
        cv2.putText(v_entrada, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detección YOLOv8 + DSP", v_entrada)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[YOLO] Frames procesados: {contador}")
    print(f"[YOLO] Tiempo total: {time.time() - tiempo_inicio:.2f}s")
    print(f"[YOLO] FPS promedio: {fps:.2f}")
    cv2.destroyAllWindows()
