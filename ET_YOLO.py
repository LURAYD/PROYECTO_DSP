import numpy as np
import cv2
import time
from ultralytics import YOLO
from seguimiento_multiple import GestorRastreo

def deteccion_yolo(q_entrada):
    model = YOLO("yolov8n.pt")
    gestor = GestorRastreo(distancia_maxima=80, max_frames_perdidos=15)

    conteo_total = 0
    ids_ya_contados = set()
    linea_cruce_x = 320
    offset = 15
    tiempo_inicio = time.time()
    frames_totales = 0

    while True:
        v_entrada = q_entrada.get()
        if v_entrada is None:
            break

        resultados = model(v_entrada, verbose=False)
        detecciones_centroides = []

        for r in resultados:
            for caja in r.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                conf = float(caja.conf[0])
                cls = int(caja.cls[0])

                if conf > 0.4 and cls == 0:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detecciones_centroides.append((cx, cy))
                    cv2.rectangle(v_entrada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(v_entrada, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(v_entrada, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        trackers_activos = gestor.actualizar(detecciones_centroides)
        cv2.line(v_entrada, (linea_cruce_x, 0), (linea_cruce_x, 480), (0, 255, 255), 2)

        for tracker in trackers_activos:
            cx, cy = tracker.prediccion
            id_obj = tracker.id

            # Trazo de trayectoria (línea roja)
            path = list(tracker.trayectoria)
            for k in range(1, len(path)):
                thickness = int(np.sqrt(20 / float(k + 1)) * 2)
                cv2.line(v_entrada, path[k - 1], path[k], (0, 0, 255), thickness)

            # Color según si está perdido o no
            if tracker.missing_count > 0:
                color = (0, 128, 255)  # Naranja si solo en predicción
            else:
                color = (0, 0, 255)    # Rojo si fue corregido

            # Punto actual (predicción)
            cv2.circle(v_entrada, (cx, cy), 5, color, -1)
            cv2.putText(v_entrada, f"ID:{id_obj}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Puntos de predicción futura
            puntos_futuros = tracker.predecir_futuro(pasos=10)
            for i in range(len(puntos_futuros)):
                cv2.circle(v_entrada, puntos_futuros[i], 3, (255, 0, 0), -1)
                if i > 0:
                    cv2.line(v_entrada, puntos_futuros[i - 1], puntos_futuros[i], (255, 0, 0), 1)

            # Conteo cruzando línea
            if (linea_cruce_x - offset) < cx < (linea_cruce_x + offset):
                if id_obj not in ids_ya_contados:
                    conteo_total += 1
                    ids_ya_contados.add(id_obj)
                    cv2.line(v_entrada, (linea_cruce_x, 0), (linea_cruce_x, 480), (0, 255, 0), 4)

        # Métricas
        frames_totales += 1
        fps = frames_totales / (time.time() - tiempo_inicio)
        cv2.putText(v_entrada, f"Count: {conteo_total}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(v_entrada, f"FPS: {fps:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("YOLO + Kalman Tracking con Oclusión", v_entrada)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total contado: {conteo_total}")
    cv2.destroyAllWindows()
