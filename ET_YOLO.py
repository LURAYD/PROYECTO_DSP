from ultralytics import YOLO
import cv2
import numpy as np
import time
from mod_kalman import Kalman2D  # Asegúrate de tener este módulo

def deteccion_yolo(q_entrada, q_salida_kalman):
    model = YOLO("yolov8n.pt")
    kalman = Kalman2D()
    trayectoria = []  # Aquí guardamos las predicciones
    total_personas = 0
    linea_y = 300
    ya_cruzo = False

    contador = 0
    tiempo_inicio = time.time()

    while True:
        v_entrada = q_entrada.get()
        if v_entrada is None:
            break

        resultados = model(v_entrada)

        for r in resultados:
            for caja in r.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                conf = float(caja.conf[0])
                cls = int(caja.cls[0])

                if conf > 0.4 and cls == 0:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    medida = np.array([[cx], [cy]])
                    kalman.predecir()
                    kalman.corregir(medida)

                    xk = int(kalman.x[0, 0])
                    yk = int(kalman.x[1, 0])

                    # Conteo
                    if yk > linea_y and not ya_cruzo:
                        total_personas += 1
                        ya_cruzo = True
                    elif yk < linea_y:
                        ya_cruzo = False

                    # Guardar trayectoria
                    trayectoria.append((xk, yk))
                    if len(trayectoria) > 50:
                        trayectoria.pop(0)

                    # Dibujar detección
                    cv2.rectangle(v_entrada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(v_entrada, (cx, cy), 4, (255, 0, 0), -1)   # centro original
                    cv2.circle(v_entrada, (xk, yk), 5, (0, 255, 255), -1) # Kalman predicción

                    # Dibujar rastro
                    for punto in trayectoria:
                        cv2.circle(v_entrada, punto, 2, (0, 0, 255), -1)

                    # Dibujar línea de conteo
                    cv2.line(v_entrada, (0, linea_y), (640, linea_y), (255, 255, 255), 2)

                    # Mostrar conteo
                    cv2.putText(v_entrada, f"Conteo: {total_personas}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        contador += 1
        fps = contador / (time.time() - tiempo_inicio)
        cv2.putText(v_entrada, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO + Kalman + Rastro", v_entrada)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[YOLO] FPS promedio: {fps:.2f}")
    cv2.destroyAllWindows()
