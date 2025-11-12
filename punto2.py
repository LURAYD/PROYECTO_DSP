import cv2
import numpy as np
import time
import threading
from queue import Queue
from ultralytics import YOLO

# ————————————————————————————————————————————
# 🎯 Filtros DSP (Gaussiano + Sobel)
# ————————————————————————————————————————————
ker_gaus = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

# ————————————————————————————————————————————
# 📦 Colas para flujos
# ————————————————————————————————————————————
q_original = Queue(maxsize=20)
q_gauss = Queue(maxsize=20)
q_sobel = Queue(maxsize=20)

# ————————————————————————————————————————————
# 🎥 Captura de video
# ————————————————————————————————————————————
def captura():
    cap = cv2.VideoCapture("video1.mp4")
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        q_original.put(frame)
    cap.release()
    q_original.put(None)

# ————————————————————————————————————————————
# 🧱 Filtro Gaussiano (para YOLO caso 1)
# ————————————————————————————————————————————
def aplicar_gauss():
    while True:
        frame = q_original.get()
        if frame is None:
            q_gauss.put(None)
            q_sobel.put(None)
            break

        # Escala de grises
        R, G, B = frame[:,:,2], frame[:,:,1], frame[:,:,0]
        gris = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)

        # Filtro Gaussiano
        suavizada = cv2.filter2D(gris, -1, ker_gaus)
        gauss_bgr = cv2.cvtColor(suavizada, cv2.COLOR_GRAY2BGR)
        q_gauss.put(gauss_bgr)

        # Preprocesamiento Sobel (Gauss + derivadas)
        gx = cv2.filter2D(suavizada, cv2.CV_32F, sobel_x)
        gy = cv2.filter2D(suavizada, cv2.CV_32F, sobel_y)
        magnitud = cv2.magnitude(gx, gy)
        sobel = cv2.convertScaleAbs(magnitud)
        sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        q_sobel.put(sobel_bgr)

# ————————————————————————————————————————————
# 🚀 Detección YOLO
# ————————————————————————————————————————————
def deteccion_yolo(queue_in, nombre_ventana, descripcion):
    model = YOLO("yolov8s.pt")
    contador = 0
    tiempo_inicio = time.time()

    while True:
        frame = queue_in.get()
        if frame is None:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.4 and cls == 0:  # clase 0 = persona
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        contador += 1
        fps = contador / (time.time() - tiempo_inicio)
        cv2.putText(frame, f"{descripcion} - FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow(nombre_ventana, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tiempo_total = time.time() - tiempo_inicio
    print(f"[{descripcion}]")
    print(f"Frames procesados: {contador}")
    print(f"Tiempo total: {tiempo_total:.2f}s")
    print(f"FPS promedio: {contador / tiempo_total:.2f}\n")
    cv2.destroyWindow(nombre_ventana)

# ————————————————————————————————————————————
# 🧵 Lanzamiento de hilos
# ————————————————————————————————————————————
hilo_cap = threading.Thread(target=captura)
hilo_dsp = threading.Thread(target=aplicar_gauss)
hilo_yolo_gauss = threading.Thread(target=deteccion_yolo, args=(q_gauss, "YOLO + Filtro Gaussiano", "GAUSS"))
hilo_yolo_sobel = threading.Thread(target=deteccion_yolo, args=(q_sobel, "YOLO + Sobel", "SOBEL"))

hilo_cap.start()
hilo_dsp.start()
hilo_yolo_gauss.start()
hilo_yolo_sobel.start()

# Esperar a que todos terminen
for hilo in [hilo_cap, hilo_dsp, hilo_yolo_gauss, hilo_yolo_sobel]:
    hilo.join()

print("✅ Comparación completada (Gauss vs Sobel).")
