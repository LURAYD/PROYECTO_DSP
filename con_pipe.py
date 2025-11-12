import cv2
import time
import threading
import queue

# DSP por etapas
def etapa_gris(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def etapa_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)

def etapa_bordes(blurred):
    return cv2.Canny(blurred, 50, 150)

# Colas entre etapas
q_capture = queue.Queue()
q_gris = queue.Queue()
q_blur = queue.Queue()

procesamiento_activo = True  # bandera de control

# Captura desde archivo de video
def captura_video(video_path):
    global procesamiento_activo, start_time
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        procesamiento_activo = False
        return
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📉 Fin del video.")
            procesamiento_activo = False
            break
        q_capture.put(frame)

    cap.release()

# Etapa 1: convertir a grises
def etapa1():
    while procesamiento_activo or not q_capture.empty():
        if not q_capture.empty():
            frame = q_capture.get()
            gris = etapa_gris(frame)
            q_gris.put(gris)

# Etapa 2: aplicar blur
def etapa2():
    while procesamiento_activo or not q_gris.empty():
        if not q_gris.empty():
            gray = q_gris.get()
            blur = etapa_blur(gray)
            q_blur.put(blur)

# Etapa 3: aplicar bordes y mostrar
def etapa3():
    global frame_count, end_time
    frame_count = 0

    while procesamiento_activo or not q_blur.empty():
        if not q_blur.empty():
            blurred = q_blur.get()
            output = etapa_bordes(blurred)
            cv2.imshow("Video con pipeline (paralelo)", output)
            frame_count += 1
            cv2.waitKey(1)

    end_time = time.time()
    total = end_time - start_time
    fps = frame_count / total
    print(f"\n[CON PIPELINE]")
    print(f"Frames procesados: {frame_count}")
    print(f"Tiempo total: {total:.2f}s")
    print(f"FPS promedio: {fps:.2f}")
    cv2.destroyAllWindows()

# Ruta del video
ruta_video = "video1.mp4"  # Cambia a tu archivo

# Lanzar hilos del pipeline
threading.Thread(target=captura_video, args=(ruta_video,), daemon=True).start()
threading.Thread(target=etapa1, daemon=True).start()
threading.Thread(target=etapa2, daemon=True).start()
threading.Thread(target=etapa3, daemon=True).start()

# Mantener vivo el programa hasta que termine
while procesamiento_activo or not q_blur.empty():
    time.sleep(0.1)
