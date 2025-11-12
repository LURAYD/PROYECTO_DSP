import cv2
import time

# DSP completo: grises → blur → bordes
def dsp_pipeline(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Cargar video desde archivo
video_path = "video1.mp4"  # Cambia por tu archivo de video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = dsp_pipeline(frame)
    cv2.imshow("Video sin pipeline (secuencial)", output)
    frame_count += 1

    if cv2.waitKey(1) == 27:  # ESC para salir (opcional)
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

# Resultados
total_time = end_time - start_time
fps = frame_count / total_time
print(f"\n[SIN PIPELINE]")
print(f"Frames procesados: {frame_count}")
print(f"Tiempo total: {total_time:.2f}s")
print(f"FPS promedio: {fps:.2f}")
