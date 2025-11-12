import cv2

ruta = "video1.mp4"  # Asegúrate de que el archivo esté en la misma carpeta

cap = cv2.VideoCapture(ruta)

if not cap.isOpened():
    print("❌ No se pudo abrir el video:", ruta)
    exit()

# Obtener propiedades
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duracion = frames / fps if fps > 0 else 0

print("✅ Video cargado correctamente")
print(f"Nombre del archivo: {ruta}")
print(f"Frames totales: {int(frames)}")
print(f"FPS (estimado): {fps:.2f}")
print(f"Duración (estimada): {duracion:.2f} segundos")

cap.release()
