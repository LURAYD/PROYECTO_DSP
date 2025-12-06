import threading
import numpy as np
import cv2
from queue import Queue
import time
from DSP_1 import datosin, grayscale, gauss, sobel, convertir_a_BGR, modo_control,gamma_correct
from filtros import ker_gaus, sobel_x, sobel_y
from ET_YOLO import deteccion_yolo
from visualizacion import visualizar_debug, demo_multiples_filtros, graficar_histograma_color_gris_gamma_desde_archivos
# Colas
q1 = Queue(maxsize=10)
q2 = Queue(maxsize=10)
q3 = Queue(maxsize=10)
q4 = Queue(maxsize=10)
q_yolo = Queue(maxsize=10)
q_debug = Queue(maxsize=2)
# Función para cambiar modo desde consola
def cambiar_modo():
    while True:
        nuevo_modo = input("Ingresa modo [Color / gris / gauss / sobel/ gamma / gamma_sobel / ver_kernel / salir: ").strip().lower()
        if nuevo_modo in ["Color", "gris", "gauss", "sobel", "gamma","gamma_sobel", "ver_kernel", "salir"]:
            modo_control(nuevo_modo)
            print(f"✅ Modo cambiado a: {nuevo_modo}")
        else:
            print("❌ Modo inválido. Usa: Color, gris, gauss, sobel")

# Hilos
hilo0 = threading.Thread(target=visualizar_debug, args=(q_debug, ker_gaus), daemon=True)

hilo1 = threading.Thread(target=datosin, args=(q1,))
hilo2 = threading.Thread(target=grayscale, args=(q1, q2,))
hilo3 = threading.Thread(target=gauss, args=(q2, q3,q_debug,))
hilo4 = threading.Thread(target=sobel, args=(q3, q4,))
hilo5 = threading.Thread(target=convertir_a_BGR, args=(q4, q_yolo,))
hilo6 = threading.Thread(target=deteccion_yolo, args=(q_yolo,))
hilo7 = threading.Thread(target=modo_control)

# Lanzamiento
hilo0.start()

hilo1.start()
hilo2.start()
hilo3.start()
hilo4.start()
hilo5.start()
hilo6.start()
hilo7.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrumpido por usuario.")


