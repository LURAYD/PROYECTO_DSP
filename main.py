import threading
import numpy as np
import cv2
from  queue import Queue
import time
from DSP_1 import datosin, grayscale, gauss, sobel, convertir_a_BGR

from ET_YOLO import deteccion_yolo

q1 = Queue(maxsize=200)
q2 = Queue(maxsize=200)
q3 = Queue(maxsize=200)
q4 = Queue(maxsize=200)      # salida de sobel
q_yolo = Queue(maxsize=200)  # imagen en BGR lista para YOLO

# Hilos
hilo1 = threading.Thread(target=datosin, args=(q1,))
hilo2 = threading.Thread(target=grayscale, args=(q1, q2,))
hilo3 = threading.Thread(target=gauss, args=(q2, q3,))
hilo4 = threading.Thread(target=sobel, args=(q3, q4,))
hilo5 = threading.Thread(target=convertir_a_BGR, args=(q4, q_yolo,))
hilo6 = threading.Thread(target=deteccion_yolo, args=(q_yolo,))

# Lanzamiento de hilos
hilo1.start()
hilo2.start()
hilo3.start()
hilo4.start()
hilo5.start()
hilo6.start()


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrumpido por usuario.")
