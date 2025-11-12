# Este va a ser un bloque de etapa de preprocesamiento mediante pipeline debido a que la idea es que vamos trabjar con muchos datos.
# Trataremos de usar la menor cantidad de funciones que automatizan procesos, esto para pode raplicar tecnicas de preprocesamiento 

import cv2
import numpy as np
import threading
import time
from queue import Queue

#Definimos primero el filtro de kernel 

ker_gaus = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# Ya definimos las matrices a usar.
#Ahora las colas
q1 = Queue(maxsize=20)  # Cambiar a grises
q2 = Queue(maxsize=20)  # Gauss
q3 = Queue(maxsize=20)  # SObel
q_yolo = Queue(maxsize=20)  # Imagen lista para YOLO (BGR)


def datosin ():
    video = cv2.VideoCapture('video_4h.mp4')  #segun internet uso 0 para la camara.
    if not video.isOpened():
        print("No se pudo abrir el video.")
        return
    while True:
        ret, frame = video.read() #hago un desempaquetado de la tupla
        if not ret: # si es false termina el video
            print('Termino el video')
            break
        q1.put(frame)  #basicamente inserto el frama en la cola con la funcion put
    video.release() #cierro el video
    q1.put(None)  # Indicar el fin de los datos(ESTO ME LO RECOMENDO AQUI MISMO XD, INCLUSO ME DICE QUE RECOMEINDA CHATGT ESTA IA ES UN CASO)

def grayscale():
    while True:
        frame = q1.get()
        if frame is None:
            q2.put(None)  # Indicar el fin de los datos
            break
        #Por el momento usaremos forma clasica o el metodo de luminosidad
        R = frame[:,:,2].astype(float)
        G = frame[:,:,1].astype(float)
        B = frame[:,:,0].astype(float)
        gris = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)

        q2.put(gris)

def gauss():
    while True:
        gris = q2.get()
        if gris is None:
            q3.put(None)  # Indicar el fin de los datos
            break
        filtrado = cv2.filter2D(gris, -1, ker_gaus)
        #la descripcion de esto de arriba seria <<UP>>dst = cv2.filter2D(src, ddepth, kernel)

        q3.put(filtrado)

def sobel():
    contador = 0
    tiempo_inicio = time.time() 

    while True:
        entrada = q3.get()
        if entrada is None:
            break
        #Ahora usamos las 2 matrices para calcular la gradicente mediante el proceso de convolucion 
        dx = cv2.filter2D(entrada, cv2.CV_32F, sobel_x)
        dy = cv2.filter2D(entrada, cv2.CV_32F, sobel_y)
        #Calculo de la magnitud del gradiente
        magnitud = cv2.magnitude(dx, dy)
        salida = cv2.convertScaleAbs(magnitud)
        contador += 1 
        tiempo_actual = time.time()
        fps = contador / (tiempo_actual - tiempo_inicio)

        cv2.imshow("DSP Pipeline Output", salida)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
def convertir_a_BGR():
    while True:
        entrada = q3.get()
        if entrada is None:
            q_yolo.put(None)  # Señal de fin
            break
        # Ya que 'entrada' es salida del filtro Gauss, hacemos Sobel como antes:
        dx = cv2.filter2D(entrada, cv2.CV_32F, sobel_x)
        dy = cv2.filter2D(entrada, cv2.CV_32F, sobel_y)
        magnitud = cv2.magnitude(dx, dy)
        salida = cv2.convertScaleAbs(magnitud)

        # 🔁 Conversión a BGR (3 canales)
        salida_bgr = cv2.cvtColor(salida, cv2.COLOR_GRAY2BGR)

        # Enviar imagen procesada (lista para YOLO)
        q_yolo.put(salida_bgr)

  
    print(f"Frames procesados: {contador}") 
    print(f"Tiempo total: {tiempo_actual - tiempo_inicio:.2f}s") 
    print(f"FPS promedio: {fps:.2f}") 
    cv2.destroyAllWindows()

# Ahora creamos los hilos
hilo_datosin = threading.Thread(target=datosin)     
hilo_grayscale = threading.Thread(target=grayscale)
hilo_gauss = threading.Thread(target=gauss)
hilo_sobel = threading.Thread(target=sobel)
# Iniciamos los hilos
hilo_datosin.start()
hilo_grayscale.start()
hilo_gauss.start()
hilo_sobel.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrumpido por usuario.")


