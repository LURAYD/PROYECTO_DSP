# Este va a ser un bloque de etapa de preprocesamiento mediante pipeline debido a que la idea es que vamos trabjar con muchos datos.
# Trataremos de usar la menor cantidad de funciones que automatizan procesos, esto para pode raplicar tecnicas de preprocesamiento 

import cv2
import numpy as np
import threading
import time
from queue import Queue
import visualizacion
from visualizacion import recorrer_varias_regiones_en_un_frame, graficar_heatmap, visualizar_kernel_sobre_imagen, ker_gaus, mostrar_kernel, demo_multiples_filtros
from visualizacion import gamma_correct, graficar_histograma_color_gris_gamma_desde_archivos, procesar_y_guardar_analisis_completo

#Definimos primero el filtro de kernel 
from filtros import ker_gaus, sobel_x, sobel_y

# Ya definimos las matrices a usar.
#Ahora las colas
q1 = Queue(maxsize=10)  # Cambiar a grises
q2 = Queue(maxsize=10)  # Gauss
q3 = Queue(maxsize=10)  # SObel
q4 = Queue(maxsize=10)  # Salida del sobel, entrada a convertir_a_BGR

q_yolo = Queue(maxsize=10)  # Imagen lista para YOLO (BGR)
q_debug = Queue(maxsize=2)

#Implemento una avariable para contrlar cuando quiero comparar los 2 modelos
modo_actual = "Color"
graficado_histograma = False  # Solo se graficará una vez
# Variables globales para controlar guardado único
guardado_Color = False
guardado_gris = False
guardado_gamma = False
guardado_gauss = False
guardado_sobel = False
guardado_bgr_triplicado = False


def datosin (q1):
    video = cv2.VideoCapture('cruzando_final.mp4')  #segun internet uso 0 para la camara.
    if not video.isOpened():
        print("No se pudo abrir el video.")
        return
    procesado = False
    while True:
        ret, frame = video.read() #hago un desempaquetado de la tupla
        if not ret: # si es false termina el video
            print('Termino el video')
            break
        if not procesado:
            from visualizacion import procesar_y_guardar_analisis_completo
            procesar_y_guardar_analisis_completo(frame)   # ← GUARDA TODAS LAS IMÁGENES

            graficar_histograma_color_gris_gamma_desde_archivos()  # ← AHORA SÍ EXISTEN
            procesado = True

        q1.put(frame)  #basicamente inserto el frama en la cola con la funcion put
    video.release() #cierro el video
    q1.put(None)  # Indicar el fin de los datos(ESTO ME LO RECOMENDO AQUI MISMO XD, INCLUSO ME DICE QUE RECOMEINDA CHATGT ESTA IA ES UN CASO)

def grayscale(q1,q2):
    global modo_actual
    global guardado_Color
    global guardado_gris  # 👈 AGREGA ESTO para que funcione

    global graficado_histograma
    while True:
        frame = q1.get()
        if frame is None:
            q2.put(None)  # Indicar el fin de los datos
            break

        if modo_actual == "Color":
            q2.put(frame)  # Si es modo color, pasar directamente

            



        else:#Por el momento usaremos forma clasica o el metodo de luminosidad
            R = frame[:,:,2].astype(float)
            G = frame[:,:,1].astype(float)
            B = frame[:,:,0].astype(float)
            gris = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)
            #Aplica gamma solo si el modo es "gamma"
            if not guardado_gris:
                cv2.imwrite("imagen_gris.png", gris)
                guardado_gris = True

            if modo_actual in ["gamma", "gamma_sobel"]:
                gamma_img = gamma_correct(gris, gamma=0.5)

                q2.put(gamma_img)

            else:
                q2.put(gris)




def gauss(q2, q3, q_debug):
    global modo_actual
    global guardado_gauss
    contador_frames = 0

    while True:
        entrada = q2.get()
        if modo_actual == "ver_kernel_demo":
            demo_multiples_filtros(entrada)


        if entrada is None:
            q3.put(None)
            q_debug.put(None)  # Cierra visualización
            break

        if modo_actual == "gauss" and q_debug.empty():
            q_debug.put(entrada.copy())  # Enviar copia segura para análisis

        if modo_actual != "gauss":
            q3.put(entrada)
            continue
        if modo_actual == "ver_kernel_demo":
            demo_multiples_filtros(entrada)

        if len(entrada.shape) == 3:
            entrada_gray = cv2.cvtColor(entrada, cv2.COLOR_BGR2GRAY)
        else:
            entrada_gray = entrada

        filtrado = cv2.filter2D(entrada_gray, -1, ker_gaus)
        q3.put(filtrado)


        if contador_frames < 20 and contador_frames % 20 == 0:
            copia_debug = entrada_gray.copy()
            q_debug.put(copia_debug)

        contador_frames += 1

        cv2.imshow("Soloo Gaus con imagen", filtrado)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def sobel(q3, q4):
    global modo_actual
    global guardado_sobel

    contador = 0
    tiempo_inicio = time.time()

    while True:
        entrada = q3.get()
        if entrada is None:
            q4.put(None)
            break

        if modo_actual not in  ["sobel","gamma_sobel" ]:
            q4.put(entrada)
            continue

        # ✅ Convertir a gris si es necesario
        if len(entrada.shape) == 3:
            entrada = cv2.cvtColor(entrada, cv2.COLOR_BGR2GRAY)

        # Aplicar Sobel
        dx = cv2.filter2D(entrada, cv2.CV_32F, sobel_x)
        dy = cv2.filter2D(entrada, cv2.CV_32F, sobel_y)
        magnitud = cv2.magnitude(dx, dy)
        salida = cv2.convertScaleAbs(magnitud)

        q4.put(salida)


        if modo_actual in ["sobel", "gamma_sobel"]:
            cv2.imshow("SALIDA DE SOBEL", salida)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        contador += 1
        tiempo_actual = time.time()
        fps = contador / (tiempo_actual - tiempo_inicio)

    print(f"Frames procesados: {contador}")
    print(f"Tiempo total: {tiempo_actual - tiempo_inicio:.2f}s")
    print(f"FPS promedio: {fps:.2f}")
    
    cv2.destroyWindow("SALIDA DE SOBEL")  # ✅ Cierra solo esa ventana correctamente

    
def convertir_a_BGR(q4, q_yolo):
    global modo_actual
    global guardado_bgr_triplicado

    while True:
        entrada = q4.get()
        if entrada is None:
            q_yolo.put(None)
            break
            # Asegurar que la salida sea de 3 canales siempre
        if len(entrada.shape) == 2:
            salida_bgr = cv2.cvtColor(entrada, cv2.COLOR_GRAY2BGR)
            
        else:
            salida_bgr = entrada.copy()


        q_yolo.put(salida_bgr)


 
def modo_control():
    global modo_actual

    while True:
        tecla = input("Modo (Color / gris / gauss / sobel / gamma / gamma_sobel / ver_kernel / ver_todo / salir): ").strip()

        if tecla in ["Color", "gris", "gauss", "sobel", "gamma", "gamma_sobel"]:
            modo_actual = tecla
            print(f"Cambiado a modo: {modo_actual}")

        elif tecla == "ver_kernel":
            visualizacion.mostrar_kernel = not visualizacion.mostrar_kernel
            print(f"✅ Mostrar matrices: {visualizacion.mostrar_kernel}")

        elif tecla == "ver_todo":
            modo_actual = "ver_kernel_demo"
            print("🎯 Modo de visualización completa activado")

        elif tecla == "salir":
            print("Saliendo del control de modo.")
            break

        else:
            print("Modo inválido.")
