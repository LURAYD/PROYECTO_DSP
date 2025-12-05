import cv2
import numpy as np

mostrar_kernel = True  # Puedes activarlo o desactivarlo desde modo_control()
def visualizar_kernel_sobre_imagen(imagen, kernel, punto_central):
    from visualizacion import mostrar_kernel
    if not mostrar_kernel:
        return  # 🚫 No hace nada si está apagado

    x, y = punto_central

    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    if x < 1 or y < 1 or x > imagen.shape[1] - 2 or y > imagen.shape[0] - 2:
        print("Punto fuera de rango")
        return

    region = imagen[y-1:y+2, x-1:x+2]
    producto = region * kernel

    print("\n=== Convolución Visual ===")
    print("Región 3x3:")
    print(region)
    print("Kernel:")
    print(kernel)
    print("Producto:")
    print(producto)
    print("Suma total:", np.sum(producto))

    imagen_vis = cv2.cvtColor(imagen.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(imagen_vis, (x-1, y-1), (x+1, y+1), (0, 0, 255), 1)
    cv2.imshow("Visualización de región", imagen_vis)
    cv2.waitKey(1)

def visualizar_debug(q_debug, kernel):
    while True:
        img = q_debug.get()
        if img is None:
            break
        h, w = img.shape[:2]
        punto = (w // 2, h // 2)
        visualizar_kernel_sobre_imagen(img, kernel, punto)
