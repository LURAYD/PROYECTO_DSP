import cv2
import numpy as np
import matplotlib.pyplot as plt
from filtros import ker_gaus, sobel_x, sobel_y

analisis_realizado = False  # 🟢 Solo se hará una vez

def demo_multiples_filtros(imagen_bgr):
    if len(imagen_bgr.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen_bgr

    region = extraer_region_centrada(imagen_gray, tam=64)
    gamma_tabla = generar_gamma_tabla(0.5)
    region_gamma = cv2.LUT(region, gamma_tabla)

    # Aplicar filtros
    salida_gauss = cv2.filter2D(region, -1, ker_gaus)
    salida_sobel_x = cv2.filter2D(region, -1, sobel_x)
    salida_sobel_y = cv2.filter2D(region, -1, sobel_y)

    # Mostrar todo junto
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].imshow(region, cmap='gray')
    axs[0, 0].set_title("Entrada (64x64)")

    axs[0, 1].imshow(salida_gauss, cmap='gray')
    axs[0, 1].set_title("Filtro Gauss")

    axs[0, 2].imshow(region_gamma, cmap='gray')
    axs[0, 2].set_title("Gamma 0.5")

    axs[1, 0].imshow(salida_sobel_x, cmap='gray')
    axs[1, 0].set_title("Sobel X")

    axs[1, 1].imshow(salida_sobel_y, cmap='gray')
    axs[1, 1].set_title("Sobel Y")

    axs[1, 2].axis('off')  # Espacio vacío (o podrías poner otro filtro si quieres)

    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("🔍 Comparación de Filtros sobre una Región 64x64", fontsize=14)
    plt.show()

def extraer_region_centrada(imagen_gray, tam=256):
    h, w = imagen_gray.shape
    x_ini = w // 2 - tam // 2
    y_ini = h // 2 - tam // 2
    return imagen_gray[y_ini:y_ini+tam, x_ini:x_ini+tam]

def generar_gamma_tabla(gamma):
    inv = 1.0 / gamma
    tabla = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return tabla

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
    
    producto = region * kernel
    suma = np.sum(producto)

    guardar_resultado_txt(region, kernel, producto, suma)  # 👈 Exportar a .txt
    graficar_heatmap(region, producto, kernel)             # 👈 Mostrar como imagen

    imagen_vis = cv2.cvtColor(imagen.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(imagen_vis, (x-1, y-1), (x+1, y+1), (0, 0, 255), 1)
    cv2.imshow("Visualización de región", imagen_vis)
    cv2.waitKey(1)


def visualizar_debug(q_debug, kernel):
    while True:
        img = q_debug.get()
        if img is None:
            break
        recorrer_varias_regiones_en_un_frame(img, kernel, paso=30)

def guardar_resultado_txt(region, kernel, producto, suma):
    with open("resultado_kernel.txt", "w") as f:
        f.write("=== Convolución Visual ===\n")
        f.write("Región 3x3:\n")
        f.write(str(region) + "\n\n")
        f.write("Kernel:\n")
        f.write(str(kernel) + "\n\n")
        f.write("Producto:\n")
        f.write(str(producto) + "\n\n")
        f.write(f"Suma total: {suma}\n")


def graficar_heatmap(region, producto, kernel):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(region, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Región 3x3')
    axs[0].axis('off')

    axs[1].imshow(kernel, cmap='coolwarm', vmin=np.min(kernel), vmax=np.max(kernel))
    axs[1].set_title('Kernel')
    axs[1].axis('off')

    axs[2].imshow(producto, cmap='viridis')
    axs[2].set_title('Producto = Región × Kernel')
    axs[2].axis('off')

    plt.suptitle('Visualización de convolución')
    plt.tight_layout()
    plt.show()
    
def recorrer_varias_regiones_en_un_frame(imagen, kernel, paso=20):
    alto, ancho = imagen.shape[:2]

    # Convertir a gris si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    for y in range(1, alto - 1, paso):
        for x in range(1, ancho - 1, paso):
            punto = (x, y)
            visualizar_kernel_sobre_imagen(imagen, kernel, punto)

            print(f"🧠 Punto analizado: ({x}, {y})")

            # Esperar para que se pueda ver
            key = cv2.waitKey(500)
            if key == ord('q'):  # puedes presionar q para detenerlo
                print("⏹ Visualización pausada por el usuario.")
                return
def recorrer_varias_regiones_en_un_frame(imagen, kernel, paso=20):
    alto, ancho = imagen.shape[:2]

    # Convertir a gris si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    for y in range(1, alto - 1, paso):
        for x in range(1, ancho - 1, paso):
            punto = (x, y)
            visualizar_kernel_sobre_imagen(imagen, kernel, punto)

            print(f" Punto analizado: ({x}, {y})")

            # Esperar para que se pueda ver
            key = cv2.waitKey(500)
            if key == ord('q'):  # puedes presionar q para detenerlo
                print("⏹ Visualización pausada por el usuario.")
                return

def procesar_y_guardar_analisis_completo(frame):
    global analisis_realizado
    if analisis_realizado:
        return

    # Guardar original
    cv2.imwrite("01_color.png", frame)

    # Gris
    R = frame[:, :, 2].astype(float)
    G = frame[:, :, 1].astype(float)
    B = frame[:, :, 0].astype(float)
    gris = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)
    cv2.imwrite("02_gris.png", gris)

    # Gamma
    gamma_img = gamma_correct(gris, gamma=0.5)
    cv2.imwrite("03_gamma.png", gamma_img)

    # Sobel
    dx = cv2.filter2D(gris, cv2.CV_32F, sobel_x)
    dy = cv2.filter2D(gris, cv2.CV_32F, sobel_y)
    magnitud = cv2.magnitude(dx, dy)
    sobel_final = cv2.convertScaleAbs(magnitud)
    cv2.imwrite("04_sobel.png", sobel_final)

    # BGR triplicado
    bgr_triplicado = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("05_bgr_triplicado.png", bgr_triplicado)

    # Gauss
    gauss = cv2.filter2D(gris, -1, ker_gaus)
    cv2.imwrite("06_gauss.png", gauss)

    analisis_realizado = True
    print("✅ Imágenes guardadas.")

def graficar_histograma_color_gris_gamma_desde_archivos():
    color_img = cv2.imread("01_color.png")
    gris_img = cv2.imread("02_gris.png", cv2.IMREAD_GRAYSCALE)
    gamma_img = cv2.imread("03_gamma.png", cv2.IMREAD_GRAYSCALE)
    sobel_img = cv2.imread("04_sobel.png", cv2.IMREAD_GRAYSCALE)

    if color_img is None or gris_img is None or gamma_img is None or sobel_img is None:
        print("❌ No se pudieron cargar una o más imágenes.")
        return

    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    sobel_rgb = cv2.cvtColor(cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    for i, col in enumerate(("Red", "Green", "Blue")):
        hist = cv2.calcHist([color_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, label=f"Color {col}")

    plt.plot(cv2.calcHist([gris_img], [0], None, [256], [0, 256]), color='black', linestyle='--', label='Gris')
    plt.plot(cv2.calcHist([gamma_img], [0], None, [256], [0, 256]), color='gray', linestyle=':', label='Gamma 0.5')

    for i, col in enumerate(("R_Sobel", "G_Sobel", "B_Sobel")):
        hist_sobel = cv2.calcHist([sobel_rgb], [i], None, [256], [0, 256])
        plt.plot(hist_sobel, linestyle='-.', label=col)

    plt.title("Comparación: Color, Gris, Gamma y Sobel")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("07_histograma_completo.png")
    plt.close()

    print("✅ Histograma completo guardado.")

def gamma_correct(imagen, gamma=1.0):
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)
