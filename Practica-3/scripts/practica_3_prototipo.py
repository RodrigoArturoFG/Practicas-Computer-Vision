# ----- PRÁCTICA 3 - "CONTEO DE OBJETOS" -----
# --------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 10-03-2026

import matplotlib.pyplot as plt
from scipy.stats import skew
from math import log2
import numpy as np
import datetime
import cv2
import os

import config
import controlador_imagen as procesadorImagen
from modelo_imagen import metadataImagen

# --------- VARIABLES GLOBALES ---------
ruta_imagen = os.path.join(config.script_dir_parent, 'resources\\input\\estrella_naranja.jpeg')
imagen_metadata = metadataImagen(ruta_imagen)

# Obtine la ruta de la imagen seleccionada
def seleccionar_imagen():
    """
    Permite al usuario seleccionar una imagen de la carpeta resources\\input.
    Actualiza la variable global ruta_imagen.
    """
    # Obtener la lista de archivos en la carpeta de entrada
    global ruta_imagen

    carpeta_input = os.path.join(config.script_dir_parent, 'resources/input')
    archivos = [f for f in os.listdir(carpeta_input) if os.path.isfile(os.path.join(carpeta_input, f))]
    if not archivos:
        print("No se encontraron imágenes en la carpeta resources/input.")
        return
    
    # Mostrar el menú de selección de imagen
    print("\n=== Selección de Imagen ===")
    for i, nombre in enumerate(archivos, start=1):
        print(f"{i}. {nombre}")
    print(f"{len(archivos)+1}. Cancelar")
    while True:
        opcion = input("Selecciona el número de la imagen a usar: ").strip()
        if opcion.isdigit():
            idx = int(opcion) - 1
            if idx == len(archivos):
                print("Selección cancelada.")
                return None
            elif 0 <= idx < len(archivos):
                # Actualizar la ruta de la imagen seleccionada
                ruta_imagen = os.path.join(carpeta_input, archivos[idx])
                print(f"Imagen seleccionada: {ruta_imagen}")
                return ruta_imagen
            else:
                print("Número fuera de rango.")
                return None

        else:
            print("Ingresa un número válido.")
            return None

# Entrada Umbral 
def solicitar_umbral():
    """Pide un número al usuario y valida que sea un entero entre 0 y 255."""
    while True:
        entrada = input("Introduce el valor del umbral (0-255) [Enter para 128]: ").strip()
        
        # Permitir un valor por defecto si solo presiona Enter
        if entrada == "":
            return 128
        
        # Validar si es un número
        if entrada.isdigit():
            valor = int(entrada)
            if 0 <= valor <= 255:
                return valor
            else:
                print("[!] Error: El valor debe estar entre 0 y 255. Umbral por defecto establecido: 128")
                return 128
        else:
            print("[!] Error: Por favor, introduce un número entero válido. Umbral por defecto establecido: 128")
            return 128

# Cargar imagen RGB con OpenCV
def cargar_imagen_rgb_opencv(imagen_metadata):
    """Carga una imagen en formato RGB y guarda sus metadatos en variable."""
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.cargar_imagen_opencv_rgb(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Convertir a grises (Versión Flexible para UI)
# Una imagen gris en OpenCV tiene 2 dimensiones: (alto, ancho)
def convertir_a_grises(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.cargar_imagen_opencv_gris(imagen_metadata)
        
    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Binarizar imagen
def binarizar_imagen(imagen_metadata, umbral=128):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_binaria(imagen_metadata, umbral)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Binarización Automática (Método de Otsu)
def binarizar_imagen_otsu(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_otsu(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Función para convertir la imagen a HSV y mostrar los canales por separado
def convertir_a_hsv(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_hsv(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata


# Función para la imagen a cmy y mostrar los canales por separado
# (CMY simulado, ya que OpenCV no lo soporta directamente)
def convertir_a_cmy(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_cmy(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Función para la imagen a YIQ y mostrar los canales por separado
# (YIQ simulado, ya que OpenCV no lo soporta directamente)
def convertir_a_yiq(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_yiq(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Función para la imagen a HSI y mostrar los canales por separado
# (HSI simulado, ya que OpenCV no lo soporta directamente)
def convertir_a_hsi(imagen_metadata):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.conversion_imagen_opencv_hsi(imagen_metadata)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata

# Mostrar imagen en pantalla (Versión Optimizada por Metadatos del Modelo)
def mostrar_imagen_opencv(imagen_metadata):
    """
    Usa el atributo 'imagen_metadata.modelo' para decidir el Colormap sin analizar la matriz.
    """
    cmap_actual = None         # Inicialización del colormap actual

    # Decidir el mapa de color (cmap) basado en el metadato del objeto
    if procesadorImagen.es_modelo_monocromatico(imagen_metadata.modelo):
        cmap_actual = 'gray'   # Si el modelo es GRIS o BINARIO, usamos 'gray', de lo contrario Matplotlib asume RGB
        if procesadorImagen.es_binaria(imagen_metadata):
                # 2. Visualización para imagen Binaria
                plt.imshow(imagen_metadata.datos, cmap=cmap_actual)
                plt.title(f"Resultado Binarización (umbral = {imagen_metadata.umbral})")
                plt.axis("off")
                plt.show()
                return

    # 3. Visualización para las demas imagenes RGB, Grises, HSV, etc.
    plt.imshow(imagen_metadata.datos, cmap=cmap_actual)
    plt.title(f"Modelo de color: [{imagen_metadata.modelo}]    -   Imagen: [{imagen_metadata.nombre}]")
    plt.axis("off")
    plt.show()

# Función Genérica para Separar y Visualizar Canales (RGB, HSV, CMY, etc.)
def separar_canales(imagen_metadata):
    """
    Separa y muestra los canales de cualquier modelo de color utilizando la configuración obtenida de: 
    procesadorImagen.obtener_config_modelo(tipo_modelo)
    \n--- EJEMPLOS DE LLAMADA ---
    # Para RGB:
    separar_canales(img_rgb, "RGB")

    # Para HSV:
    separar_canales(img_hsv, "HSV")

    # Para CMY:
    separar_canales(img_cmy, "CMY")

    # Para Gris o Binaria:
    separar_canales(img_gray, "GRIS")
    """
    # 1. Obtener la configuración específica para este modelo
    conf = procesadorImagen.obtener_config_modelo(imagen_metadata.modelo)
    nombres = conf["nombres"]
    mapas = conf["cmaps"]

    # 2. Separación de canales con OpenCV
    canales = cv2.split(imagen_metadata.datos)
    num_canales = len(canales)
    
    # 3. Visualización Dinámica
    plt.figure(figsize=(4 * num_canales, 4))
    
    for i in range(num_canales):
        plt.subplot(1, num_canales, i + 1)
        
        # Asignar nombre y mapa de color desde la configuración
        nombre_actual = nombres[i] if nombres and i < len(nombres) else f'Canal {i+1}'
        cmap_actual = mapas[i] if mapas and i < len(mapas) else 'gray'
        
        plt.imshow(canales[i], cmap=cmap_actual)
        plt.title(nombre_actual)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return canales

# Determina automáticamente qué tipo de histograma calcular
def calcular_histograma_automatico(imagen_metadata):
    """
    Determina automáticamente qué tipo de histograma calcular (dependiendo del modelo de color).
    Las propiedades calculadas incluyen energía, entropía, asimetría (skewness), media y varianza.
    \nSi es una imagen en grises o binaria solo se analiza un canal de intensidad (valores entre 0 y 255), lo
    que simplifica el cálculo del histograma y las propiedades estadísticas.
    """
    # 1. Llamamos a la función maestra pasando el objeto de metadatos
    respuesta = procesadorImagen.proceso_histograma_completo(imagen_metadata)

    # 2. Actualizamos nuestra referencia (por si hubo cambios)
    imagen_metadata = respuesta["objeto"]
    
    # 3. Gestionamos el mensaje de error o éxito del paquete
    if respuesta["error"]:
        print(f"\n[!] ERROR EN HISTOGRAMA: {respuesta['mensaje']}")
    else:
        # El mensaje de éxito ya confirmará que se guardaron los datos
        print(f"\n[*] {respuesta['mensaje']}")
    
    # 4. Obtener la configuración visual del modelo actual (RGB, HSV, CMY, etc.)
    config = procesadorImagen.obtener_config_modelo(imagen_metadata.modelo)    
    nombres = config["nombres"]
    colores = config["cmaps"]

    # 5. Mostrar y Graficar después (Usando los colores de la config)
    print(f"\n--- Análisis Estadístico: Modelo [{imagen_metadata.modelo}] ---")
    imprimir_tabla_estadisticas(imagen_metadata.histograma, "Resultados por Canal")
    
    # 6. Mapeamos los nombres de matplotlib si los colores son descriptivos
    mapa_colores_plt = [procesadorImagen.diccionario_colores.get(c.replace('_r', ''), 'black') for c in colores] # Limpiamos '_r' para el plot
    
    # 7. Graficar COMPUESTO o individuales según el número de canales
    #if len(imagen_metadata.histograma) > 1:
    graficar_y_guardar_histograma_compuesto(imagen_metadata.histograma, mapa_colores_plt, imagen_metadata.modelo, imagen_metadata.nombre)
    # graficar_y_guardar_histogramas(imagen_metadata.histograma, mapa_colores_plt, prefijo_archivo="analisis")
    
    # Pausa para que el usuario lea en consola
    print("\n" + "-"*45)
    input("Presiona ENTER para volver al menú principal...")

    return imagen_metadata

def graficar_y_guardar_histogramas(resultados, colores_plot, prefijo_archivo="hist"):
    """Genera las gráficas y las guarda en disco."""
    for i, (canal, props) in enumerate(resultados.items()):
        plt.figure()
        plt.title(f'Histograma del canal {canal}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        plt.plot(props['histograma_raw'], color=colores_plot[i])
        plt.grid(True)
        
        # Guardado automático
        ruta_salida = os.path.join(config.script_dir_parent, "resources", "output")
        os.makedirs(ruta_salida, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(ruta_salida, f"{prefijo_archivo}_{canal.lower()}_{ts}.png"))
        plt.show()
        plt.close() # <--- liberar memoria

def graficar_y_guardar_histograma_compuesto(resultados, colores_plot, modelo, nombre_img):
    """Genera una sola gráfica con todos los canales superpuestos."""
    plt.figure(figsize=(10, 6))
    
    for i, (canal, props) in enumerate(resultados.items()):
        # Usamos alpha=0.7 para que se vea la superposición si las líneas coinciden
        plt.plot(props['histograma_raw'], color=colores_plot[i], label=canal, alpha=0.7)
    
    plt.title(f'Histograma Compuesto\nModelo de color: [{modelo}]   -   Imagen: [{nombre_img}]')
    plt.xlabel('Intensidad de Píxel (0-255)')
    plt.ylabel('Frecuencia (Píxeles)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Guardado
    ruta_salida = os.path.join(config.script_dir_parent, "resources", "output")
    os.makedirs(ruta_salida, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(ruta_salida, f"HIST_COMPUESTO_{modelo}_{ts}.png"))
    
    plt.show()
    plt.close() # <--- liberar memoria


def imprimir_tabla_estadisticas(resultados, titulo):
    """Muestra los números en la consola."""
    print(f"\n--- {titulo} ---")
    for canal, props in resultados.items():
        print(f'\nCanal {canal}:')
        for llave, valor in props.items():
            if llave != 'histograma_raw': # No imprimimos el array del histograma
                print(f'  {llave}: {valor:.4f}')

# Función para agregar ruido sal y pimienta
def agregar_ruido_sal_pimienta(imagen_metadata, cantidad):
    # Agregar validacion para cuando una imagen no sea escala de grises o binaria
    imagen_ruido = imagen_metadata.datos.copy()
    filas, columnas = imagen_ruido.shape
    num_ruido = int(cantidad * filas * columnas)
    for _ in range(num_ruido):
        i = np.random.randint(0, columnas)
        j = np.random.randint(0, filas)
        if np.random.rand() < 0.5:
            imagen_ruido[j, i] = 0 # Pimienta (negro)
        else:
            imagen_ruido[j, i] = 255 # Sal (blanco)
    
    # -------------------------------
    # Mostrar resultados
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(imagen_metadata.datos, cmap='gray')
    axs[0].set_title('Imagen original')
    axs[0].axis('off')
    axs[1].imshow(imagen_ruido, cmap='gray')
    axs[1].set_title('Ruido sal y pimienta')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    return imagen_ruido

# Función para agregar ruido gaussiano
def agregar_ruido_gaussiano(imagen_metadata, media=0, sigma=20):
    gauss = np.random.normal(media, sigma, imagen_metadata.datos.shape).astype(np.int16)
    imagen_ruido = imagen_metadata.datos.astype(np.int16) + gauss
    imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)

    # -------------------------------
    # Mostrar resultados
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(imagen_metadata.datos, cmap='gray')
    axs[0].set_title('Imagen original')
    axs[0].axis('off')
    axs[1].imshow(imagen_ruido, cmap='gray')
    axs[1].set_title('Ruido gaussiano')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    return imagen_ruido

def operaciones_aritmeticas(imagen_metadata):
    # Operaciones aritméticas 
    # Modificar este metodo para realizar las operaciones aritmeticas entre dos imagenes
    # ya que actualmente solo las esta realizando para una sola imagen

    suma = cv2.add(imagen_metadata.datos, 50) # Suma un escalar
    resta = cv2.subtract(imagen_metadata.datos, 50) # Resta un escalar
    multiplicacion = cv2.multiply(imagen_metadata.datos, 1.2) # Multiplica por un escalar

    # Mostrar resultados
    cv2.imshow('Imagen Original', imagen_metadata.datos)
    cv2.imshow('Imagen Suma', suma)
    cv2.imshow('Imagen Resta', resta)
    cv2.imshow('Imagen Multiplicación', multiplicacion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def operaciones_logicas():
    # Cargar dos imágenes
    img1 = cv2.imread(os.path.join(config.script_dir_parent, 'resources\\input\\estrella_naranja.jpeg'))
    img2 = cv2.imread(os.path.join(config.script_dir_parent, 'resources\\input\\estrella_verde.jpeg'))

    # Asegurarse de que las imágenes tengan el mismo tamaño
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # Operaciones lógicas
    and_img = cv2.bitwise_and(img1, img2)
    or_img = cv2.bitwise_or(img1, img2)
    xor_img = cv2.bitwise_xor(img1, img2)

    # Mostrar resultados
    cv2.imshow('Imagen AND', and_img)
    cv2.imshow('Imagen OR', or_img)
    cv2.imshow('Imagen XOR', xor_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu_principal():
    global ruta_imagen, imagen_actual, imagen_metadata
    
    # Carga inicial al arrancar el programa
    imagen_metadata = cargar_imagen_rgb_opencv(imagen_metadata)

    while True:
        print("\n" + "="*40)
        print("       EXPLORADOR DE IMAGEN DIGITAL")
        print("="*40)
        print(f" Archivo: [{imagen_metadata.nombre}]")
        print(f" Estado en memoria: [{imagen_metadata.modelo}]")
        print("-" * 40)
        print(" 1. Seleccionar nueva imagen.")
        print(" 2. Aplicar ruido: Sal y Pimienta.")
        print(" 3. Aplicar ruido: Gaussiano.")
        print(" 4. Operaciones Aritméticas.")
        print(" 5. Binarizar imagen (Otsu)")
        print(" 6. Operaciones Lógicas.")
        print(" 7. Convertir a modelo CMY.")
        print(" 8. Convertir a modelo YIQ.")
        print(" 9. Convertir a modelo HSI.")
        print(" 10. Calcular Histograma Automático (RGB, GRIS, HSV, ETC).")
        print(" 11. RESTABLECER IMAGEN ORIGINAL.")
        print(" 12. Salir.")
        print("="*40)
        
        opcion = input("Selecciona una Opción: ").strip()

        # --- GESTIÓN DE ARCHIVOS ---

        if opcion == "1":
            # crear metodo para cargar imagen con metadatos que luego usaremos a lo largo del código
            imagen_metadata.ruta = seleccionar_imagen() # Actualiza ruta_imagen
            imagen_metadata = cargar_imagen_rgb_opencv(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)
        
        elif opcion == "11":
            # Recarga el archivo original descartando cambios previos
            imagen_metadata = cargar_imagen_rgb_opencv(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        # --- PROCESAMIENTO ---

        elif opcion == "2":
            # Aplicar ruido sal y pimienta a una imagen en escala de grises
            imagen_sal_pimienta = agregar_ruido_sal_pimienta(imagen_metadata, cantidad=0.02)

        elif opcion == "3":
            # Aplicar ruido gaussiano a una imagen en escala de grises
            imagen_gaussiana = agregar_ruido_gaussiano(imagen_metadata, media=0, sigma=20)

        elif opcion == "4":
            # Aplicamos operaciones aritmeticas
            operaciones_aritmeticas(imagen_metadata)
        
        elif opcion == "5":
            imagen_metadata = binarizar_imagen_otsu(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        elif opcion == "6":
            # Aplicamos operaciones lógicas
            operaciones_logicas()

        elif opcion == "7":
            imagen_actual = convertir_a_cmy(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)
        
        elif opcion == "8":
            imagen_actual = convertir_a_yiq(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        elif opcion == "9":
            imagen_actual = convertir_a_hsi(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        # --- ANÁLISIS ---
        elif opcion == "10":
            imagen_metadata = calcular_histograma_automatico(imagen_metadata)

        # --- SALIDA ---
        elif opcion == "12" or opcion.upper() == "SALIR":
            print(" Saliendo del programa...")
            break
        else:
            print(" Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu_principal()

