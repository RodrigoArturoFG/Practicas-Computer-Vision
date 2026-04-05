# ----- PRÁCTICA 4 - "MM BINARIA Y EN LATTICES" -----
# ---------------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 01-04-2026
# Version: 1.0

import matplotlib.pyplot as plt
from scipy.stats import skew
from math import log2
import cv2
import os

import config
import controlador_imagen as procesadorImagen
from modelo_imagen import metadataImagen

# --------- VARIABLES GLOBALES ---------
ruta_imagen = os.path.join(config.script_dir_parent, 'resources/input/estrella_amarilla.jpeg')
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

# Función para agregar ruido SAL (píxeles blancos aleatorios)
def agregar_ruido_sal(imagen_metadata, cantidad=0.02):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.agregar_ruido_sal(imagen_metadata, cantidad)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata


# Función para agregar ruido PIMIENTA (píxeles negros aleatorios)
def agregar_ruido_pimienta(imagen_metadata, cantidad=0.02):
    # Llamamos a la función y recibimos el wrapper con la respuesta
    respuesta = procesadorImagen.agregar_ruido_pimienta(imagen_metadata, cantidad)

    # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
    imagen_metadata = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        
    return imagen_metadata


# Función para agregar ruido gaussiano
def agregar_ruido_gaussiano(imagen_metadata, media=0, sigma=20):
    respuesta = procesadorImagen.agregar_ruido_gaussiano(imagen_metadata, media, sigma)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


# Selecciona una imagen secundaria sin tocar la variable global ruta_imagen
def seleccionar_imagen_secundaria():
    """
    Igual que seleccionar_imagen() pero sin modificar el estado global.
    Devuelve un objeto metadataImagen cargado en RGB, o None si se cancela.
    """
    carpeta_input = os.path.join(config.script_dir_parent, 'resources/input')
    archivos = [f for f in os.listdir(carpeta_input) if os.path.isfile(os.path.join(carpeta_input, f))]
    if not archivos:
        print("No se encontraron imágenes en la carpeta resources/input.")
        return None

    print("\n=== Seleccionar Imagen Secundaria (B) ===")
    for i, nombre in enumerate(archivos, start=1):
        print(f"  {i}. {nombre}")
    print(f"  {len(archivos)+1}. Cancelar")

    opcion = input("Selecciona el número de la imagen B: ").strip()
    if not opcion.isdigit():
        print("[!] Entrada inválida. Operación cancelada.")
        return None

    idx = int(opcion) - 1
    if idx == len(archivos):
        print("Selección cancelada.")
        return None
    if not (0 <= idx < len(archivos)):
        print("[!] Número fuera de rango. Operación cancelada.")
        return None

    ruta_b = os.path.join(carpeta_input, archivos[idx])
    imagen_b = metadataImagen(ruta_b)
    respuesta = procesadorImagen.cargar_imagen_opencv_rgb(imagen_b)
    imagen_b = respuesta["objeto"]

    if respuesta["error"]:
        print(f"\n[!] ERROR al cargar imagen B: {respuesta['mensaje']}")
        return None

    print(f"\n[*] Imagen B cargada: [{imagen_b.nombre}]")
    return imagen_b


# --- Wrappers de operaciones aritméticas ---

def sumar_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.sumar_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def restar_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.restar_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def multiplicar_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.multiplicar_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def submenu_operaciones_aritmeticas(imagen_metadata):
    """
    Submenú para operaciones aritméticas entre dos imágenes.
    Pide la operación, luego la imagen B, ejecuta y muestra el resultado.
    """
    print("\n--- Operaciones Aritméticas ---")
    print(" 1. Suma       (A + B)")
    print(" 2. Resta      (A - B)")
    print(" 3. Multiplicación (A × B)")
    print(" 4. Cancelar")
    opcion = input("Selecciona una operación: ").strip()

    if opcion == "4" or opcion == "":
        print("Operación cancelada.")
        return imagen_metadata

    if opcion not in ("1", "2", "3"):
        print("[!] Opción inválida.")
        return imagen_metadata

    # Seleccionar imagen B (sin tocar el estado global)
    imagen_b = seleccionar_imagen_secundaria()
    if imagen_b is None:
        return imagen_metadata

    if opcion == "1":
        imagen_metadata = sumar_imagenes(imagen_metadata, imagen_b)
    elif opcion == "2":
        imagen_metadata = restar_imagenes(imagen_metadata, imagen_b)
    elif opcion == "3":
        imagen_metadata = multiplicar_imagenes(imagen_metadata, imagen_b)

    mostrar_imagen_opencv(imagen_metadata)
    return imagen_metadata

# --- Wrappers de operaciones lógicas ---

def and_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.and_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def or_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.or_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def xor_imagenes(imagen_metadata, imagen_b):
    respuesta = procesadorImagen.xor_imagenes(imagen_metadata, imagen_b)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def not_imagen(imagen_metadata):
    respuesta = procesadorImagen.not_imagen(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


# --- Wrappers de operaciones relacionales ---

def relacional_mayor(imagen_metadata, umbral):
    respuesta = procesadorImagen.relacional_mayor(imagen_metadata, umbral)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def relacional_menor(imagen_metadata, umbral):
    respuesta = procesadorImagen.relacional_menor(imagen_metadata, umbral)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def relacional_igual(imagen_metadata, umbral):
    respuesta = procesadorImagen.relacional_igual(imagen_metadata, umbral)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata


def submenu_operaciones_logicas(imagen_metadata):
    """
    Submenú para operaciones lógicas y relacionales.
    Lógicas (AND/OR/XOR/NOT): operan sobre imágenes binarias.
    Relacionales (>/</==): comparan intensidad de grises contra un umbral escalar.
    """
    print("\n--- Operaciones Lógicas y Relacionales ---")
    print(" Lógicas (requieren imagen B):")
    print("  1. AND  — blanco donde AMBAS tienen blanco")
    print("  2. OR   — blanco donde AL MENOS UNA tiene blanco")
    print("  3. XOR  — blanco donde las imágenes son DIFERENTES")
    print(" Lógica (solo imagen A):")
    print("  4. NOT  — invierte fondo y objetos")
    print(" Relacionales (comparan contra umbral):")
    print("  5. Mayor que  ( > umbral )")
    print("  6. Menor que  ( < umbral )")
    print("  7. Igual a    ( == umbral )")
    print("  8. Cancelar")
    opcion = input("Selecciona una operación: ").strip()

    if opcion == "8" or opcion == "":
        print("Operación cancelada.")
        return imagen_metadata

    # --- Operaciones lógicas con imagen B ---
    if opcion in ("1", "2", "3"):
        imagen_b = seleccionar_imagen_secundaria()
        if imagen_b is None:
            return imagen_metadata
        if opcion == "1":
            imagen_metadata = and_imagenes(imagen_metadata, imagen_b)
        elif opcion == "2":
            imagen_metadata = or_imagenes(imagen_metadata, imagen_b)
        elif opcion == "3":
            imagen_metadata = xor_imagenes(imagen_metadata, imagen_b)

    # --- NOT: solo imagen A ---
    elif opcion == "4":
        imagen_metadata = not_imagen(imagen_metadata)

    # --- Operaciones relacionales con umbral ---
    elif opcion in ("5", "6", "7"):
        umbral = solicitar_umbral()
        if opcion == "5":
            imagen_metadata = relacional_mayor(imagen_metadata, umbral)
        elif opcion == "6":
            imagen_metadata = relacional_menor(imagen_metadata, umbral)
        elif opcion == "7":
            imagen_metadata = relacional_igual(imagen_metadata, umbral)

    else:
        print("[!] Opción inválida.")
        return imagen_metadata

    mostrar_imagen_opencv(imagen_metadata)
    return imagen_metadata

def _visualizar_vecindad(imagen_metadata, respuesta, titulo_labels):
    """
    Visualiza los resultados de un análisis de vecindad:
    mapa de etiquetas coloreado + imagen con contornos numerados.
    """
    print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    labels           = respuesta["labels"]
    imagen_contornos = respuesta["imagen_contornos"]
    num_objetos      = respuesta["num_objetos"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].imshow(labels, cmap='jet')
    axs[0].set_title(f'{titulo_labels} — {num_objetos} objeto(s)')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f'Contornos numerados | [{imagen_metadata.nombre}]')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    input("\nPresiona ENTER para continuar...")


def analizar_vecindad_4(imagen_metadata):
    respuesta = procesadorImagen.analizar_vecindad_4(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
        return imagen_metadata
    _visualizar_vecindad(imagen_metadata, respuesta, "Vecindad-4")
    return imagen_metadata


def analizar_vecindad_8(imagen_metadata):
    respuesta = procesadorImagen.analizar_vecindad_8(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
        return imagen_metadata
    _visualizar_vecindad(imagen_metadata, respuesta, "Vecindad-8")
    return imagen_metadata


def submenu_analizar_vecindad(imagen_metadata):
    """
    Submenú para el etiquetado de componentes conexas.
    Permite analizar con vecindad-4, vecindad-8 o comparar ambas en una sola figura.
    La imagen se binariza automáticamente si no lo está.
    """
    print("\n--- Etiquetado de Componentes Conexas ---")
    print(" 1. Vecindad-4  (conexiones ortogonales)")
    print(" 2. Vecindad-8  (ortogonales + diagonales)")
    print(" 3. Comparar ambas")
    print(" 4. Cancelar")
    opcion = input("Selecciona una opción: ").strip()

    if opcion == "4" or opcion == "":
        print("Operación cancelada.")
        return imagen_metadata

    if opcion not in ("1", "2", "3"):
        print("[!] Opción inválida.")
        return imagen_metadata

    if opcion == "1":
        imagen_metadata = analizar_vecindad_4(imagen_metadata)

    elif opcion == "2":
        imagen_metadata = analizar_vecindad_8(imagen_metadata)

    elif opcion == "3":
        # Obtener resultados de ambas vecindades
        resp4 = procesadorImagen.analizar_vecindad_4(imagen_metadata)
        resp8 = procesadorImagen.analizar_vecindad_8(imagen_metadata)

        if resp4["error"]:
            print(f"\n[!] ERROR vecindad-4: {resp4['mensaje']}")
            return imagen_metadata
        if resp8["error"]:
            print(f"\n[!] ERROR vecindad-8: {resp8['mensaje']}")
            return imagen_metadata

        imagen_metadata = resp8["objeto"]  # persiste el estado de la última operación

        n4 = resp4["num_objetos"]
        n8 = resp8["num_objetos"]
        print(f"\n[*] Vecindad-4: {n4} objeto(s)  |  Vecindad-8: {n8} objeto(s)")
        print(f"    Diferencia: {abs(n4 - n8)} objeto(s)")

        # Figura comparativa de 2×2: etiquetas y contornos de cada vecindad
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Comparación de Vecindad | [{imagen_metadata.nombre}]',
            fontsize=13
        )

        axs[0, 0].imshow(resp4["labels"], cmap='jet')
        axs[0, 0].set_title(f'Vecindad-4 — mapa de etiquetas ({n4} obj.)')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(cv2.cvtColor(resp4["imagen_contornos"], cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title('Vecindad-4 — contornos numerados')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(resp8["labels"], cmap='jet')
        axs[1, 0].set_title(f'Vecindad-8 — mapa de etiquetas ({n8} obj.)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(cv2.cvtColor(resp8["imagen_contornos"], cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title('Vecindad-8 — contornos numerados')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
        input("\nPresiona ENTER para continuar...")

    return imagen_metadata

def erosion(imagen_metadata, kernel_size=11, iteraciones=2, forma="disco"):
    respuesta = procesadorImagen.erosion(imagen_metadata, kernel_size, iteraciones, forma)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata

def dilatacion(imagen_metadata, kernel_size=11, iteraciones=2, forma="disco"):
    respuesta = procesadorImagen.dilatacion(imagen_metadata, kernel_size, iteraciones, forma)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata

def apertura(imagen_metadata, kernel_size=11, iteraciones=1, forma="disco"):
    respuesta = procesadorImagen.apertura(imagen_metadata, kernel_size, iteraciones, forma)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata

def cierre(imagen_metadata, kernel_size=22, iteraciones=1, forma="disco"):
    respuesta = procesadorImagen.cierre(imagen_metadata, kernel_size, iteraciones, forma)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        print(f"\n[!] ERROR: {respuesta['mensaje']}")
    else:
        print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
    return imagen_metadata



def agregar_ruido_sal_pimienta(imagen_metadata, cantidad=0.02):
    """Aplica ruido bipolar (sal y pimienta) en una sola llamada."""
    imagen_metadata = agregar_ruido_sal(imagen_metadata, cantidad)
    imagen_metadata = agregar_ruido_pimienta(imagen_metadata, cantidad)
    return imagen_metadata


def _solicitar_porcentaje(nombre, defecto=2):
    """Pide un porcentaje entre 0 y 100 y lo devuelve como flotante 0-1."""
    entrada = input(f"  Porcentaje de {nombre} (0-100) [Enter={defecto}%]: ").strip()
    if entrada == "":
        return defecto / 100.0
    if entrada.replace('.', '', 1).isdigit():
        val = float(entrada)
        if 0 <= val <= 100:
            return val / 100.0
    print(f"  [!] Valor inválido, usando {defecto}%.")
    return defecto / 100.0


def _solicitar_sigma(defecto=20):
    """Pide el valor de sigma para ruido gaussiano."""
    entrada = input(f"  Sigma (desviación estándar) [Enter={defecto}]: ").strip()
    if entrada == "":
        return defecto
    if entrada.isdigit():
        val = int(entrada)
        if 1 <= val <= 255:
            return val
    print(f"  [!] Valor inválido, usando {defecto}.")
    return defecto


def submenu_ruido(imagen_metadata):
    """
    Submenú de ruido: unipolar sal, unipolar pimienta, bipolar sal+pimienta, gaussiano.
    """
    print("\n  --- Ruido ---")
    print("   1. Sal         (unipolar — píxeles blancos aleatorios)")
    print("   2. Pimienta     (unipolar — píxeles negros aleatorios)")
    print("   3. Sal+Pimienta (bipolar)")
    print("   4. Gaussiano")
    print("   5. Cancelar")
    opcion = input("  Selecciona tipo de ruido: ").strip()

    if opcion == "1":
        cantidad = _solicitar_porcentaje("sal")
        imagen_metadata = agregar_ruido_sal(imagen_metadata, cantidad)
    elif opcion == "2":
        cantidad = _solicitar_porcentaje("pimienta")
        imagen_metadata = agregar_ruido_pimienta(imagen_metadata, cantidad)
    elif opcion == "3":
        cantidad = _solicitar_porcentaje("sal+pimienta")
        imagen_metadata = agregar_ruido_sal_pimienta(imagen_metadata, cantidad)
    elif opcion == "4":
        sigma = _solicitar_sigma()
        imagen_metadata = agregar_ruido_gaussiano(imagen_metadata, media=0, sigma=sigma)
    else:
        print("  Operación cancelada.")
        return imagen_metadata

    mostrar_imagen_opencv(imagen_metadata)
    return imagen_metadata

def _solicitar_forma_kernel():
    """
    Submenú para elegir la forma del elemento estructurante.
    Retorna: "disco" o "cuadrado", o None si el usuario cancela.
    """
    print("\n  --- Forma del Elemento Estructurante ---")
    print("   1. Disco    (preserva formas convexas, bordes naturales)")
    print("   2. Cuadrado (más agresivo en esquinas)")
    print("   3. Cancelar")
    opcion = input("  Selecciona una forma: ").strip()
    if opcion == "1":
        return "disco"
    elif opcion == "2":
        return "cuadrado"
    else:
        print("  Operación cancelada.")
        return None


def submenu_morfologia_basica(imagen_metadata, operacion):
    """
    Submenú genérico para las 4 operaciones morfológicas.
    Pide la forma del EE y ejecuta la operación indicada con los defaults agresivos.

    operacion: "erosion" | "dilatacion" | "apertura" | "cierre"
    """
    forma = _solicitar_forma_kernel()
    if forma is None:
        return imagen_metadata

    ops = {
        "erosion":    erosion,
        "dilatacion": dilatacion,
        "apertura":   apertura,
        "cierre":     cierre,
    }
    return ops[operacion](imagen_metadata, forma=forma)



def submenu_morfologia_binaria(imagen_metadata):
    """
    Submenú de Morfología Binaria avanzada.
    Requiere imagen BINARIO (binarizar primero con opción 2 ó 3).
    """
    print("\n  --- Morfología Binaria Avanzada ---")
    print("   1. Frontera")
    print("   2. Hit or Miss")
    print("   3. Adelgazamiento")
    print("   4. Esqueleto Morfológico")
    print("   5. Cancelar")
    opcion = input("  Selecciona una operación: ").strip()

    if opcion == "1":
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        respuesta = procesadorImagen.frontera(imagen_metadata, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "2":
        respuesta = procesadorImagen.hit_or_miss(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "3":
        respuesta = procesadorImagen.adelgazamiento(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "4":
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        respuesta = procesadorImagen.esqueleto(imagen_metadata, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    else:
        print("  Operación cancelada.")

    return imagen_metadata


def submenu_morfologia_laticces(imagen_metadata):
    """
    Submenú de Morfología en Laticces (grises).
    Requiere imagen GRIS (convertir con modelo Escala de Grises primero).
    """
    print("\n  --- Morfología en Laticces ---")
    print("   1. Gradiente morfológico simétrico")
    print("   2. Gradiente por dilatación")
    print("   3. Gradiente por erosión")
    print("   4. Top Hat")
    print("   5. Bot Hat (Black Hat)")
    print("   6. Suavizado morfológico (Apertura → Cierre)")
    print("   7. Cancelar")
    opcion = input("  Selecciona una operación: ").strip()

    ops_gradiente = {"1": "simetrico", "2": "dilatacion", "3": "erosion"}
    if opcion in ops_gradiente:
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        tipo = ops_gradiente[opcion]
        respuesta = procesadorImagen.gradiente_morfologico(imagen_metadata, tipo=tipo, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "4":
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        respuesta = procesadorImagen.top_hat(imagen_metadata, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "5":
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        respuesta = procesadorImagen.bot_hat(imagen_metadata, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    elif opcion == "6":
        forma = _solicitar_forma_kernel()
        if forma is None: return imagen_metadata
        respuesta = procesadorImagen.suavizado_morfologico(imagen_metadata, forma=forma)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]: print(f"\n[!] ERROR: {respuesta['mensaje']}")
        else: print(f"\n[*] ÉXITO: {respuesta['mensaje']}")
        mostrar_imagen_opencv(imagen_metadata)

    else:
        print("  Operación cancelada.")

    return imagen_metadata

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
        print(" 2. Binarizar imagen (Umbral personalizado).")
        print(" 3. Binarizar imagen (Otsu).")
        print(" 4. Ruido (Sal / Pimienta / Gaussiano).")
        print(" 5. Morfología Básica (Dilatación/Erosión/Apertura/Cierre).")
        print(" 6. Morfología Binaria avanzada.")
        print(" 7. Morfología en Laticces.")
        print(" 8. Operaciones Lógicas y Relacionales.")
        print(" 9. RESTABLECER IMAGEN ORIGINAL.")
        print("10. Salir.")
        print("="*40)
        
        opcion = input("Selecciona una Opción: ").strip()

        # --- GESTIÓN DE ARCHIVOS ---

        if opcion == "1":
            imagen_metadata.ruta = seleccionar_imagen()
            imagen_metadata = cargar_imagen_rgb_opencv(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)
        
        # --- PROCESAMIENTO ---
        elif opcion == "2":
            umbral_usuario = solicitar_umbral()
            imagen_metadata = binarizar_imagen(imagen_metadata, umbral_usuario)
            mostrar_imagen_opencv(imagen_metadata)

        elif opcion == "3":
            imagen_metadata = binarizar_imagen_otsu(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        elif opcion == "4":
            imagen_metadata = submenu_ruido(imagen_metadata)

        elif opcion == "5":
            imagen_metadata = submenu_morfologia_basica(imagen_metadata)

        elif opcion == "6":
            imagen_metadata = submenu_morfologia_binaria(imagen_metadata)

        elif opcion == "7":
            imagen_metadata = submenu_morfologia_laticces(imagen_metadata)

        elif opcion == "8":
            imagen_metadata = submenu_operaciones_logicas(imagen_metadata)

        # --- GESTIÓN ---
        elif opcion == "9":
            imagen_metadata = cargar_imagen_rgb_opencv(imagen_metadata)
            mostrar_imagen_opencv(imagen_metadata)

        # --- SALIDA ---
        elif opcion == "10" or opcion.upper() == "SALIR":
            print(" Saliendo del programa...")
            break
        else:
            print(" Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    menu_principal()