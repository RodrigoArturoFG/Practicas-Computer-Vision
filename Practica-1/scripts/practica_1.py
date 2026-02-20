# --------- PRACTICA 1 "CREANDO MI MAPA DE CALOR" ---------
# --------- ACTIVIDAD 1 -----------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-18-2026

import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Importar elementos locales
from imagen_pseudocolor import ImagenPseudocolor  # Importar la clase ImagenPseudocolor
from config import mapas_color # Importar el diccionario de mapas de color desde config.py
from config import script_dir  # Importar la variable script_dir desde config.py
from config import colores_pastel, colores_tierra, colores_pastel_personalizados # Importar las listas de colores personalizados desde config.py


# --------- VARIABLES GLOBALES ---------
imagen_path = os.path.join(script_dir, 'resources\\input\\rostro_humano.jpg')

# --------- FUNCIONES DE PROCESAMIENTO DE IMAGENES ---------
def comparar_mapas_color(imagen_gris):
    """
    Muestra la imagen en escala de grises y todas las versiones pseudocoloreadas disponibles.
    Organizando la cuadrícula de manera dinámica según la cantidad de paletas de OpenCV detectadas.
    """

    print("Creando imagenes con pseudocolor con todos los mapas de color disponibles en OpenCV...")
    
    # Obtener la lista de nombres de colormaps disponibles
    nombres_colormaps = list(mapas_color.keys())
    n_colormaps = len(nombres_colormaps)
    total_imgs = n_colormaps + 1  # +1 para la imagen en escala de grises

    # Calcular filas y columnas para la cuadrícula
    n_cols = min(5, total_imgs)  # máximo 5 columnas para mejor visualización
    n_rows = int(np.ceil(total_imgs / n_cols))

    # Crear la figura y los ejes para la cuadrícula
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axs = np.array(axs).reshape(-1)  # aplanar para indexar fácilmente

    # Imagen en escala de grises en la primera posición
    axs[0].imshow(imagen_gris, cmap='gray')
    axs[0].set_title('Escala de grises')
    axs[0].axis('off')

    # Agregar cada pseudocolor en la cuadrícula y el título correspondiente
    for idx, nombre in enumerate(nombres_colormaps):
        pseudocolor = ImagenPseudocolor.aplicar_pseudocolor(imagen_gris, nombre)
        axs[idx+1].imshow(cv2.cvtColor(pseudocolor.imagen, cv2.COLOR_BGR2RGB))
        axs[idx+1].set_title(nombre)
        axs[idx+1].axis('off')

    # Desactivar ejes sobrantes si hay
    for ax in axs[total_imgs:]:
        ax.axis('off')

    # Ajustar el layout para evitar recortes y superposiciones
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(bottom=0.03, top=0.97, left=0.03, right=0.97, hspace=0.25, wspace=0.15)

    # Guardar la figura antes de mostrarla, sin elementos extra
    # El nombre del archivo incluye la fecha y hora para evitar sobreescrituras
    nombre_archivo = f"comparacion_mapas_color_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
    os.makedirs(ruta_carpeta, exist_ok=True)
    ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
    # Guardar la figura con un pequeño margen para evitar recortes de títulos o bordes
    fig.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
    print(f"Comparación guardada en: {ruta_imagen}")

    plt.show()


def menu_mapas_color(imagen_gris):
    """
    Menú que permite al usuario seleccionar un mapa de color específico para aplicar a la imagen en escala de grises.
    Muestra la imagen resultante y ofrece la opción de guardarla.
    """
    
    while True:
        print("\n=== Menú de Mapas de Color Disponibles ===")
        for i, nombre in enumerate(mapas_color.keys(), start=1):
            print(f"{i}. {nombre}")
        print(f"{len(mapas_color)+1}. Regresar al Menú Principal [Regresar, Salir]")

        opcion_usuario = input("Selecciona un mapa de color por nombre o número: ").strip()

        # Permitir selección por número además de por nombre
        if opcion_usuario.isdigit():
            indice = int(opcion_usuario) - 1
            if indice == len(mapas_color):
                print("Regresando al Menú Principal...")
                break
            elif 0 <= indice < len(mapas_color):
                opcion_usuario = list(mapas_color.keys())[indice]
            else:
                print("Número fuera de rango.")
                continue
        elif opcion_usuario.upper() in ["REGRESAR", "SALIR"]:
            print("Regresando al Menú Principal...")
            break

        try:
            resultado = ImagenPseudocolor.aplicar_pseudocolor(imagen_gris, opcion_usuario)
            resultado.mostrar()
            image_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ruta_guardada = resultado.guardar(image_name)
            print(f"Imagen guardada en: {ruta_guardada}")
            pausa = input("Presiona Enter para continuar...")
        except ValueError as e:
            print("Error:", e)


def mostrar_personalizacion_mapas(imagen_gris):
    """
    Visualiza la imagen en escala de grises y con dos mapas de color personalizados (pastel y tierra).
    """

    print("Mostrando imagen en escala de grises con mapas de color personalizados...")

    # Este ejemplo permite crear una imagen en escala de grises como gradiente horizontal (prueba usando otra imagen)
    # Cada fila tiene valores de 0 a 255 en forma de gradiente
    # imagen_gris = np.tile(np.linspace(0, 255, 256), (100,1)).astype(np.uint8)

    # Crear los mapas de color personalizados utilizando LinearSegmentedColormap
    mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
    mapa_tierra = LinearSegmentedColormap.from_list("TierraMap", colores_tierra, N=256)
    mapa_pastel_personalizado = LinearSegmentedColormap.from_list("PastelPersonalizadoMap", colores_pastel_personalizados, N=256)

    # Visualizar la imagen original y la imagen con pseudocolor pastel y tierra
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = np.array(axs).reshape(-1)

    axs[0].imshow(imagen_gris, cmap='gray')
    axs[0].set_title('Imagen en escala de grises')
    axs[0].axis('off')

    axs[1].imshow(imagen_gris, cmap=mapa_pastel)
    axs[1].set_title('Mapa de color pastel')
    axs[1].axis('off')

    axs[2].imshow(imagen_gris, cmap=mapa_tierra)
    axs[2].set_title('Mapa de color tierra')
    axs[2].axis('off')

    axs[3].imshow(imagen_gris, cmap=mapa_pastel_personalizado)
    axs[3].set_title('Mapa de color pastel personalizado')
    axs[3].axis('off')

    plt.tight_layout()

    # Guardar la figura antes de mostrarla, sin elementos extra
    # El nombre del archivo incluye la fecha y hora para evitar sobreescrituras
    nombre_archivo = f"mapas_color_personalizados_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
    os.makedirs(ruta_carpeta, exist_ok=True)
    ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
    # Guardar la figura con un pequeño margen para evitar recortes de títulos o bordes
    fig.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
    print(f"Comparación guardada en: {ruta_imagen}")

    plt.show()

# --------- FUNCIONES DE PROCESAMIENTO DE IMAGENES ---------
def seleccionar_imagen():
    """
    Permite al usuario seleccionar una imagen de la carpeta resources/input.
    Actualiza la variable global imagen_path.
    """
    # Obtener la lista de archivos en la carpeta de entrada
    global imagen_path
    carpeta_input = os.path.join(script_dir, 'resources/input')
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
                return
            elif 0 <= idx < len(archivos):
                # Actualizar la ruta de la imagen seleccionada
                imagen_path = os.path.join(carpeta_input, archivos[idx])
                print(f"Imagen seleccionada: {imagen_path}")
                return
            else:
                print("Número fuera de rango.")
        else:
            print("Ingresa un número válido.")


def menu_principal():
    global imagen_path
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print(f"Imagen actual: {imagen_path}")
        print("1. Seleccionar Imagen a Procesar")
        print("2. Aplicar un Mapa de Color a la Imagen en Escala de Grises")
        print("3. Comparación Visual de Mapas de Color Disponibles en OpenCV")
        print("4. Personalización del Mapa de Color")
        print("5. Salir del Programa")
        opcion = input("Selecciona una Opción: ").strip()

        if opcion == "1":
            seleccionar_imagen()
        elif opcion == "2":
            imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if imagen_gris is None:
                raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y extensión.")
                
            menu_mapas_color(imagen_gris)
        elif opcion == "3":
            imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if imagen_gris is None:
                raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y extensión.")
            comparar_mapas_color(imagen_gris)
        elif opcion == "4":
            imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if imagen_gris is None:
                raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y extensión.")
            mostrar_personalizacion_mapas(imagen_gris)
        elif opcion == "5" or opcion.upper() == "SALIR":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    menu_principal()
