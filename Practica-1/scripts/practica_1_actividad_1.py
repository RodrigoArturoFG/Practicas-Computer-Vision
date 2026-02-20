# --------- PRACTICA 1 "CREANDO MI MAPA DE CALOR" ---------
# --------- ACTIVIDAD 1 -----------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-18-2026

import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

from config import mapas_color # Importar el diccionario de mapas de color desde config.py
from config import script_dir  # Importar la variable script_dir desde config.py
from imagen_pseudocolor import ImagenPseudocolor 



# --------- VARIABLES GLOBALES ---------
imagen_path = os.path.join(script_dir, 'resources/input/rostro-humano.jpg')

# --------- FUNCIONES DE PROCESAMIENTO DE IMAGENES ---------

# Función para comparar visualmente la imagen con varias versiones pseudocoloreadas
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


# Menú para seleccionar un mapa de color específico y aplicarlo a la imagen en escala de grises
def menu_mapas_color(imagen_gris):
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
        print("4. Salir del Programa")
        opcion = input("Selecciona una Opción: ").strip()

        if opcion == "1":
            seleccionar_imagen()
        if opcion == "2":
            imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if imagen_gris is None:
                print("No se pudo cargar la imagen. Verifica la ruta y extensión.")
                continue
            menu_mapas_color(imagen_gris)
        elif opcion == "3":
            imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if imagen_gris is None:
                print("No se pudo cargar la imagen. Verifica la ruta y extensión.")
                continue
            comparar_mapas_color(imagen_gris)
        elif opcion == "4" or opcion.upper() == "SALIR":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    menu_principal()
