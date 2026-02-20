# --------- CLASE IMAGEN PSEUDOCOLOR ---------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-19-2026

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import mapas_color # Importar el diccionario de mapas de color desde config.py
from config import script_dir  # Importar la variable script_dir desde config.py

class ImagenPseudocolor:
    """
    Clase para aplicar y almacenar el resultado de un mapa de color (pseudocolor) a una imagen en escala de grises.
    Atributos: 
        - nombre: str → identificador del mapa de color aplicado
        - imagen: np.ndarray → imagen pseudocoloreada
    """
    def __init__(self, imagen_gris: np.ndarray, nombre: str) -> None:
        """
        Constructor que aplica el mapa de color indicado a la imagen en escala de grises.
        imagen_gris: np.ndarray → imagen en escala de grises
        nombre: str → identificador del mapa de color a aplicar
        """
        # Convertir el nombre a mayúsculas para asegurar la coincidencia con las claves del diccionario
        self.nombre: str = nombre.upper()

        # Verificar que el nombre del mapa de color sea válido antes de aplicar el colormap
        if self.nombre not in mapas_color:
            raise ValueError(f"Opción '{self.nombre}' no válida. Opciones disponibles: {list(mapas_color.keys())}")
        
        # Aplicar el mapa de color a la imagen en escala de grises utilizando OpenCV
        self.imagen: np.ndarray = cv2.applyColorMap(imagen_gris, mapas_color[self.nombre])

    @classmethod
    def aplicar_pseudocolor(cls, imagen_gris: np.ndarray, opcion: str):
        """
        Método de clase para crear un objeto ImagenPseudocolor aplicando el colormap deseado.
        """
        # Retorna una instancia de la clase aplicando el mapa de color seleccionado
        return cls(imagen_gris, opcion)

    def mostrar(self, imagen_gris: np.ndarray = None) -> None:
        """
        Visualiza la imagen pseudocolor sola o junto con la imagen en escala de grises si se proporciona.
        Si imagen_gris es None, solo muestra la pseudocolor.
        """
        if imagen_gris is not None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Imagen en escala de grises')
            axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f'Pseudocolor: {self.nombre}')
            axs[1].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Pseudocolor: {self.nombre}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def guardar(self, ruta_base: str = "resultado", imagen_gris: np.ndarray = None) -> str:
        """
        Guarda la imagen pseudocolor sola o junto con la imagen en escala de grises si se proporciona.
        Si imagen_gris es None, solo guarda la pseudocolor.
        El archivo se nombra automáticamente con el colormap seleccionado.
        Retorna la ruta completa del archivo guardado.
        """
        if imagen_gris is not None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Imagen en escala de grises')
            axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f'Pseudocolor: {self.nombre}')
            axs[1].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Pseudocolor: {self.nombre}')
            ax.axis('off')
        plt.tight_layout()

        nombre_imagen = f"{ruta_base}_{self.nombre}.png"
        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
        os.makedirs(ruta_carpeta, exist_ok=True)
        ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
        fig.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return ruta_imagen
