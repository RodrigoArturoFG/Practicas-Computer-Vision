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

    def mostrar(self) -> None:
        """
        Método auxiliar para visualizar la imagen pseudocolor con matplotlib.
        Usa el atributo 'nombre' como título de la figura.
        """
        # Convertir la imagen de BGR a RGB para mostrarla correctamente con matplotlib
        plt.imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
        plt.title(f"Pseudocolor: {self.nombre}")
        plt.axis("off")
        plt.show()
    
    def guardar(self, ruta_base: str = "resultado") -> str:
        """
        Método auxiliar para guardar la imagen pseudocolor en disco.
        El archivo se nombra automáticamente con el colormap seleccionado.
        
        Parámetros:
            ruta_base (str): Nombre base del archivo (sin extensión).
        
        Retorna:
            str: Ruta completa del archivo guardado.
        """
        # Construir el nombre del archivo utilizando el nombre del mapa de color y la ruta base proporcionada
        nombre_imagen = f"{ruta_base}_{self.nombre}.png"
        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')

        # Crear la carpeta si no existe
        os.makedirs(ruta_carpeta, exist_ok=True)

        # Construir la ruta completa del archivo y guardar la imagen utilizando OpenCV
        ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
        cv2.imwrite(ruta_imagen, self.imagen)

        # Retornar la ruta completa del archivo guardado para referencia futura
        return ruta_imagen
