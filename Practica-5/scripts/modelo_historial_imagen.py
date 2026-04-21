# ----- ENTIDADES DE DATOS - IMAGENES --------
# --------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 08-03-2026

import os

class metadataHistorialImagen:

    def __init__(self, ruta):
        self.ruta = ruta
        self.nombre = os.path.basename(ruta)
        self.modelo = "RGB"  # Modelo inicial por defecto
        self.thumbnail = None    # Array de NumPy (así se guarda la información de la miniatura de la imagen)
        self.umbral = None   # El valor numérico del umbral que se utilizó para binarización
        self.histograma = {} # Diccionario: {'Canal': {'Media': 0.0, ...}}
        self.es_derivable = True  # True: se puede reconstruir desde disco + metadatos
        self.datos = None         # Array NumPy — solo se guarda cuando es_derivable=False
        self.es_resultado_logico = False  # True cuando proviene de AND/OR/XOR/NOT/Relacional