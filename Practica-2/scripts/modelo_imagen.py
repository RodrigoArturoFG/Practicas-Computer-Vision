# ----- ENTIDADES DE DATOS - IMAGENES --------
# --------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 08-03-2026

import os

class metadataImagen:

    def __init__(self, ruta):
        self.ruta = ruta
        self.nombre = os.path.basename(ruta)
        self.modelo = "RGB"  # Modelo inicial por defecto
        self.datos = None    # Array de NumPy (así se guarda la información de una imagen en Open CV)
        self.umbral = None   # El valor numérico del umbral que se utilizó para binarización
        self.histograma = {} # Diccionario: {'Canal': {'Media': 0.0, ...}}

    def actualizar_modelo(self, nuevo_modelo):
        """Actualiza el modelo de color y la estampa de tiempo."""
        self.modelo = nuevo_modelo.upper()

    def registrar_histograma(self, resultados_dict):
        """Almacena los cálculos estadísticos en el objeto."""
        self.histograma = resultados_dict
