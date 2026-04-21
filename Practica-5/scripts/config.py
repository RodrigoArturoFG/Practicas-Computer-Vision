# --------- CONFIGURACIÓN DE PARÁMETROS PARA LA PRÁCTICA 3  ---------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-03-2026

import os

# Obtiene la ruta absoluta del archivo actual y luego su directorio
script_dir = os.path.dirname(os.path.abspath(__file__))

# Obtener el directorio del script y regresar un nivel en la jerarquía de carpetas
script_dir_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Definir nombre y tamaño de la fuente para la configuración de estilos de la GUI para asegurar una apariencia consistente.
nombre_fuente = "Arial" # Nombre de la fuente a utilizar en la GUI
tamano_fuente = 14  # Tamaño de fuente en puntos
