# Configuración de parámetros para la práctica de procesamiento de imágenes
import os
import cv2

# Obtener el directorio del script y regresar un nivel en la jerarquía de carpetas
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diccionario de mapas de color disponibles en OpenCV
# Obtener automáticamente todos los colormaps disponibles
mapas_color = {name.replace("COLORMAP_", ""): getattr(cv2, name) 
               for name in dir(cv2) if name.startswith("COLORMAP_")}