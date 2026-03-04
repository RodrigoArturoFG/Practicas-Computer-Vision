# --------- GESTOR DE IMAGENES  ----------
# ----------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 03-03-2026

from PIL import Image
import cv2

# 1. Lectura de imagen con Pillow
def cargar_imagen_pillow(ruta):
    imagen = Image.open(ruta)
    return imagen

# 2. Lectura de imagen con OpenCV
def cargar_imagen_opencv(ruta):
    imagen_cv = cv2.imread(ruta)
    imagen_cv_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    return imagen_cv_rgb
