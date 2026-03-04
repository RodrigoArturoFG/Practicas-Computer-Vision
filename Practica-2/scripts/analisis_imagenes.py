# ----- PRACTICA 2 "EXPLORANDO LA IMAGEN DIGITAL CON PYTHON" -----
# ----------------------------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-03-2026

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import config
import image_manager

# 1. Lectura de imagen con Pillow
def lectura_imagen_pillow(ruta):
    imagen = image_manager.cargar_imagen_pillow(ruta)
    #plt.figure(figsize=(5, 5))
    plt.imshow(imagen)
    plt.title("Imagen original (Pillow)")
    plt.axis("off")
    plt.show()
    return imagen

# 2. Lectura de imagen con OpenCV
def lectura_imagen_opencv(ruta):
    imagen_cv_rgb = image_manager.cargar_imagen_opencv(ruta)
    #plt.figure(figsize=(5, 5))
    plt.imshow(imagen_cv_rgb)
    plt.title("Imagen con OpenCV (RGB)")
    plt.axis("off")
    plt.show()
    return imagen_cv_rgb

# 3. Funciones para separar canales RGB de una imagen y visualizarlos por separado
def separar_rgb(imagen):
    r, g, b = imagen.split()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds')
    plt.title("Componente R")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens')
    plt.title("Componente G")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues')
    plt.title("Componente B")
    plt.axis("off")
    plt.show()
    return r, g, b

# 4. Función para convertir imagen a escala de grises
def convertir_a_grises(imagen):
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    plt.imshow(imagen_cv, cmap='gray')
    plt.title("Imagen en escala de grises")
    plt.axis("off")
    plt.show()
    return imagen_cv

# 5. Función para binarizar la imagen en escala de grises utilizando un umbral fijo
def binarizar_imagen(imagen_gris, umbral=128):
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray')
    plt.title(f"Imagen binarizada (umbral = {umbral})")
    plt.axis("off")
    plt.show()
    return binaria

# 6. Función para convertir la imagen a hsv y mostrar los canales por separado
# Definir estructura, objeto, o diccionario según sea mas optimo que contenga: imagen_HSV, H (matiz), S (saturación), V (valor)
def convertir_a_hsv(imagen_cv):
    imagen_hsv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imagen_hsv)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(h, cmap="hsv")
    plt.title("Canal H")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(s, cmap="gray")
    plt.title("Canal S")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(v, cmap="gray")
    plt.title("Canal V")
    plt.axis("off")
    plt.suptitle("Modelo HSV")
    plt.show()
    return imagen_hsv

# 7. Función para la imagen a cmy y mostrar los canales por separado
# (CMY simulado, ya que OpenCV no lo soporta directamente)
def convertir_a_cmy(imagen_cv_rgb):
    imagen_cmy = 255 - imagen_cv_rgb
    c, m, y = cv2.split(imagen_cmy)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(c, cmap="Blues")
    plt.title("Canal C")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(m, cmap="Purples")
    plt.title("Canal M")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(y, cmap="Oranges")
    plt.title("Canal Y")
    plt.axis("off")
    plt.suptitle("Modelo CMY (simulado)")
    plt.show()
    return imagen_cmy

if __name__ == "__main__":
    imagen_pillow = lectura_imagen_pillow(os.path.join(config.script_dir_parent, 'resources\\input\\pentium4_microscopio.jpg'))
    separar_rgb(imagen_pillow)
    imagen_gris = convertir_a_grises(imagen_pillow)
    binarizar_imagen(imagen_gris, umbral=128)

    imagen_opencv = lectura_imagen_opencv(os.path.join(config.script_dir_parent, 'resources\\input\\pentium4_microscopio.jpg'))
    imagen_hsv = convertir_a_hsv(imagen_opencv)
    imagen_cmy = convertir_a_cmy(imagen_opencv)