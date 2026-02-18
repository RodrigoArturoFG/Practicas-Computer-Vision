# Practica 1-a). "Creando mi mapa de calor"

import cv2
import matplotlib.pyplot as plt
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
print("Directorio del script:", script_dir) # Intenta cargar la imagen 

imagen_path = os.path.join(script_dir, 'rostro-humano.jpg')
print("Ruta de la imagen:", imagen_path)

imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE) 

if imagen_gris is None: 
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y extensión.")

# Aplicar diferentes mapas de color (pseudocolor) 1
imagen_jet = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_JET)
imagen_hot = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_HOT)
imagen_ocean = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_OCEAN)

# Aplicar diferentes mapas de color (pseudocolor) 2
imagen_autumn = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_AUTUMN)
imagen_bone = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_BONE)
imagen_cool = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_COOL)
imagen_hsv = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_HSV)

# Mostrar las imágenes en una cuadrícula para comparación visual 1
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(imagen_gris, cmap='gray')
axs[0, 0].set_title('Imagen en escala de grises')
axs[0, 1].imshow(cv2.cvtColor(imagen_jet, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Pseudocolor: JET')
axs[1, 0].imshow(cv2.cvtColor(imagen_hot, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Pseudocolor: HOT')
axs[1, 1].imshow(cv2.cvtColor(imagen_ocean, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Pseudocolor: OCEAN')

# Quitar los ejes para mejor visualización
for ax in axs.flat:
    ax.axis('off')

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()


# Mostrar las imágenes en una cuadrícula para comparación visual 2
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(imagen_gris, cmap='gray')
axs[0, 1].imshow(cv2.cvtColor(imagen_autumn, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Pseudocolor: AUTUMN')
axs[1, 0].imshow(cv2.cvtColor(imagen_bone, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Pseudocolor: BONE')
axs[1, 1].imshow(cv2.cvtColor(imagen_cool, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Pseudocolor: COOL')
axs[1, 1].imshow(cv2.cvtColor(imagen_hsv, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Pseudocolor: HSV')

# Quitar los ejes para mejor visualización
for ax in axs.flat:
    ax.axis('off')

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()