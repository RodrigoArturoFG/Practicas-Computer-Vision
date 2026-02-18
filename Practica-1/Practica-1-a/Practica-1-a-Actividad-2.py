import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Este ejemplo permite crear una imagen en escala de grises como gradiente horizontal (prueba usando otra imagen)
# Cada fila tiene valores de 0 a 255 en forma de gradiente
imagen_gris = np.tile(np.linspace(0, 255, 256), (100,1)).astype(np.uint8)

# Definir colores pastel en formato RGB normalizado (valores entre 0 y 1)
colores_pastel = [
(1.0, 0.8, 0.9), # rosa claro
(0.8, 1.0, 0.8), # verde menta
(0.8, 0.9, 1.0), # azul lavanda
(1.0, 1.0, 0.8), # amarillo suave
(0.9, 0.8, 1.0) # violeta claro
]

# Crear el mapa de color personalizado
mapa_pastel = LinearSegmentedColormap.from_list("PastelMap",
colores_pastel, N=256)

# Visualizar la imagen original y la imagen con pseudocolor pastel
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Imagen en escala de grises
axs[0].imshow(imagen_gris, cmap='gray')
axs[0].set_title('Imagen en escala de grises')
axs[0].axis('off')

# Imagen con mapa de color pastel
axs[1].imshow(imagen_gris, cmap=mapa_pastel)
axs[1].set_title('Imagen con un mapa de color personalizado tipo pastel')
axs[1].axis('off')
plt.tight_layout()
plt.show()