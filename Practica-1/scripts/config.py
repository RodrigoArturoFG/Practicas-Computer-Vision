# --------- CONFIGURACIÓN DE PARÁMETROS PARA LA PRÁCTICA 1  ---------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-19-2026

import os
import cv2


# Obtener el directorio del script y regresar un nivel en la jerarquía de carpetas
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diccionario de mapas de color disponibles en OpenCV
# Obtener automáticamente todos los colormaps disponibles
mapas_color = {name.replace("COLORMAP_", ""): getattr(cv2, name) 
               for name in dir(cv2) if name.startswith("COLORMAP_")}

# Mapas de color personalizados
# Definir colores pastel en formato RGB normalizado (valores entre 0 y 1)
colores_pastel = [
    (1.0, 0.8, 0.9), # rosa claro
    (0.8, 1.0, 0.8), # verde menta
    (0.8, 0.9, 1.0), # azul lavanda
    (1.0, 1.0, 0.8), # amarillo suave
    (0.9, 0.8, 1.0) # violeta claro
]

# Definir colores tierra en formato RGB normalizado (valores entre 0 y 1)
colores_tierra = [
    (0.6, 0.4, 0.2), # marrón oscuro
    (0.8, 0.7, 0.5), # marrón claro
    (0.9, 0.8, 0.6), # beige
    (0.7, 0.5, 0.3), # marrón medio
    (0.5, 0.3, 0.1) # marrón muy oscuro
]

# Definir colores pastel personalizado en formato RGB normalizado (valores entre 0 y 1)
# Evitar tonos oscuros y mantiener la imagen luminosa.
# Para más color, bajar  los valores mínimos a 0.75–0.8, manteniendo la suavidad pastel pero con más contraste y color.
"""
# Primer version personalizada, más suave y clara, pero con menos contraste y color:
colores_pastel_personalizados = [
    (1.0, 0.95, 0.85),  # crema claro
    (0.95, 0.85, 1.0),  # lila pastel
    (0.85, 1.0, 0.95),  # verde menta claro
    (1.0, 1.0, 0.85),   # amarillo pastel
    (0.95, 0.95, 1.0),  # azul cielo pastel
    (1.0, 0.85, 0.95),  # rosa suave
    (0.9, 0.95, 1.0),   # celeste muy claro
]
"""

"""
# Más contraste y color, pero manteniendo la suavidad pastel:
colores_pastel_personalizados = [
    (1.0, 0.85, 0.95),  # rosa pastel
    (0.85, 1.0, 0.85),  # verde menta
    (0.8, 0.9, 1.0),    # azul lavanda
    (1.0, 1.0, 0.8),    # amarillo suave
    (0.95, 0.8, 1.0),   # lila pastel
    (0.9, 0.95, 0.8),   # crema pastel
    (0.95, 0.95, 0.85), # beige pastel
]
"""

# Más saturación y contraste, pero manteniendo la suavidad pastel:
colores_pastel_personalizados = [
    (1.0, 0.8, 0.9),   # rosa pastel
    (0.8, 1.0, 0.8),   # verde menta
    (0.75, 0.85, 1.0), # azul lavanda
    (1.0, 1.0, 0.8),   # amarillo suave
    (0.85, 0.75, 1.0), # lila pastel
    (0.9, 0.85, 0.8),  # crema pastel
    (0.85, 0.9, 0.75), # beige pastel
]
