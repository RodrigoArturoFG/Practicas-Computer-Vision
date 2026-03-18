# --------- PROCESADOR DE IMAGENES  ----------
# --------------------------------------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 03-03-2026

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import skew
from math import log2

# Diccionario que contiene tanto el objeto (metadatos imagen) como el estado del error.
def wrapper_respuesta(imagen_metadata, exito=True, mensaje="Operación exitosa"):
    return {
        "objeto": imagen_metadata,  # Aquí va la referencia del modelo imagen
        "error": not exito,         # Booleano: True si falló
        "mensaje": mensaje          # Descripción del resultado
    }

# Mapeamos los nombres de matplotlib a colores sólidos válidos para líneas
diccionario_colores = {
    "Reds": "red",
    "Greens": "green",
    "Blues": "blue",
    "hsv": "magenta",
    "gray": "gray",
    "Purples": "purple",
    "YlOrBr": "orange"
}

# Definimos los modelos que se deben ver en blanco y negro
# Agregamos "BINARIO" porque también es de un solo canal
modelos_monocromaticos = ["GRIS", "BINARIO"]

# Carga de una imagen RGB con OpenCV
def cargar_imagen_opencv_rgb(imagen_metadata):
    """Lectura de imagen RGB con OpenCV"""
    imagen_cv = cv2.imread(imagen_metadata.ruta)
    if imagen_cv is None:
        return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
    
    # Hacemos conversión de BGR a RGB antes de devolver la imagen
    imagen_metadata.datos = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
    imagen_metadata.modelo = "RGB"
    return wrapper_respuesta(imagen_metadata)

# Carga de imagen en escala de grises con OpenCV
def cargar_imagen_opencv_gris(imagen_metadata):
    """Lectura de imagen con OpenCV en grises."""
    # 1. Validación de canales: si la imagen ya tiene un solo canal (Gris/Binaria)
    if es_modelo_monocromatico(imagen_metadata.modelo):        
        # La imagen ya se encuentra en escala de grises. Omitir conversión.
        return wrapper_respuesta(imagen_metadata, False, "Parece que la imagen ya se encuentra en escala de grises. Omitiendo conversión para evitar errores.")

    imagen_cv = cv2.imread(imagen_metadata.ruta)
    if imagen_cv is None:
        return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
    
    # 2. Hacemos conversión de BGR a grises antes de devolver la imagen
    imagen_metadata.datos = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    imagen_metadata.modelo = "GRIS"
    return wrapper_respuesta(imagen_metadata)

# Binarización con umbral dínamico
def conversion_imagen_opencv_binaria(imagen_metadata, umbral=128):
    """Binarización de imagen con umbral dínamico"""
    # 1. Validación de canales: Si es color, la convertimos a gris primero
    if es_modelo_monocromatico(imagen_metadata.modelo) == False:
        respuesta = cargar_imagen_opencv_gris(imagen_metadata)
        # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
        imagen_metadata = respuesta["objeto"]

        if respuesta["error"]:
            return respuesta
    # 2. Validación de contenido: ¿Ya es binaria?        
    elif es_binaria(imagen_metadata):
        # La imagen ya se encuentra binarizada. Se omitirá el proceso para evitar pérdida de datos.
        # [Si se elige un umbral por debajo del valor mínimo o por encima del máximo (por ejemplo, un umbral de 256), 
        # se perderá toda la información de la imagen.]
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya es binaria. Omitiendo conversión para evitar pérdida de datos.")

    imagen_metadata.umbral, imagen_metadata.datos = cv2.threshold(imagen_metadata.datos, umbral, 255, cv2.THRESH_BINARY)
    imagen_metadata.modelo = "BINARIO"
    return wrapper_respuesta(imagen_metadata)


# Binarización con umbral de Otsu
def conversion_imagen_opencv_otsu(imagen_metadata):
    """
    Aplica el algoritmo de Otsu para encontrar el umbral óptimo automáticamente.
    Actualiza el umbral real calculado en los metadatos.
    """
    # 1. Validación de canales: Otsu requiere forzosamente una imagen de un solo canal (Gris)
    #    Si es color, la convertimos a gris primero:
    if es_modelo_monocromatico(imagen_metadata.modelo) == False:
        respuesta = cargar_imagen_opencv_gris(imagen_metadata)
        # Actualizamos nuestra referencia con el objeto que viene dentro de la respuesta
        imagen_metadata = respuesta["objeto"]

        if respuesta["error"]:
            return respuesta
    elif es_binaria(imagen_metadata):
        # La imagen ya se encuentra binarizada. Se omitirá el proceso para evitar pérdida de datos.
        # [Si se elige un umbral por debajo del valor mínimo o por encima del máximo (por ejemplo, un umbral de 256), 
        # se perderá toda la información de la imagen.]
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya es binaria. Omitiendo conversión para evitar pérdida de datos.")
    
    # Se aplica cv2.THRESH_OTSU sumado al binario normal
    # 'umbral_calculado' recibirá el valor óptimo que Otsu encontró 
    imagen_metadata.umbral, imagen_metadata.datos = cv2.threshold(imagen_metadata.datos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagen_metadata.modelo = "BINARIO"
    return wrapper_respuesta(imagen_metadata)

# Cambiar modelo de color a HSV
def conversion_imagen_opencv_hsv(imagen_metadata):
    """
    Convierte de RGB a HSV con validación de entrada.
    """
    # 1. Validación: Si la imagen ya es HSV (tomando en cuenta metadatos de la imagen)
    if es_HSV(imagen_metadata):
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya esta en un modelo HSV. Omitiendo conversión para evitar errores.")
    # 2. Validación: Si la imagen ya es Gris/Binaria (2D)
    elif es_modelo_monocromatico(imagen_metadata.modelo):
        # Covertir imagen Gris/Binaria a RGB 
        imagen_cv = cv2.imread(imagen_metadata.ruta)
        if imagen_cv is None:
            return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
        imagen_metadata.datos = imagen_cv

    # 3. Conversión segura
    imagen_metadata.datos = cv2.cvtColor(imagen_metadata.datos, cv2.COLOR_RGB2HSV)
    imagen_metadata.modelo = "HSV"
    
    return wrapper_respuesta(imagen_metadata)

# Cambiar modelo de color a CMY
def conversion_imagen_opencv_cmy(imagen_metadata):
    """
    Simula el modelo CMY con protección contra re-inversión.
    """
    # 1. Validación: Si la imagen ya es CMY
    if es_CMY(imagen_metadata):
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya esta en un modelo CMY. Omitiendo conversión para evitar errores.")
    # 2. Validación: Si la imagen esta en otro modelo que no es RGB ni CMY
    elif es_RGB(imagen_metadata) == False:
        # Covertir imagen Gris/Binaria a RGB 
        imagen_cv = cv2.imread(imagen_metadata.ruta)
        if imagen_cv is None:
            return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
        imagen_metadata.datos = imagen_cv


    # 3. Conversión segura
    print("Aplicando modelo CMY (Inversión de canales RGB)...")
    imagen_metadata.datos = 255 - imagen_metadata.datos
    imagen_metadata.modelo = "CMY"
    
    return wrapper_respuesta(imagen_metadata)

def conversion_imagen_opencv_yiq(imagen_metadata):
    """Convierte de RGB a YIQ usando la matriz de transformación NTSC."""
    # 1. Validación: Si la imagen ya es YIQ
    if es_YIQ(imagen_metadata):
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya esta en un modelo YIQ. Omitiendo conversión para evitar errores.")
    # 2. Validación: Si la imagen esta en otro modelo que no es RGB ni YIQ
    elif es_RGB(imagen_metadata) == False:
        # Covertir imagen Gris/Binaria a RGB 
        imagen_cv = cv2.imread(imagen_metadata.ruta)
        if imagen_cv is None:
            return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
        imagen_metadata.datos = imagen_cv

    try:
        # 1. Convertir a float 0-1 para el cálculo
        img_float = imagen_metadata.datos.astype(np.float32) / 255.0
        r, g, b = cv2.split(img_float)

        # 2. Fórmulas NTSC
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b

        # 3. NORMALIZACIÓN para evitar el "Clipping error"
        # Escalamos I y Q de su rango teórico [-0.6, 0.6] a [0, 1]
        # Esto permite que Matplotlib y OpenCV los manejen sin problemas
        i_norm = (i + 0.5957) / 1.1914
        q_norm = (q + 0.5226) / 1.0452
        
        # Aseguramos que los valores estén estrictamente entre 0 y 1
        y = np.clip(y, 0, 1)
        i_norm = np.clip(i_norm, 0, 1)
        q_norm = np.clip(q_norm, 0, 1)

        # 4. Guardar como float32 (Matplotlib lo entiende perfecto si es 0-1)
        imagen_metadata.datos = cv2.merge([y, i_norm, q_norm])
        imagen_metadata.modelo = "YIQ"
        
        
        return wrapper_respuesta(imagen_metadata)
        
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en YIQ: {str(e)}")

def conversion_imagen_opencv_hsi(imagen_metadata):
    """Convierte de RGB a HSI usando la matriz de transformación NTSC."""
    # 1. Validación: Si la imagen ya es HSI
    if es_YIQ(imagen_metadata):
        return wrapper_respuesta(imagen_metadata, False, "La imagen ya esta en un modelo HSI. Omitiendo conversión para evitar errores.")
    elif es_RGB(imagen_metadata) == False:
        # Covertir imagen Gris/Binaria a RGB 
        imagen_cv = cv2.imread(imagen_metadata.ruta)
        if imagen_cv is None:
            return wrapper_respuesta(imagen_metadata, False, f"No se pudo cargar la imagen en: {imagen_metadata.ruta}")
        imagen_metadata.datos = imagen_cv

    try:
        # 1. Normalizar RGB a [0, 1] para cálculos precisos
        img_float = imagen_metadata.datos.astype(np.float32) / 255.0
        r, g, b = cv2.split(img_float)

        # 2. Intensidad (Promedio aritmético)
        intensity = (r + g + b) / 3.0

        # 3. Saturación
        min_rgb = np.minimum(np.minimum(r, g), b)
        # Evitar división por cero si la intensidad es 0 (negro)
        denominador_s = (r + g + b + 1e-6)
        saturation = 1 - (3 / denominador_s * min_rgb)

        # 4. Matiz (Hue) - Algoritmo trigonométrico
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b) * (g - b)) + 1e-6
        theta = np.arccos(np.clip(num / den, -1, 1)) # Clip para evitar errores de precisión en arccos

        hue = theta
        hue[b > g] = 2 * np.pi - hue[b > g] # Ajustar el círculo cromático
        
        # NORMALIZACIÓN CRÍTICA:
        # Convertimos Hue de [0, 2pi] a [0, 1] para que Matplotlib lo entienda
        hue_norm = hue / (2 * np.pi)

        # 5. Empaquetar y asegurar rango [0, 1] para evitar Clipping
        hsi_final = cv2.merge([
            np.clip(hue_norm, 0, 1), 
            np.clip(saturation, 0, 1), 
            np.clip(intensity, 0, 1)
        ])

        imagen_metadata.datos = hsi_final
        imagen_metadata.modelo = "HSI"

        return wrapper_respuesta(imagen_metadata)

    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en HSI: {str(e)}")

def proceso_histograma_completo(imagen_metadata):
    """
    Gestiona el cálculo del histograma, adaptándose dinámicamente al modelo de color actual del objeto.
    """
    # 1. Obtener la configuración visual del modelo actual (RGB, HSV, CMY, etc.)
    config = obtener_config_modelo(imagen_metadata.modelo)    
    nombres = config["nombres"]

    # 2. SEPARACIÓN DE LÓGICA: Calcular primero
    try:
        dict_stats = calcular_estadisticas_canales(imagen_metadata.datos, nombres)
        
        # 3. Guardar en el objeto (Metadatos persistentes)
        imagen_metadata.histograma = dict_stats
        return wrapper_respuesta(imagen_metadata, True, "Histogramas generados y guardados.")
        
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error al calcular histograma: {str(e)}")

def calcular_estadisticas_canales(datos_imagen, nombres_canales):
    """
    Realiza los cálculos estadísticos de forma pura.
    Devuelve un diccionario con los resultados.
    """
    resultados = {}
    
    # Si la imagen es gris, la convertimos a una lista de un solo canal para el loop
    canales = [datos_imagen] if len(datos_imagen.shape) == 2 else cv2.split(datos_imagen)
    
    for i, datos_canal in enumerate(canales):
        nombre = nombres_canales[i]
        datos_planos = datos_canal.flatten()
        
        # Histograma y Probabilidad
        hist, _ = np.histogram(datos_planos, bins=256, range=(0, 256))
        prob = hist / hist.sum()
        
        # Propiedades (Energía, Entropía, Asimetría, Media, Varianza)
        energia = np.sum(prob ** 2)
        entropia = -np.sum([p * log2(p) for p in prob if p > 0])
        asimetria = skew(datos_planos)
        media = np.mean(datos_planos)
        varianza = np.var(datos_planos)
        
        resultados[nombre] = {
            'Energía': energia, 'Entropía': entropia, 
            'Asimetría': asimetria, 'Media': media, 
            'Varianza': varianza, 'histograma_raw': hist # Guardamos el hist para graficar después
        }
    return resultados

# ══════════════════════════════════════════════════════════════
#  RUIDO
# ══════════════════════════════════════════════════════════════

def _preparar_imagen_binaria(imagen_metadata):
    """
    Helper centralizado: garantiza que imagen_metadata esté en modelo BINARIO
    limpio antes de cualquier operación que lo requiera (ruido, vecindad, lógicas).

    Lógica canónica:
      - Ya es BINARIO → recarga desde disco y re-aplica imagen_metadata.umbral,
        que puede ser un valor elegido por el usuario desde la UI. De este modo
        el umbral que el usuario seleccionó se respeta en todas las operaciones
        posteriores y no se sobreescribe con uno automático.
      - Cualquier otro modelo → binariza con Otsu (umbral óptimo automático)
        y guarda el valor calculado en imagen_metadata.umbral.

    En ambos casos la imagen se recarga desde disco para descartar cualquier
    ruido o modificación acumulada en memoria.

    Retorna: wrapper_respuesta con imagen_metadata en modelo BINARIO.
    """
    if es_binaria(imagen_metadata):
        # Respetar el umbral elegido por el usuario (o 128 si nunca se definió)
        umbral_previo = imagen_metadata.umbral if imagen_metadata.umbral is not None else 128

        # Reset temporal a "RGB" para que cargar_imagen_opencv_gris no rechace
        # la imagen por ya ser monocromática (GRIS o BINARIO).
        imagen_metadata.modelo = "RGB"
        respuesta_gris = cargar_imagen_opencv_gris(imagen_metadata)
        imagen_metadata = respuesta_gris["objeto"]
        if respuesta_gris["error"]:
            return respuesta_gris

        respuesta_bin = conversion_imagen_opencv_binaria(imagen_metadata, umbral_previo)
        imagen_metadata = respuesta_bin["objeto"]
        if respuesta_bin["error"]:
            return respuesta_bin

        return wrapper_respuesta(
            imagen_metadata, True,
            f"Imagen binaria recargada limpia (umbral={umbral_previo})"
        )
    else:
        # Primera vez que se binariza: Otsu calcula el umbral óptimo y lo guarda
        respuesta_otsu = conversion_imagen_opencv_otsu(imagen_metadata)
        imagen_metadata = respuesta_otsu["objeto"]
        if respuesta_otsu["error"]:
            return respuesta_otsu

        return wrapper_respuesta(
            imagen_metadata, True,
            f"Imagen binarizada con Otsu (umbral={imagen_metadata.umbral})"
        )


def _preparar_imagen_gris(imagen_metadata):
    """
    Garantiza que imagen_metadata tenga datos en escala de grises limpios
    (sin ruido acumulado) antes de aplicar ruido gaussiano.

    Casos:
      - Ya es GRIS   : recarga desde disco para descartar ruido previo.
      - Es BINARIO   : recarga desde disco como gris (la naturaleza binaria
                       se pierde al aplicar gaussiano, esto es esperado).
      - Cualquier otro modelo (RGB, HSV, etc.): convierte a gris directamente.

    Retorna: wrapper_respuesta con imagen_metadata en modelo GRIS.
    """
    if es_modelo_monocromatico(imagen_metadata.modelo):
        # Resetear modelo a "RGB" para que cargar_imagen_opencv_gris
        # no rechace la imagen por ya ser monocromática.
        imagen_metadata.modelo = "RGB"

    respuesta_gris = cargar_imagen_opencv_gris(imagen_metadata)
    imagen_metadata = respuesta_gris["objeto"]
    if respuesta_gris["error"]:
        return respuesta_gris

    return wrapper_respuesta(imagen_metadata, True, "Imagen preparada en escala de grises.")


def agregar_ruido_gaussiano(imagen_metadata, media=0, sigma=20):
    """
    Agrega ruido gaussiano a una imagen en escala de grises.
    Si la imagen es color o binaria, la convierte a grises automáticamente.
    Si ya era gris o binaria, la recarga desde disco para no acumular ruido previo.
    El modelo resultante siempre es GRIS (el ruido gaussiano destruye la naturaleza binaria).
    Persiste el resultado en imagen_metadata.datos.
    Retorna: wrapper_respuesta con imagen_metadata actualizado.
    """
    # 1. Validar que haya datos cargados
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")

    # 2. Garantizar imagen en escala de grises limpia (sin ruido acumulado)
    respuesta = _preparar_imagen_gris(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        return respuesta

    # 3. Aplicar ruido gaussiano
    try:
        # Generar ruido con distribución normal y sumarlo a la imagen
        # int16 para evitar overflow durante la suma antes del clip
        gauss = np.random.normal(media, sigma, imagen_metadata.datos.shape).astype(np.int16)
        imagen_ruido = imagen_metadata.datos.astype(np.int16) + gauss
        imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)

        # 4. Persistir resultado (modelo permanece GRIS)
        imagen_metadata.datos = imagen_ruido

        return wrapper_respuesta(
            imagen_metadata, True,
            f"Ruido gaussiano aplicado (media={media}, sigma={sigma})"
        )
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error al aplicar ruido gaussiano: {str(e)}")


def agregar_ruido_sal(imagen_metadata, cantidad=0.02):
    """
    Agrega ruido SAL (píxeles en 255) sobre una imagen binaria limpia.
    Si la imagen no es binaria, aplica Otsu automáticamente.
    Si ya era binaria, la recarga desde disco para no acumular ruido previo.
    Persiste el resultado en imagen_metadata.datos.
    Retorna: wrapper_respuesta con imagen_metadata actualizado.
    """
    # 1. Validar que haya datos cargados
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")

    # 2. Garantizar imagen binaria limpia (sin ruido acumulado)
    respuesta = _preparar_imagen_binaria(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        return respuesta

    # 3. Aplicar ruido SAL (solo píxeles blancos = 255)
    try:
        imagen_ruido = imagen_metadata.datos.copy()
        filas, columnas = imagen_ruido.shape       # Seguro: imagen ya es 2D (BINARIO)
        num_pixeles_ruido = int(cantidad * filas * columnas)

        filas_rand = np.random.randint(0, filas,    num_pixeles_ruido)
        cols_rand  = np.random.randint(0, columnas, num_pixeles_ruido)
        imagen_ruido[filas_rand, cols_rand] = 255  # SAL → blanco

        # 4. Persistir resultado
        imagen_metadata.datos = imagen_ruido

        return wrapper_respuesta(
            imagen_metadata, True,
            f"Ruido SAL aplicado — {num_pixeles_ruido} píxeles afectados ({cantidad*100:.1f}%)"
        )
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error al aplicar ruido SAL: {str(e)}")


def agregar_ruido_pimienta(imagen_metadata, cantidad=0.02):
    """
    Agrega ruido PIMIENTA (píxeles en 0) sobre una imagen binaria limpia.
    Si la imagen no es binaria, aplica Otsu automáticamente.
    Si ya era binaria, la recarga desde disco para no acumular ruido previo.
    Persiste el resultado en imagen_metadata.datos.
    Retorna: wrapper_respuesta con imagen_metadata actualizado.
    """
    # 1. Validar que haya datos cargados
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")

    # 2. Garantizar imagen binaria limpia (sin ruido acumulado)
    respuesta = _preparar_imagen_binaria(imagen_metadata)
    imagen_metadata = respuesta["objeto"]
    if respuesta["error"]:
        return respuesta

    # 3. Aplicar ruido PIMIENTA (solo píxeles negros = 0)
    try:
        imagen_ruido = imagen_metadata.datos.copy()
        filas, columnas = imagen_ruido.shape       # Seguro: imagen ya es 2D (BINARIO)
        num_pixeles_ruido = int(cantidad * filas * columnas)

        filas_rand = np.random.randint(0, filas,    num_pixeles_ruido)
        cols_rand  = np.random.randint(0, columnas, num_pixeles_ruido)
        imagen_ruido[filas_rand, cols_rand] = 0    # PIMIENTA → negro

        # 4. Persistir resultado
        imagen_metadata.datos = imagen_ruido

        return wrapper_respuesta(
            imagen_metadata, True,
            f"Ruido PIMIENTA aplicado — {num_pixeles_ruido} píxeles afectados ({cantidad*100:.1f}%)"
        )
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error al aplicar ruido PIMIENTA: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  OPERACIONES ARITMÉTICAS
# ══════════════════════════════════════════════════════════════

def _preparar_par_imagenes(imagen_a, imagen_b):
    """
    Normaliza el par de imágenes para que sean compatibles antes de
    cualquier operación aritmética o lógica entre dos imágenes:
      1. Convierte imagen_b al mismo modelo de color que imagen_a.
      2. Redimensiona imagen_b al mismo tamaño que imagen_a.

    No modifica imagen_a. Devuelve los datos de imagen_b ya ajustados
    como un array NumPy listo para operar.

    Retorna: (datos_b_ajustados, error:bool, mensaje:str)
    """
    if imagen_a.datos is None or imagen_b.datos is None:
        return None, True, "Ambas imágenes deben estar cargadas."

    datos_b = imagen_b.datos.copy()

    # 1. Igualar número de canales (modelo de color)
    canales_a = 1 if len(imagen_a.datos.shape) == 2 else imagen_a.datos.shape[2]
    canales_b = 1 if len(datos_b.shape) == 2        else datos_b.shape[2]

    if canales_a != canales_b:
        if canales_a == 1:
            # A es gris/binaria → convertir B a gris
            datos_b = cv2.cvtColor(datos_b, cv2.COLOR_BGR2GRAY) if canales_b == 3 else datos_b
        else:
            # A es color → convertir B a color (gris → BGR → igualar canales)
            if canales_b == 1:
                datos_b = cv2.cvtColor(datos_b, cv2.COLOR_GRAY2BGR)

    # 2. Igualar tamaño (redimensionar B al tamaño de A)
    alto_a, ancho_a = imagen_a.datos.shape[:2]
    alto_b, ancho_b = datos_b.shape[:2]
    if (alto_a, ancho_a) != (alto_b, ancho_b):
        datos_b = cv2.resize(datos_b, (ancho_a, alto_a), interpolation=cv2.INTER_LINEAR)

    return datos_b, False, "Par de imágenes preparado correctamente."


def sumar_imagenes(imagen_a, imagen_b):
    """
    Suma imagen_a e imagen_b píxel a píxel con saturación en 255.
    El resultado se guarda en imagen_a.datos.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_imagenes(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        imagen_a.datos = cv2.add(imagen_a.datos, datos_b)
        return wrapper_respuesta(imagen_a, True, f"Suma aplicada: [{imagen_a.nombre}] + [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en suma: {str(e)}")


def restar_imagenes(imagen_a, imagen_b):
    """
    Resta imagen_b a imagen_a píxel a píxel con saturación en 0.
    El resultado se guarda en imagen_a.datos.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_imagenes(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        imagen_a.datos = cv2.subtract(imagen_a.datos, datos_b)
        return wrapper_respuesta(imagen_a, True, f"Resta aplicada: [{imagen_a.nombre}] - [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en resta: {str(e)}")


def multiplicar_imagenes(imagen_a, imagen_b):
    """
    Multiplica imagen_a e imagen_b píxel a píxel.
    Normaliza dividiendo entre 255 para que el resultado permanezca en [0, 255]
    (sin normalización, el producto de dos uint8 se satura a 255 en casi todos los píxeles).
    El resultado se guarda en imagen_a.datos.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_imagenes(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        # Normalizar a [0,1] → multiplicar → escalar de vuelta a [0,255]
        a_norm = imagen_a.datos.astype(np.float32) / 255.0
        b_norm = datos_b.astype(np.float32)        / 255.0
        resultado = np.clip(a_norm * b_norm * 255.0, 0, 255).astype(np.uint8)
        imagen_a.datos = resultado
        return wrapper_respuesta(imagen_a, True, f"Multiplicación aplicada: [{imagen_a.nombre}] × [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en multiplicación: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  OPERACIONES LÓGICAS Y RELACIONALES
# ══════════════════════════════════════════════════════════════

def _preparar_par_logico(imagen_a, imagen_b):
    """
    Prepara el par de imágenes para operaciones lógicas (AND, OR, XOR):
      1. Binariza imagen_a usando _preparar_imagen_binaria (respeta umbral del usuario).
      2. Binariza imagen_b usando _preparar_imagen_binaria (respeta su umbral si lo tiene).
      3. Iguala tamaño y canales de B al de A usando _preparar_par_imagenes.

    Retorna: (datos_b_ajustados, error:bool, mensaje:str)
    """
    if imagen_a.datos is None or imagen_b.datos is None:
        return None, True, "Ambas imágenes deben estar cargadas."

    # Binarizar A respetando umbral del usuario (helper canónico)
    respuesta_a = _preparar_imagen_binaria(imagen_a)
    imagen_a.datos  = respuesta_a["objeto"].datos
    imagen_a.modelo = respuesta_a["objeto"].modelo
    imagen_a.umbral = respuesta_a["objeto"].umbral
    if respuesta_a["error"]:
        return None, True, f"No se pudo binarizar imagen A: {respuesta_a['mensaje']}"

    # Binarizar B temporalmente con el mismo helper (respeta su umbral si lo tiene)
    respuesta_b = _preparar_imagen_binaria(imagen_b)
    if respuesta_b["error"]:
        return None, True, f"No se pudo binarizar imagen B: {respuesta_b['mensaje']}"
    imagen_b_temp = respuesta_b["objeto"]

    # Igualar tamaño y canales de B al de A
    datos_b, error, mensaje = _preparar_par_imagenes(imagen_a, imagen_b_temp)
    return datos_b, error, mensaje


def and_imagenes(imagen_a, imagen_b):
    """
    AND bit a bit entre imagen_a e imagen_b binarias.
    Resultado: píxel blanco solo donde AMBAS imágenes tienen píxel blanco.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_logico(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        imagen_a.datos = cv2.bitwise_and(imagen_a.datos, datos_b)
        return wrapper_respuesta(imagen_a, True, f"AND aplicado: [{imagen_a.nombre}] AND [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en AND: {str(e)}")


def or_imagenes(imagen_a, imagen_b):
    """
    OR bit a bit entre imagen_a e imagen_b binarias.
    Resultado: píxel blanco donde AL MENOS UNA imagen tiene píxel blanco.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_logico(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        imagen_a.datos = cv2.bitwise_or(imagen_a.datos, datos_b)
        return wrapper_respuesta(imagen_a, True, f"OR aplicado: [{imagen_a.nombre}] OR [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en OR: {str(e)}")


def xor_imagenes(imagen_a, imagen_b):
    """
    XOR bit a bit entre imagen_a e imagen_b binarias.
    Resultado: píxel blanco donde las imágenes son DIFERENTES entre sí.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    datos_b, error, mensaje = _preparar_par_logico(imagen_a, imagen_b)
    if error:
        return wrapper_respuesta(imagen_a, False, mensaje)
    try:
        imagen_a.datos = cv2.bitwise_xor(imagen_a.datos, datos_b)
        return wrapper_respuesta(imagen_a, True, f"XOR aplicado: [{imagen_a.nombre}] XOR [{imagen_b.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en XOR: {str(e)}")


def not_imagen(imagen_a):
    """
    NOT bit a bit sobre imagen_a (inversión de píxeles).
    Binariza con Otsu si no es binaria.
    Resultado: fondo y objetos se intercambian.
    Retorna: wrapper_respuesta con imagen_a actualizada.
    """
    if imagen_a.datos is None:
        return wrapper_respuesta(imagen_a, False, "No hay imagen cargada.")
    try:
        # Binarizar si no lo es
        if not es_binaria(imagen_a):
            respuesta = conversion_imagen_opencv_otsu(imagen_a)
            imagen_a = respuesta["objeto"]
            if respuesta["error"]:
                return respuesta
        imagen_a.datos = cv2.bitwise_not(imagen_a.datos)
        return wrapper_respuesta(imagen_a, True, f"NOT aplicado: [{imagen_a.nombre}]")
    except Exception as e:
        return wrapper_respuesta(imagen_a, False, f"Error en NOT: {str(e)}")


# --- Operaciones relacionales ---
# Comparan la intensidad de cada píxel contra un umbral escalar.
# La imagen de entrada debe ser en escala de grises (convierte automáticamente).
# El resultado es siempre una máscara binaria útil para segmentación.

def _preparar_imagen_para_relacional(imagen_metadata):
    """
    Garantiza que la imagen esté en escala de grises para las operaciones relacionales.
    Si es BINARIO o GRIS recarga desde disco limpio; si es color convierte a gris.
    Retorna: wrapper_respuesta con imagen en modelo GRIS.
    """
    if es_modelo_monocromatico(imagen_metadata.modelo):
        imagen_metadata.modelo = "RGB"   # reset para saltar la guardia de cargar_gris
    return cargar_imagen_opencv_gris(imagen_metadata)


def relacional_mayor(imagen_metadata, umbral):
    """
    Segmenta los píxeles cuya intensidad es MAYOR que el umbral.
    Resultado: máscara binaria — blanco donde píxel > umbral, negro en el resto.
    Retorna: wrapper_respuesta con imagen_metadata actualizada (modelo BINARIO).
    """
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")
    try:
        respuesta = _preparar_imagen_para_relacional(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]:
            return respuesta

        mascara = (imagen_metadata.datos > umbral).astype(np.uint8) * 255
        imagen_metadata.datos  = mascara
        imagen_metadata.modelo = "BINARIO"
        imagen_metadata.umbral = umbral
        return wrapper_respuesta(imagen_metadata, True, f"Relacional '>' aplicado (umbral={umbral})")
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en relacional '>': {str(e)}")


def relacional_menor(imagen_metadata, umbral):
    """
    Segmenta los píxeles cuya intensidad es MENOR que el umbral.
    Resultado: máscara binaria — blanco donde píxel < umbral, negro en el resto.
    Retorna: wrapper_respuesta con imagen_metadata actualizada (modelo BINARIO).
    """
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")
    try:
        respuesta = _preparar_imagen_para_relacional(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]:
            return respuesta

        mascara = (imagen_metadata.datos < umbral).astype(np.uint8) * 255
        imagen_metadata.datos  = mascara
        imagen_metadata.modelo = "BINARIO"
        imagen_metadata.umbral = umbral
        return wrapper_respuesta(imagen_metadata, True, f"Relacional '<' aplicado (umbral={umbral})")
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en relacional '<': {str(e)}")


def relacional_igual(imagen_metadata, umbral):
    """
    Segmenta los píxeles cuya intensidad es IGUAL al umbral.
    Resultado: máscara binaria — blanco donde píxel == umbral, negro en el resto.
    Útil para aislar un nivel de intensidad exacto.
    Retorna: wrapper_respuesta con imagen_metadata actualizada (modelo BINARIO).
    """
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")
    try:
        respuesta = _preparar_imagen_para_relacional(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]:
            return respuesta

        mascara = (imagen_metadata.datos == umbral).astype(np.uint8) * 255
        imagen_metadata.datos  = mascara
        imagen_metadata.modelo = "BINARIO"
        imagen_metadata.umbral = umbral
        return wrapper_respuesta(imagen_metadata, True, f"Relacional '==' aplicado (umbral={umbral})")
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en relacional '==': {str(e)}")


# ══════════════════════════════════════════════════════════════
#  ETIQUETADO DE COMPONENTES CONEXAS (VECINDAD)
# ══════════════════════════════════════════════════════════════

def _etiquetar_y_dibujar(imagen_binaria, connectivity):
    """
    Aplica connectedComponents y dibuja contornos numerados sobre una copia en color.
    Retorna: (num_objetos, labels, imagen_contornos_BGR)
    """
    num_labels, labels = cv2.connectedComponents(imagen_binaria, connectivity=connectivity)
    num_objetos = num_labels - 1  # excluir el fondo (etiqueta 0)

    imagen_contornos = cv2.cvtColor(imagen_binaria, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(imagen_contornos, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            imagen_contornos, f'Obj {i + 1}',
            (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    return num_objetos, labels, imagen_contornos


def analizar_vecindad_4(imagen_metadata):
    """
    Etiqueta componentes conexas usando vecindad-4 (solo conexiones ortogonales).
    Binariza la imagen automáticamente si no lo es.
    Retorna wrapper_respuesta con claves adicionales:
      - 'num_objetos': int
      - 'labels':     array NumPy con etiquetas por píxel
      - 'imagen_contornos': array BGR con contornos y números dibujados
    """
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")
    try:
        respuesta = _preparar_imagen_binaria(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]:
            return respuesta

        num_objetos, labels, imagen_contornos = _etiquetar_y_dibujar(
            imagen_metadata.datos, connectivity=4
        )
        resultado = wrapper_respuesta(
            imagen_metadata, True,
            f"Vecindad-4: {num_objetos} objeto(s) detectado(s) en [{imagen_metadata.nombre}]"
        )
        resultado["num_objetos"]      = num_objetos
        resultado["labels"]           = labels
        resultado["imagen_contornos"] = imagen_contornos
        return resultado
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en vecindad-4: {str(e)}")


def analizar_vecindad_8(imagen_metadata):
    """
    Etiqueta componentes conexas usando vecindad-8 (conexiones ortogonales + diagonales).
    Detecta más conexiones que vecindad-4, útil para objetos con bordes diagonales.
    Binariza la imagen automáticamente si no lo es.
    Retorna wrapper_respuesta con claves adicionales:
      - 'num_objetos': int
      - 'labels':     array NumPy con etiquetas por píxel
      - 'imagen_contornos': array BGR con contornos y números dibujados
    """
    if imagen_metadata.datos is None:
        return wrapper_respuesta(imagen_metadata, False, "No hay imagen cargada.")
    try:
        respuesta = _preparar_imagen_binaria(imagen_metadata)
        imagen_metadata = respuesta["objeto"]
        if respuesta["error"]:
            return respuesta

        num_objetos, labels, imagen_contornos = _etiquetar_y_dibujar(
            imagen_metadata.datos, connectivity=8
        )
        resultado = wrapper_respuesta(
            imagen_metadata, True,
            f"Vecindad-8: {num_objetos} objeto(s) detectado(s) en [{imagen_metadata.nombre}]"
        )
        resultado["num_objetos"]      = num_objetos
        resultado["labels"]           = labels
        resultado["imagen_contornos"] = imagen_contornos
        return resultado
    except Exception as e:
        return wrapper_respuesta(imagen_metadata, False, f"Error en vecindad-8: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  HELPERS — VALIDACIÓN DE MODELO
# ══════════════════════════════════════════════════════════════

# Verifica si la imagen tiene un modelo monoromatico
def es_modelo_monocromatico(tipo_modelo):
    if tipo_modelo in modelos_monocromaticos:
        return True
    else:
        return False

# Verifica si una imagen ya es binaria
def es_binaria(imagen_metadata):
    """
        imagen = imagen_metadata.datos
        valores_unicos = np.unique(imagen)
    if len(valores_unicos) <= 2:
        # La imagen ya se encuentra binarizada. 
        return True
    else:
        return False
    """
    if imagen_metadata.modelo == "BINARIO":
        return True
    else:
        return False  # FIXED: faltaba 'return'

def es_RGB(imagen_metadata):
    if imagen_metadata.modelo == "RGB":
        return True
    else:
        return False

def es_HSV(imagen_metadata):
    """
    Validación: Si la imagen ya es HSV (Basado en el rango del canal H: 0-179)
    if imagen_metadata.datos[:, :, 0].max() <= 180:
        return True
    else
        return False
    """
    if imagen_metadata.modelo == "HSV":
        return True
    else:
        return False

def es_CMY(imagen_metadata):
    """
    Nota sobre CMY: No hay forma de saber si una matriz es CMY solo por sus valores,
    por lo que únicamente podemos deducir esto por los metadatos de la imágen.
    """
    if imagen_metadata.modelo == "CMY":
        return True
    else:
        return False

def es_YIQ(imagen_metadata):
    """
    Nota sobre YIQ: No hay forma de saber si una matriz es YIQ solo por sus valores,
    por lo que únicamente podemos deducir esto por los metadatos de la imágen.
    """
    if imagen_metadata.modelo == "YIQ":
        return True
    else:
        return False

def es_HSI(imagen_metadata):
    """
    Nota sobre YIQ: No hay forma de saber si una matriz es YIQ solo por sus valores,
    por lo que únicamente podemos deducir esto por los metadatos de la imágen.
    """
    if imagen_metadata.modelo == "HSI":
        return True
    else:
        return False

# . Devuelve nombres y mapas de color según el modelo
def obtener_config_modelo(tipo_modelo="RGB"):
    """Devuelve nombres y mapas de color según el modelo solicitado."""
    configuraciones = {
        "RGB": {
            "nombres": ["Canal Rojo", "Canal Verde", "Canal Azul"],
            "cmaps": ["Reds", "Greens", "Blues"]
        },
        "HSV": {
            "nombres": ["Matiz (Hue)", "Saturación", "Valor (Brillo)"],
            "cmaps": ["hsv", "gray", "gray"]
        },
        "CMY": {
            "nombres": ["Cian (C)", "Magenta (M)", "Amarillo (Y)"],
            "cmaps": ["Blues_r", "Purples_r", "YlOrBr_r"]
        },
        "GRIS": {
            "nombres": ["Intensidad Gris"],
            "cmaps": ["gray"]
        },
        "BINARIO": {
            "nombres": ["Intensidad Gris"],
            "cmaps": ["gray"]
        },
        "YIQ": {
            "nombres": ["Luminancia (Y)", "Fase-I", "Cuadratura-Q"],
            "cmaps": ["gray", "coolwarm", "coolwarm"] # Coolwarm ayuda a ver valores +/-
        },
        "HSI": {
            "nombres": ["Matiz (Hue)", "Saturación", "Intensidad"],
            "cmaps": ["hsv", "gray", "gray"]
        }
    }
    # Devuelve la config solicitada o una genérica por defecto
    return configuraciones.get(tipo_modelo.upper(), {"nombres": None, "cmaps": None})

# ── Actualiza la imagen actual a partir de un estado anterior guardado en el historial ──
def cargar_estado_historial(historial_imagen_metadata, imagen_actual_metadata):
    respuesta = None
    match historial_imagen_metadata.modelo:
        case "RGB":
            respuesta = cargar_imagen_opencv_rgb(imagen_actual_metadata)
        case "HSV":
            respuesta = conversion_imagen_opencv_hsv(imagen_actual_metadata)
        case "CMY":
            respuesta = conversion_imagen_opencv_cmy(imagen_actual_metadata)
        case "GRIS":
            respuesta = cargar_imagen_opencv_gris(imagen_actual_metadata)
        case "BINARIO":
            respuesta = conversion_imagen_opencv_binaria(imagen_actual_metadata, historial_imagen_metadata.umbral)
        case "YIQ":
            respuesta = conversion_imagen_opencv_yiq(imagen_actual_metadata)
        case "HSI":
            respuesta = conversion_imagen_opencv_hsi(imagen_actual_metadata)
        case _:
            respuesta = wrapper_respuesta(imagen_actual_metadata, False, f"No se encontró el modelo: {historial_imagen_metadata.modelo}")
    return respuesta


# ══════════════════════════════════════════════════════════════
#  GUARDADO DE ARCHIVOS
# ══════════════════════════════════════════════════════════════

# Paleta de colores compartida para los plots de guardado
_COLOR_MAP_GUARDADO = {
    "Reds": "#FF5555", "Greens": "#50FA7B", "Blues": "#8BE9FD",
    "hsv": "#FF79C6", "gray": "#CDD9E5", "Purples": "#BD93F9",
    "YlOrBr": "#FFB86C", "coolwarm": "#6272A4",
    "Blues_r": "#8BE9FD", "Purples_r": "#BD93F9", "YlOrBr_r": "#FFB86C"
}


def _generar_timestamp():
    """Genera un timestamp con formato YYYYMMDD_HHMMSS."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def _nombre_base(imagen_metadata):
    """Devuelve el nombre del archivo sin extensión."""
    return os.path.splitext(imagen_metadata.nombre)[0]


def _datos_a_bgr(imagen_metadata):
    """
    Convierte los datos internos (que pueden ser float o RGB) a BGR uint8
    listo para cv2.imwrite. Devuelve None si no se puede convertir.
    """
    datos = imagen_metadata.datos
    if datos is None:
        return None

    # Asegurar uint8
    if datos.dtype != np.uint8:
        datos = (np.clip(datos, 0, 1) * 255).astype(np.uint8)

    # Imagen monocromática (2D) → devolver tal cual
    if len(datos.shape) == 2:
        return datos

    # RGB → BGR para OpenCV
    if imagen_metadata.modelo == "RGB":
        return cv2.cvtColor(datos, cv2.COLOR_RGB2BGR)

    # HSV almacenado como RGB→HSV → reconvertir a BGR para guardar legible
    if imagen_metadata.modelo == "HSV":
        return cv2.cvtColor(cv2.cvtColor(datos, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2BGR)

    # CMY, YIQ, HSI y otros modelos float normalizados: guardar como RGB→BGR
    return cv2.cvtColor(datos, cv2.COLOR_RGB2BGR)


def guardar_imagen(imagen_metadata, carpeta_destino):
    """
    Guarda la imagen actual en disco.
    Nombre: <base>_<MODELO>_original_<timestamp>.png
    Devuelve: { "error": bool, "mensaje": str, "archivos": [rutas] }
    """
    try:
        if imagen_metadata.datos is None:
            return {"error": True, "mensaje": "No hay imagen cargada.", "archivos": []}

        ts   = _generar_timestamp()
        base = _nombre_base(imagen_metadata)
        nombre_archivo = f"{base}_{imagen_metadata.modelo}_original_{ts}.png"
        ruta = os.path.join(carpeta_destino, nombre_archivo)

        bgr = _datos_a_bgr(imagen_metadata)
        if bgr is None:
            return {"error": True, "mensaje": "No se pudieron convertir los datos para guardar.", "archivos": []}

        ok = cv2.imwrite(ruta, bgr)
        if not ok:
            return {"error": True, "mensaje": f"cv2.imwrite falló al escribir: {nombre_archivo}", "archivos": []}

        return {"error": False, "mensaje": f"Imagen guardada: {nombre_archivo}", "archivos": [ruta]}

    except Exception as e:
        return {"error": True, "mensaje": f"Error al guardar imagen: {str(e)}", "archivos": []}


def guardar_histograma(imagen_metadata, carpeta_destino):
    """
    Genera el histograma compuesto en alta resolución y lo guarda en disco.
    Nombre: <base>_<MODELO>_histograma_<timestamp>.png
    Devuelve: { "error": bool, "mensaje": str, "archivos": [rutas] }
    """
    try:
        if not imagen_metadata.histograma:
            # Intentar calcularlo antes de fallar
            resp = proceso_histograma_completo(imagen_metadata)
            if resp["error"]:
                return {"error": True, "mensaje": "No hay histograma y no se pudo calcular.", "archivos": []}
            imagen_metadata = resp["objeto"]

        conf   = obtener_config_modelo(imagen_metadata.modelo)
        cmaps  = conf.get("cmaps") or []
        colores = [_COLOR_MAP_GUARDADO.get(c.replace("_r", ""), "#CDD9E5") for c in cmaps]

        bg = "#080C10"; fg = "#CDD9E5"; grid_c = "#1E2D3D"
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor("#0D1117")

        for i, (canal, props) in enumerate(imagen_metadata.histograma.items()):
            color = colores[i] if i < len(colores) else "#CDD9E5"
            ax.plot(props["histograma_raw"], color=color, label=canal, alpha=0.85, linewidth=1.8)

        ax.set_title(
            f"Histograma Compuesto  ·  {imagen_metadata.modelo}  ·  {imagen_metadata.nombre}",
            color=fg, fontsize=12, pad=14
        )
        ax.set_xlabel("Intensidad (0–255)", color="#6B8FA8", fontsize=10)
        ax.set_ylabel("Frecuencia", color="#6B8FA8", fontsize=10)
        ax.tick_params(colors="#6B8FA8")
        ax.legend(facecolor="#0D1117", labelcolor=fg, fontsize=10, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.25, color=grid_c)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_c)

        plt.tight_layout()

        ts   = _generar_timestamp()
        base = _nombre_base(imagen_metadata)
        nombre_archivo = f"{base}_{imagen_metadata.modelo}_histograma_{ts}.png"
        ruta = os.path.join(carpeta_destino, nombre_archivo)

        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=bg)
        plt.close(fig)

        return {"error": False, "mensaje": f"Histograma guardado: {nombre_archivo}", "archivos": [ruta]}

    except Exception as e:
        plt.close("all")
        return {"error": True, "mensaje": f"Error al guardar histograma: {str(e)}", "archivos": []}


def guardar_canales(imagen_metadata, carpeta_destino):
    """
    Genera la visualización de canales separados en alta resolución y la guarda en disco.
    Nombre: <base>_<MODELO>_canales_<timestamp>.png
    Devuelve: { "error": bool, "mensaje": str, "archivos": [rutas] }
    """
    try:
        if imagen_metadata.datos is None:
            return {"error": True, "mensaje": "No hay imagen cargada.", "archivos": []}

        conf    = obtener_config_modelo(imagen_metadata.modelo)
        nombres = conf["nombres"]
        mapas   = conf["cmaps"]

        datos = imagen_metadata.datos
        canales = [datos] if len(datos.shape) == 2 else cv2.split(datos)
        n = len(canales)

        bg = "#080C10"; fg = "#CDD9E5"
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=150)
        fig.patch.set_facecolor(bg)
        if n == 1:
            axes = [axes]

        for i, (c, ax) in enumerate(zip(canales, axes)):
            ax.set_facecolor(bg)
            nombre = nombres[i] if nombres and i < len(nombres) else f"Canal {i + 1}"
            cmap   = mapas[i]   if mapas   and i < len(mapas)   else "gray"
            ax.imshow(c, cmap=cmap)
            ax.set_title(nombre, color=fg, fontsize=11)
            ax.axis("off")

        fig.suptitle(
            f"Canales  ·  {imagen_metadata.modelo}  ·  {imagen_metadata.nombre}",
            color=fg, fontsize=12
        )
        plt.tight_layout()

        ts   = _generar_timestamp()
        base = _nombre_base(imagen_metadata)
        nombre_archivo = f"{base}_{imagen_metadata.modelo}_canales_{ts}.png"
        ruta = os.path.join(carpeta_destino, nombre_archivo)

        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=bg)
        plt.close(fig)

        return {"error": False, "mensaje": f"Canales guardados: {nombre_archivo}", "archivos": [ruta]}

    except Exception as e:
        plt.close("all")
        return {"error": True, "mensaje": f"Error al guardar canales: {str(e)}", "archivos": []}