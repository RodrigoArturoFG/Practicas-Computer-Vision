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
        False

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