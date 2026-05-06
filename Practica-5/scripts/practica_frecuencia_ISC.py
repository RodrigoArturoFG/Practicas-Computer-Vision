# Ver 2.0 - Extendida por Rodrigo Arturo Fernández González
# Basado en Ver 1.0 por mcruzm@ipn.mx
# -*- coding: utf-8 -*-
"""
Práctica de laboratorio: Transformaciones en el dominio de la frecuencia (FFT y DCT)
ISC 7º semestre — Python

VERSIÓN EXTENDIDA CON TRABAJO AUTÓNOMO:
- Filtro Notch (rechazo de banda) para eliminar frecuencias específicas
- Selección de K coeficientes DCT de mayor energía
- Generación de tablas comparativas automáticas
- Análisis sistemático de filtros

Este script incluye:
- Carga de imagen (PIL) y conversión a escala de grises.
- Cálculo y visualización del espectro de magnitud y fase (FFT 2D).
- Filtros en el dominio de la frecuencia: ideal, gaussiano, Butterworth y NOTCH.
- Compresión basada en DCT por bloques de 8x8 con cuantización tipo JPEG.
- Compresión DCT alternativa: selección de top-k coeficientes.
- Reconstrucción y métricas (PSNR).

Uso rápido (ejemplos):
python practica_frecuencia_ISC.py --imagen data/girl.gif --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --dct_q 0.5
python practica_frecuencia_ISC.py --imagen data/cam.gif --filtro notch --notch_centros "60,60;-60,-60" --notch_radio 10
python practica_frecuencia_ISC.py --imagen data/mandrill.gif --dct_topk 10
python practica_frecuencia_ISC.py --modo tabla_comparativa --imagen data/pcb.gif

Si no se proporciona imagen, se genera una imagen de prueba (tablero damero + círculos).
"""

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# ----------------------------- Utilidades de imagen -----------------------------

def cargar_imagen(ruta=None, tamaño_max=512):
    """Carga una imagen y la convierte a escala de grises float32 [0,1].
    Si no hay ruta, genera una imagen sintética de prueba.
    """
    if ruta and os.path.exists(ruta):
        img = Image.open(ruta).convert('L')
    else:
        # Imagen sintética: damero + formas
        n = tamaño_max
        img = Image.new('L', (n, n), color=0)
        # damero
        tile = n // 16
        for i in range(0, n, tile):
            for j in range(0, n, tile):
                if ((i//tile) + (j//tile)) % 2 == 0:
                    ImageDraw.Draw(img).rectangle([i, j, i+tile-1, j+tile-1], fill=180)
        # círculos
        draw = ImageDraw.Draw(img)
        for r,c in [(n//4, n//4), (3*n//4, 3*n//4), (n//4, 3*n//4)]:
            draw.ellipse([r-40, c-40, r+40, c+40], outline=255, width=3)
    # redimensionar si es muy grande
    if max(img.size) > tamaño_max:
        scale = tamaño_max / float(max(img.size))
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# ----------------------------- FFT y filtros -----------------------------

def fft2_imagen(img):
    """Calcula FFT 2D, espectros de magnitud (log) y fase, y devuelve el fft desplazado."""
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitud = np.log(1 + np.abs(Fshift))
    fase = np.angle(Fshift)
    return F, Fshift, magnitud, fase


def crear_mascara(img_shape, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2):
    """Crea una máscara de filtro en el dominio de la frecuencia.
    - filtro: 'ideal', 'gaussiano', 'butterworth'
    - tipo: 'lowpass' o 'highpass'
    - cutoff: radios fraccionarios (0-0.5 aprox), relativo al tamaño mínimo.
    - orden: solo usado para Butterworth.
    """
    rows, cols = img_shape
    crow, ccol = rows//2, cols//2
    Y, X = np.ogrid[:rows, :cols]
    # distancia al centro
    D = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    Dnorm = D / float(min(crow, ccol))  # normalizar por semitamaño mínimo

    if filtro == 'ideal':
        H = (Dnorm <= cutoff).astype(np.float32)
    elif filtro == 'gaussiano':
        # H = exp(-(D^2)/(2*Dc^2)) con Dc = cutoff
        H = np.exp(-(Dnorm**2) / (2 * (cutoff**2)))
    elif filtro == 'butterworth':
        # H = 1 / (1 + (D/Dc)^(2*orden))
        H = 1 / (1 + (Dnorm / (cutoff + 1e-8))**(2*orden))
    else:
        raise ValueError('Filtro desconocido')

    if tipo == 'lowpass':
        mask = H
    elif tipo == 'highpass':
        mask = 1 - H
    else:
        raise ValueError('Tipo de filtro debe ser lowpass o highpass')

    return mask.astype(np.float32)


def crear_mascara_notch(img_shape, centros_notch, radio_notch=10):
    """
    EXTENSIÓN: Filtro Notch (rechazo de banda).
    
    Crea una máscara que rechaza frecuencias en puntos específicos del espectro.
    Útil para eliminar patrones periódicos como:
    - Ruido de escaneo
    - Patrones de moiré
    - Interferencias periódicas
    
    Args:
        img_shape: tuple (rows, cols) - forma de la imagen
        centros_notch: lista de tuplas [(u1, v1), (u2, v2), ...] 
                      Coordenadas relativas al centro (0,0) = centro del espectro
                      Ejemplo: [(60, 60), (-60, -60)] rechaza frecuencias simétricas
        radio_notch: radio de rechazo alrededor de cada centro (en píxeles)
    
    Returns:
        mask: máscara float32 con 0s en las frecuencias rechazadas y 1s en el resto
        
    Nota: El filtro rechaza tanto el punto como su simétrico (por ser el espectro simétrico)
    """
    rows, cols = img_shape
    crow, ccol = rows//2, cols//2
    
    # Inicializar máscara con unos (pasa todo)
    mask = np.ones((rows, cols), dtype=np.float32)
    
    # Para cada centro de rechazo
    for u_rel, v_rel in centros_notch:
        # Convertir coordenadas relativas a absolutas
        u_abs = crow + u_rel
        v_abs = ccol + v_rel
        
        # Crear máscara circular de rechazo
        Y, X = np.ogrid[:rows, :cols]
        
        # Distancia al centro de rechazo
        D1 = np.sqrt((Y - u_abs)**2 + (X - v_abs)**2)
        
        # Punto simétrico (por simetría del espectro de Fourier)
        u_sim = crow - u_rel
        v_sim = ccol - v_rel
        D2 = np.sqrt((Y - u_sim)**2 + (X - v_sim)**2)
        
        # Rechazar (poner en 0) las frecuencias dentro del radio
        mask[D1 <= radio_notch] = 0
        mask[D2 <= radio_notch] = 0
    
    return mask


def aplicar_filtro_fft(img, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2, 
                       centros_notch=None, radio_notch=10):
    """Aplica el filtro elegido en el dominio de la frecuencia y reconstruye la imagen.
    
    Args:
        img: imagen de entrada
        filtro: 'ideal', 'gaussiano', 'butterworth', 'notch'
        tipo: 'lowpass' o 'highpass' (no aplica para notch)
        cutoff: radio de corte normalizado
        orden: orden del filtro (solo Butterworth)
        centros_notch: lista de tuplas para filtro notch
        radio_notch: radio de rechazo para filtro notch
    """
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    
    if filtro == 'notch':
        if centros_notch is None:
            raise ValueError('Debe proporcionar centros_notch para filtro notch')
        mask = crear_mascara_notch(img.shape, centros_notch, radio_notch)
    else:
        mask = crear_mascara(img.shape, filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden)
    
    Gshift = Fshift * mask
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G)
    g = np.real(g)
    g = np.clip(g, 0, 1)
    return g, mask

# ----------------------------- DCT por bloques 8x8 -----------------------------

def dct_matrix(N=8):
    """Genera la matriz de transformada DCT tipo II (ortogonal) de tamaño N."""
    C = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        alpha = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
        for n in range(N):
            C[k, n] = alpha * math.cos(((2*n + 1) * k * math.pi) / (2*N))
    return C

C8 = dct_matrix(8)


def dct_bloque_2d(b):
    """DCT 2D (tipo II) por multiplicación matricial: D = C * b * C^T."""
    return C8 @ b @ C8.T


def idct_bloque_2d(D):
    """IDCT 2D (tipo III equivalente al inverso ortogonal): b = C^T * D * C."""
    return C8.T @ D @ C8

# Tabla de cuantización luminancia estándar JPEG (aproximada)
Q_JPEG = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float64)


def pad_a_multiplo(img, N=8):
    """Rellena (padding) la imagen para que ambos ejes sean múltiplos de N."""
    h, w = img.shape
    nh = ((h + N - 1)//N)*N
    nw = ((w + N - 1)//N)*N
    padded = np.zeros((nh, nw), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w


def dct_compresion(img, q_factor=0.5):
    """Aplica compresión DCT por bloques 8x8 con cuantización.
    q_factor en [0.1, 2.0] aprox: menor -> más compresión (más pérdida).
    Devuelve: reconstruida, psnr, img_padded, reconstruida_padded
    """
    img_p = img.copy()
    padded, h, w = pad_a_multiplo(img_p, 8)
    H, W = padded.shape

    Q = Q_JPEG * q_factor

    # contenedores
    coef_cuant = np.zeros_like(padded, dtype=np.float64)
    recon = np.zeros_like(padded, dtype=np.float64)

    # procesar bloques
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            b = padded[i:i+8, j:j+8]
            # centrar señal a rango [-0.5, 0.5]
            b_shift = b - 0.5
            D = dct_bloque_2d(b_shift)
            # cuantización
            Dq = np.round(D / Q)
            # almacenamiento (opcional)
            coef_cuant[i:i+8, j:j+8] = Dq
            # de-cuantización y reconstrucción
            Dr = Dq * Q
            br = idct_bloque_2d(Dr) + 0.5
            recon[i:i+8, j:j+8] = br

    # recortar a tamaño original y saturar a [0,1]
    recon = np.clip(recon[:h, :w], 0, 1)

    psnr = calcular_psnr(img[:h, :w], recon)
    return recon, psnr, padded, recon


def dct_compresion_topk(img, k=10):
    """
    EXTENSIÓN: Compresión DCT por selección de top-k coeficientes de mayor energía.
    
    En lugar de cuantización uniforme, este método:
    1. Calcula DCT de cada bloque 8x8
    2. Selecciona solo los k coeficientes de mayor magnitud
    3. Pone en cero todos los demás coeficientes
    4. Reconstruye la imagen
    
    Esto simula compresión adaptativa donde se preserva solo la información
    más importante en términos de energía.
    
    Args:
        img: imagen de entrada [0, 1]
        k: número de coeficientes DCT a preservar por bloque (1-64)
           k=1: solo DC (promedio del bloque)
           k=10: DC + 9 coeficientes AC de mayor energía
           k=64: sin compresión
    
    Returns:
        recon: imagen reconstruida [0, 1]
        psnr: Peak Signal-to-Noise Ratio
        ratio_compresion: porcentaje de coeficientes preservados
    """
    if k < 1 or k > 64:
        raise ValueError('k debe estar entre 1 y 64')
    
    img_p = img.copy()
    padded, h, w = pad_a_multiplo(img_p, 8)
    H, W = padded.shape
    
    recon = np.zeros_like(padded, dtype=np.float64)
    total_coefs = 0
    coefs_preservados = 0
    
    # Procesar bloques
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            b = padded[i:i+8, j:j+8]
            # Centrar señal
            b_shift = b - 0.5
            
            # DCT 2D
            D = dct_bloque_2d(b_shift)
            
            # Seleccionar top-k coeficientes
            D_flat = D.flatten()
            magnitudes = np.abs(D_flat)
            
            # Índices de los k coeficientes de mayor magnitud
            idx_topk = np.argsort(magnitudes)[-k:]
            
            # Crear máscara: preservar solo top-k
            D_compressed = np.zeros_like(D_flat)
            D_compressed[idx_topk] = D_flat[idx_topk]
            D_compressed = D_compressed.reshape(8, 8)
            
            # Reconstrucción
            br = idct_bloque_2d(D_compressed) + 0.5
            recon[i:i+8, j:j+8] = br
            
            # Estadísticas
            total_coefs += 64
            coefs_preservados += k
    
    # Recortar y saturar
    recon = np.clip(recon[:h, :w], 0, 1)
    
    # Calcular PSNR
    psnr = calcular_psnr(img[:h, :w], recon)
    
    # Ratio de compresión
    ratio_compresion = (coefs_preservados / total_coefs) * 100
    
    return recon, psnr, ratio_compresion


def calcular_psnr(img_ref, img_rec):
    """Calcula Peak Signal-to-Noise Ratio entre imagen de referencia y reconstruida."""
    mse = np.mean((img_ref - img_rec)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)

# ----------------------------- Visualización -----------------------------

def mostrar_fft(img, filtro='ideal', tipo='lowpass', cutoff=0.2, orden=2, 
                centros_notch=None, radio_notch=10, guardar=None):
    """Visualiza FFT y filtrado."""
    F, Fshift, magnitud, fase = fft2_imagen(img)
    filtrada, mask = aplicar_filtro_fft(img, filtro=filtro, tipo=tipo, cutoff=cutoff, 
                                        orden=orden, centros_notch=centros_notch, 
                                        radio_notch=radio_notch)

    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen original (escala de grises)')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(magnitud, cmap='gray')
    plt.title('Espectro de magnitud (log)')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(fase, cmap='twilight')
    plt.title('Espectro de fase')
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(mask, cmap='gray')
    if filtro == 'notch':
        plt.title(f'Máscara Notch\nradio={radio_notch}')
    else:
        plt.title(f'Máscara {filtro} {"pasa bajas" if tipo=="lowpass" else "pasa altas"}\ncutoff={cutoff}, orden={orden}')
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.imshow(filtrada, cmap='gray')
    plt.title('Imagen filtrada (IFFT)')
    plt.axis('off')

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120)
    plt.show()


def mostrar_dct(img, q_factor=0.5, guardar=None):
    """Visualiza compresión DCT con cuantización."""
    rec, psnr, _, _ = dct_compresion(img, q_factor=q_factor)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rec, cmap='gray')
    plt.title(f'Reconstruida DCT (q={q_factor})\nPSNR={psnr:.2f} dB')
    plt.axis('off')

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120)
        
        # Guardar PSNR en archivo de texto para extracción automática
        carpeta_salida = os.path.dirname(guardar)
        archivo_psnr = os.path.join(carpeta_salida, 'psnr.txt')
        with open(archivo_psnr, 'w') as f:
            f.write(f'{psnr:.2f}\n')
    
    plt.show()


def mostrar_dct_topk(img, k_values=[1, 5, 10, 20], guardar=None):
    """Visualiza compresión DCT con diferentes valores de k."""
    n_plots = len(k_values) + 1
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    plt.figure(figsize=(12, 4*rows))
    
    # Original
    plt.subplot(rows, cols, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Para cada k
    psnr_values = []
    for idx, k in enumerate(k_values, start=2):
        rec, psnr, ratio = dct_compresion_topk(img, k=k)
        psnr_values.append(psnr)
        plt.subplot(rows, cols, idx)
        plt.imshow(rec, cmap='gray')
        plt.title(f'Top-{k} coefs\nPSNR={psnr:.2f} dB\nRatio={ratio:.1f}%')
        plt.axis('off')
    
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120)
        
        # Guardar PSNR promedio en archivo de texto
        carpeta_salida = os.path.dirname(guardar)
        archivo_psnr = os.path.join(carpeta_salida, 'psnr.txt')
        psnr_promedio = sum(psnr_values) / len(psnr_values) if psnr_values else 0
        with open(archivo_psnr, 'w') as f:
            f.write(f'{psnr_promedio:.2f}\n')
            # También guardar los valores individuales como comentario
            f.write(f'# Valores individuales: {", ".join([f"k={k}: {p:.2f}" for k, p in zip(k_values, psnr_values)])}\n')
    
    plt.show()

# ----------------------------- Análisis sistemático -----------------------------

def generar_tabla_comparativa(img, nombre_img, salidas_dir):
    """
    Genera tabla comparativa CSV de filtros FFT.
    Parte del trabajo autónomo.
    """
    resultados = []
    
    # Configuraciones a probar
    filtros = ['ideal', 'gaussiano', 'butterworth']
    tipos = ['lowpass', 'highpass']
    cutoffs = [0.05, 0.1, 0.15, 0.2]
    orden = 2  # Para Butterworth
    
    for filtro in filtros:
        for tipo in tipos:
            for cutoff in cutoffs:
                try:
                    filtrada, mask = aplicar_filtro_fft(
                        img, filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden
                    )
                    
                    # Calcular métricas básicas
                    mean_orig = np.mean(img)
                    mean_filt = np.mean(filtrada)
                    std_orig = np.std(img)
                    std_filt = np.std(filtrada)
                    
                    resultados.append({
                        'Filtro': filtro,
                        'Tipo': tipo,
                        'Cutoff': cutoff,
                        'Orden': orden if filtro == 'butterworth' else '-',
                        'Imagen': nombre_img,
                        'Media_Original': f'{mean_orig:.3f}',
                        'Media_Filtrada': f'{mean_filt:.3f}',
                        'StdDev_Original': f'{std_orig:.3f}',
                        'StdDev_Filtrada': f'{std_filt:.3f}'
                    })
                except Exception as e:
                    print(f"Error en {filtro} {tipo} cutoff={cutoff}: {e}")
    
    # Guardar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(salidas_dir, f'tabla_comparativa_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if resultados:
            writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
            writer.writeheader()
            writer.writerows(resultados)
    
    print(f"Tabla comparativa guardada en: {csv_path}")
    return csv_path

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Práctica de Transformaciones en Frecuencia (FFT y DCT) - Versión Extendida')
    
    # Parámetros básicos
    parser.add_argument('--imagen', type=str, default=None, help='Ruta a imagen .jpg/.png/.gif (opcional)')
    parser.add_argument('--salidas', type=str, default='salidas', help='Carpeta donde guardar figuras')
    
    # Modo de operación
    parser.add_argument('--modo', type=str, default='normal', 
                       choices=['normal', 'tabla_comparativa', 'notch', 'topk'],
                       help='Modo de operación')
    
    # Parámetros FFT
    parser.add_argument('--filtro', type=str, default='butterworth', 
                       choices=['ideal','gaussiano','butterworth','notch'], 
                       help='Tipo de filtro en frecuencia')
    parser.add_argument('--tipo', type=str, default='lowpass', 
                       choices=['lowpass','highpass'], 
                       help='Tipo de filtro: pasa bajas o pasa altas')
    parser.add_argument('--cutoff', type=float, default=0.15, 
                       help='Radio de corte normalizado (0-0.5 aprox)')
    parser.add_argument('--orden', type=int, default=2, 
                       help='Orden (solo Butterworth)')
    
    # Parámetros Notch
    parser.add_argument('--notch_centros', type=str, default=None,
                       help='Centros notch como "u1,v1;u2,v2" ej: "60,60;-60,-60"')
    parser.add_argument('--notch_radio', type=int, default=10,
                       help='Radio de rechazo notch')
    
    # Parámetros DCT
    parser.add_argument('--dct_q', type=float, default=0.5, 
                       help='Factor de cuantización DCT (≈0.3-1.0)')
    parser.add_argument('--dct_topk', type=int, default=None,
                       help='Número de coeficientes DCT a preservar (1-64)')

    args = parser.parse_args()

    os.makedirs(args.salidas, exist_ok=True)

    img = cargar_imagen(args.imagen)
    nombre_img = os.path.basename(args.imagen) if args.imagen else "sintetica"

    # ===== MODO NORMAL =====
    if args.modo == 'normal':
        # Parsear centros notch si es filtro notch
        centros_notch = None
        if args.filtro == 'notch':
            if args.notch_centros:
                # Parsear "60,60;-60,-60" -> [(60,60), (-60,-60)]
                try:
                    pares = args.notch_centros.split(';')
                    centros_notch = [tuple(map(int, p.split(','))) for p in pares]
                except:
                    print("Error parseando notch_centros. Use formato: 'u1,v1;u2,v2'")
                    return
            else:
                print("Filtro notch requiere --notch_centros")
                return
        
        # Detectar si se pidió DCT explícitamente
        # (si no se especificó --dct_q en línea de comandos, usar None)
        import sys
        dct_solicitado = '--dct_q' in sys.argv or '--dct_topk' in sys.argv
        
        # Mostrar FFT y filtrado (siempre, excepto si solo se pidió DCT)
        if not (dct_solicitado and '--filtro' not in sys.argv):
            mostrar_fft(img, filtro=args.filtro, tipo=args.tipo, cutoff=args.cutoff, 
                       orden=args.orden, centros_notch=centros_notch, 
                       radio_notch=args.notch_radio,
                       guardar=os.path.join(args.salidas, 'fft_filtrado.png'))

        # Mostrar DCT solo si se solicitó explícitamente
        if dct_solicitado:
            if args.dct_topk is not None:
                mostrar_dct_topk(img, k_values=[args.dct_topk],
                               guardar=os.path.join(args.salidas, 'dct_topk.png'))
            else:
                mostrar_dct(img, q_factor=args.dct_q, 
                           guardar=os.path.join(args.salidas, 'dct_reconstruccion.png'))

    # ===== MODO TABLA COMPARATIVA =====
    elif args.modo == 'tabla_comparativa':
        generar_tabla_comparativa(img, nombre_img, args.salidas)

    # ===== MODO NOTCH =====
    elif args.modo == 'notch':
        if not args.notch_centros:
            print("Modo notch requiere --notch_centros")
            return
        try:
            pares = args.notch_centros.split(';')
            centros_notch = [tuple(map(int, p.split(','))) for p in pares]
        except:
            print("Error parseando notch_centros. Use formato: 'u1,v1;u2,v2'")
            return
        
        mostrar_fft(img, filtro='notch', tipo='lowpass', cutoff=0, orden=0,
                   centros_notch=centros_notch, radio_notch=args.notch_radio,
                   guardar=os.path.join(args.salidas, 'notch_filtrado.png'))

    # ===== MODO TOP-K =====
    elif args.modo == 'topk':
        k_vals = [1, 5, 10, 20, 30, 40] if args.dct_topk is None else [args.dct_topk]
        mostrar_dct_topk(img, k_values=k_vals,
                       guardar=os.path.join(args.salidas, 'dct_topk_comparacion.png'))

    print(f'Listo. Figuras guardadas en: {args.salidas}')

if __name__ == '__main__':
    main()