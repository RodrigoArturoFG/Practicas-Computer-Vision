#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script auxiliar para extraer métricas PSNR de los resultados
y generar tabla para el reporte de la Práctica 5

Uso:
    python extraer_metricas.py --carpeta salidas
    python extraer_metricas.py --carpeta salidas --salida mi_tabla

Genera archivos CSV y Markdown con timestamp automático en la misma carpeta
"""

import os
import argparse
import re
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

def extraer_psnr_de_imagen(ruta_imagen):
    """
    Intenta extraer el valor PSNR del título de una figura guardada.
    Asume que el PSNR está en el formato: "PSNR=XX.XX dB"
    """
    try:
        # El archivo dct_reconstruccion.png tiene el PSNR en el título
        # Pero como Matplotlib no guarda el título en metadatos,
        # necesitaríamos usar OCR o parsing de la imagen
        
        # Por ahora, intentar buscar en el nombre del archivo
        nombre = ruta_imagen.stem
        
        # Buscar patrón PSNR en el nombre (si lo guardamos así)
        match = re.search(r'psnr[_-]?(\d+\.?\d*)', nombre.lower())
        if match:
            return float(match.group(1))
        
        return None
    except:
        return None


def buscar_psnr_en_carpeta(carpeta_path):
    """
    Busca el valor PSNR en los archivos de una carpeta de resultados DCT.
    Primero intenta leer de psnr.txt, luego del nombre de archivo.
    """
    # 1. Buscar archivo psnr.txt
    archivo_psnr = carpeta_path / 'psnr.txt'
    if archivo_psnr.exists():
        try:
            with open(archivo_psnr, 'r') as f:
                contenido = f.read().strip()
                # Buscar número decimal
                match = re.search(r'(\d+\.?\d*)', contenido)
                if match:
                    return f"{float(match.group(1)):.2f}"
        except:
            pass
    
    # 2. Buscar archivo dct_reconstruccion.png con PSNR en nombre
    archivos_dct = list(carpeta_path.glob('dct_*.png'))
    
    for archivo in archivos_dct:
        psnr = extraer_psnr_de_imagen(archivo)
        if psnr:
            return f"{psnr:.2f}"
    
    # Si no se pudo extraer, el usuario debe hacerlo manualmente
    return '(calcular)'


def analizar_carpeta_resultados(carpeta):
    """
    Analiza una carpeta de resultados y extrae información
    de los nombres de archivos y subcarpetas.
    """
    resultados = []
    
    carpeta_path = Path(carpeta)
    
    if not carpeta_path.exists():
        print(f"Error: La carpeta '{carpeta}' no existe")
        return resultados
    
    # Buscar subcarpetas de resultados
    subcarpetas = sorted([d for d in carpeta_path.iterdir() if d.is_dir()])
    
    for subcarpeta in subcarpetas:
        nombre = subcarpeta.name
        
        # Ignorar carpeta tabla_comparativa
        if 'tabla_comparativa' in nombre or 'tabla_test' in nombre or 'tabla_resultados' in nombre:
            continue
        
        # Parsear el nombre de la subcarpeta
        # Formato esperado: 01_ideal_low_005_cam1, 14_dct_q03_fce5, etc.
        
        # Determinar tipo de experimento
        filtro = None
        parametros = []
        imagen = 'Desconocida'
        
        # Detectar tipo de filtro
        if '_ideal_' in nombre:
            filtro = 'Ideal'
        elif '_gauss' in nombre:
            filtro = 'Gaussiano'
        elif '_butter' in nombre:
            filtro = 'Butterworth'
        elif '_notch_' in nombre:
            filtro = 'Notch'
        elif '_dct_' in nombre:
            if 'topk' in nombre:
                filtro = 'DCT Top-K'
            else:
                filtro = 'DCT Cuantización'
        
        if filtro is None:
            continue  # Saltar carpetas no reconocidas
        
        # Detectar tipo (lowpass/highpass)
        if '_low_' in nombre or '_lowpass_' in nombre:
            parametros.append('Lowpass')
        elif '_high_' in nombre or '_highpass_' in nombre:
            parametros.append('Highpass')
        
        # Detectar cutoff para filtros FFT
        if filtro in ['Ideal', 'Gaussiano', 'Butterworth']:
            # Buscar patrón _XXX_ donde XXX es el cutoff
            # Ejemplos: _005_ = 0.05, _015_ = 0.15, _01_ = 0.1
            match_cutoff = re.search(r'_0*(\d+)_', nombre)
            if match_cutoff:
                cutoff_num = match_cutoff.group(1)
                # Determinar valor correcto
                if cutoff_num == '1':
                    cutoff = 0.1
                elif cutoff_num == '5':
                    cutoff = 0.05
                elif cutoff_num == '10':
                    cutoff = 0.1
                elif cutoff_num == '15':
                    cutoff = 0.15
                elif cutoff_num == '005':
                    cutoff = 0.05
                elif cutoff_num == '015':
                    cutoff = 0.15
                elif cutoff_num == '01':
                    cutoff = 0.1
                else:
                    cutoff = float(cutoff_num) / 1000.0
                parametros.append(f'cutoff={cutoff}')
        
        # Orden para Butterworth
        if filtro == 'Butterworth':
            parametros.append('n=2')
        
        # Parámetros para DCT cuantización
        if filtro == 'DCT Cuantización':
            match_q = re.search(r'_q(\d{2,3})_', nombre)
            if match_q:
                q_num = match_q.group(1)
                q = float(q_num) / 10.0
                parametros.append(f'q={q}')
        
        # Parámetros para Top-K
        if filtro == 'DCT Top-K':
            parametros.append('k=5,10,20,30,40')
        
        # Parámetros para Notch
        if filtro == 'Notch':
            parametros.append('centros=(60,60;-60,-60)')
            parametros.append('r=10')
        
        # Detectar imagen
        if 'cam1' in nombre:
            imagen = 'cam1.gif'
        elif 'pcb2' in nombre:
            imagen = 'pcb2.gif'
        elif 'fce5' in nombre:
            imagen = 'fce5.gif'
        elif 'fce1' in nombre:
            imagen = 'fce1.gif'
        elif 'blb1' in nombre:
            imagen = 'blb1.gif'
        elif 'ape1' in nombre:
            imagen = 'ape1.gif'
        
        # Buscar archivos PNG
        archivos = list(subcarpeta.glob('*.png'))
        
        # PSNR solo aplica para DCT
        if filtro in ['DCT Cuantización', 'DCT Top-K']:
            if archivos:
                # Intentar extraer PSNR
                psnr_val = buscar_psnr_en_carpeta(subcarpeta)
                psnr = psnr_val if psnr_val else '(calcular)'
            else:
                psnr = 'N/A'
        else:
            # Para FFT, el PSNR no aplica
            psnr = 'N/A'
        
        params_str = ', '.join(parametros) if parametros else 'N/A'
        
        resultados.append({
            'Subcarpeta': nombre,
            'Filtro': filtro,
            'Parámetros': params_str,
            'Imagen': imagen,
            'PSNR (dB)': psnr,
            'Observaciones': ''
        })
    
    return resultados


def generar_tabla_csv(resultados, archivo_salida):
    """Genera archivo CSV con los resultados."""
    
    if not resultados:
        print("No hay resultados para generar tabla")
        return
    
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        campos = ['Filtro', 'Parámetros', 'Imagen', 'PSNR (dB)', 'Observaciones']
        writer = csv.DictWriter(f, fieldnames=campos)
        
        writer.writeheader()
        for resultado in resultados:
            # Solo escribir campos relevantes para el reporte
            writer.writerow({
                'Filtro': resultado['Filtro'],
                'Parámetros': resultado['Parámetros'],
                'Imagen': resultado['Imagen'],
                'PSNR (dB)': resultado['PSNR (dB)'],
                'Observaciones': resultado['Observaciones']
            })
    
    print(f"\n✓ Tabla CSV generada: {archivo_salida}")


def generar_tabla_markdown(resultados, archivo_salida):
    """Genera archivo Markdown con tabla formateada."""
    
    if not resultados:
        print("No hay resultados para generar tabla")
        return
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("# Tabla de Resultados - Práctica 5\n\n")
        f.write("## Parte A: Filtrado FFT\n\n")
        
        # Tabla
        f.write("| Filtro | Parámetros | Imagen | PSNR (dB) | Observaciones |\n")
        f.write("|--------|------------|--------|-----------|---------------|\n")
        
        for resultado in resultados:
            if 'DCT' not in resultado['Filtro']:
                f.write(f"| {resultado['Filtro']} | {resultado['Parámetros']} | "
                       f"{resultado['Imagen']} | {resultado['PSNR (dB)']} | "
                       f"{resultado['Observaciones']} |\n")
        
        f.write("\n## Parte B: Compresión DCT\n\n")
        f.write("| Filtro | Parámetros | Imagen | PSNR (dB) | Observaciones |\n")
        f.write("|--------|------------|--------|-----------|---------------|\n")
        
        for resultado in resultados:
            if 'DCT' in resultado['Filtro']:
                f.write(f"| {resultado['Filtro']} | {resultado['Parámetros']} | "
                       f"{resultado['Imagen']} | {resultado['PSNR (dB)']} | "
                       f"{resultado['Observaciones']} |\n")
    
    print(f"✓ Tabla Markdown generada: {archivo_salida}")


def main():
    parser = argparse.ArgumentParser(
        description='Extrae métricas de resultados de Práctica 5'
    )
    parser.add_argument('--carpeta', type=str, required=True,
                       help='Carpeta con resultados (ej: salidas)')
    parser.add_argument('--salida', type=str, default='tabla_resultados',
                       help='Nombre base para archivos de salida (sin timestamp)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXTRACTOR DE MÉTRICAS - PRÁCTICA 5")
    print("="*70)
    
    # Generar timestamp en formato YYYYMMDD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Construir nombre de subcarpeta con timestamp
    nombre_subcarpeta = f"{args.salida}_{timestamp}"
    
    # Determinar carpeta de salida (crear subcarpeta dentro de la carpeta de entrada)
    carpeta_entrada = Path(args.carpeta)
    carpeta_salida = carpeta_entrada / nombre_subcarpeta
    
    # Crear la subcarpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    
    # Crear rutas completas para archivos de salida
    archivo_csv = carpeta_salida / f"{args.salida}.csv"
    archivo_md = carpeta_salida / f"{args.salida}.md"
    
    # Analizar carpeta
    print(f"\nAnalizando carpeta: {args.carpeta}")
    resultados = analizar_carpeta_resultados(args.carpeta)
    
    if resultados:
        print(f"\n✓ Se encontraron {len(resultados)} experimentos")
        
        # Generar archivos
        generar_tabla_csv(resultados, str(archivo_csv))
        generar_tabla_markdown(resultados, str(archivo_md))
        
        print("\n" + "="*70)
        print("ARCHIVOS GENERADOS")
        print("="*70)
        print(f"Ubicación: {carpeta_salida.absolute()}")
        print(f"1. {args.salida}.csv  - Para importar a Excel")
        print(f"2. {args.salida}.md   - Para visualizar en Markdown")
        print(f"\nCarpeta: {nombre_subcarpeta}/")
        print("\nNOTA SOBRE PSNR:")
        print("  - FFT: PSNR = N/A (no aplica en filtrado)")
        print("  - DCT: Si existe psnr.txt en la carpeta, se extrae automáticamente")
        print("  - DCT: Si no existe psnr.txt, aparece '(calcular)' - revisar figuras")
        print(f"\nArchivos guardados con timestamp: {timestamp}")
    else:
        print("\n⚠ No se encontraron resultados en la carpeta")


if __name__ == '__main__':
    main()