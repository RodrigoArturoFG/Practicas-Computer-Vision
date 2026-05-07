# Práctica 5: Transformadas de Frecuencia (FFT y DCT)

---

**Institución:** ESCOM - Instituto Politécnico Nacional
**Estudiante:** Rodrigo Arturo Fernández González
**Matrícula:** 2009630357
**Carrera:** Ingeniería en Sistemas Computacionales
**Grupo:** 7CM3
**Asignatura:** Análisis de Imágenes
**Profesora:** M. en C. María Elena Cruz Meza

---

## Descripción

Esta práctica integra el análisis frecuencial en el dashboard multi-práctica **Vision Lab**.
Se implementan filtros en el dominio de la frecuencia (FFT) y compresión por transformada discreta del coseno (DCT), con una interfaz interactiva que permite explorar los resultados de forma visual.

El tab **"ANÁLISIS FRECUENCIAL"** se agrega al dashboard que ya incluía las prácticas 1–4.

---

## Estructura del Proyecto

```
Practica-5/
├── scripts/
│   ├── analisis_imagen_dashboard.py   # Dashboard principal (GUI PyQt5)
│   ├── controlador_imagen.py          # Backend con toda la lógica de procesamiento
│   ├── modelo_imagen.py               # Entidad de datos de imagen activa
│   ├── modelo_historial_imagen.py     # Entidad de datos del historial/carrusel
│   ├── config.py                      # Tema visual y constantes
│   ├── extraer_metricas.py            # Script auxiliar: genera tabla CSV/MD de resultados
│   └── practica_frecuencia_ISC.py     # Script CLI: experimentos por lote
├── data/                              # Imágenes de prueba (formato HIPR2)
│   ├── cam1.gif, pcb2.gif, fce5.gif, ape1.gif, art4.gif, art8.gif
│   └── ... (16 imágenes en total)
├── resources/
│   ├── input/                         # Copia de imágenes de prueba para el dashboard
│   └── output/                        # Resultados exportados por el dashboard
│       └── {nombre_imagen}_{timestamp}/
│           ├── imagen_GRIS_original.png
│           ├── imagen_GRIS_histograma.png
│           ├── imagen_fft_{timestamp}_multiview.png
│           └── imagen_fft_{timestamp}_espectro.png
├── salidas/                           # Resultados del script CLI por lote
├── reporte/
│   ├── Guia_Practica_Transformaciones_Frecuencia_AI.pdf
│   ├── Plantilla_Reporte_Practica_Frecuencia.docx
│   └── TABLA_EXPERIMENTOS_PRACTICA5.pdf
├── ejecutar_practica5.bat             # Ejecuta todos los experimentos CLI de una vez
└── README.md                          # Este archivo
```

---

## Ejecución

### Dashboard interactivo (recomendado)

```bash
cd "Practica-5/scripts"
python analisis_imagen_dashboard.py
```

El dashboard incluye las prácticas 1 a 5 unificadas. Para usar el análisis frecuencial:

1. Cargar imagen (botón "Cargar Imagen")
2. Convertir a escala de grises: tab **Preprocesamiento** → "Escala de Grises"
3. Ir al tab **"ANÁLISIS FRECUENCIAL"**
4. Aplicar FFT o DCT con los controles disponibles

### Script CLI (experimentos por lote)

```bash
cd "Practica-5/scripts"

# Filtro FFT (Butterworth lowpass)
python practica_frecuencia_ISC.py --imagen ../data/cam1.gif --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --salidas ../salidas/prueba

# Compresión DCT (cuantización JPEG)
python practica_frecuencia_ISC.py --imagen ../data/fce5.gif --dct_q 0.5 --salidas ../salidas/prueba

# Filtro Notch
python practica_frecuencia_ISC.py --modo notch --imagen ../data/pcb2.gif --notch_centros "60,60;-60,-60" --notch_radio 10 --salidas ../salidas/prueba

# DCT Top-K
python practica_frecuencia_ISC.py --modo topk --imagen ../data/ape1.gif --salidas ../salidas/prueba

# Todos los experimentos de una vez
ejecutar_practica5.bat
```

### Generar tabla comparativa de métricas

```bash
python extraer_metricas.py --carpeta ../salidas
```

Produce un archivo `.csv` y un `.md` con las métricas de todos los experimentos guardados en `salidas/`.

---

## Funcionalidades del Dashboard (tab "ANÁLISIS FRECUENCIAL")

### Sección FFT — Filtrado en Frecuencia

| Control | Descripción |
|---|---|
| Tipo de filtro | Ideal / Gaussiano / Butterworth |
| Modo | Lowpass (pasa-bajas) / Highpass (pasa-altas) |
| Cutoff | Frecuencia de corte normalizada (0.01 – 0.50) |
| Orden Butterworth | 2 / 3 / 4 |
| Botón "Aplicar Filtro FFT" | Calcula FFT, aplica máscara, reconstruye imagen |

Al aplicar, se muestra una **vista múltiple con 3 imágenes horizontales**:

```
[ Original ]  |  [ Espectro de Magnitud ]  |  [ Imagen Filtrada ]
```

### Sección Notch — Rechazo de Frecuencias Específicas (extensión)

Permite eliminar frecuencias puntuales del espectro, útil para ruido periódico.

| Control | Descripción |
|---|---|
| Centros | Coordenadas en formato `u1,v1;u2,v2` |
| Radio | Tamaño del área de rechazo (5 – 50 px) |
| Botón "Aplicar Filtro Notch" | Aplica máscara Notch simétrica |

### Sección DCT — Compresión

#### Modo: Cuantización JPEG

| Control | Descripción |
|---|---|
| q_factor | Factor de calidad (0.1 = máxima compresión, 2.0 = mínima) |
| Botón "Aplicar DCT" | Aplica cuantización en bloques 8×8 |

Resultado: vista con 2 imágenes + PSNR en dB.

```
[ Original ]  |  [ Reconstruida — PSNR: XX.XX dB ]
```

#### Modo: Top-K (extensión)

Selecciona los k coeficientes DCT de mayor magnitud por bloque 8×8.

| Control | Descripción |
|---|---|
| Checkboxes k | Valores disponibles: 5, 10, 20, 30, 40 |
| Botón "Aplicar DCT" | Genera una reconstrucción por cada k marcado |

Resultado: grid 2×3 con cada valor de k, su PSNR y ratio de compresión.

---

## Funcionalidades del Dashboard (generales)

### Historial / Carrusel

Cada resultado de FFT o DCT se agrega automáticamente al carrusel inferior.
Al hacer click en una miniatura se restaura la vista: imagen individual o multi-vista según corresponda.

### Auto-guardado

Checkbox **"Guardar automáticamente"**: al aplicar cualquier transformada frecuencial, los archivos se exportan a `resources/output/` sin necesidad de usar el botón de guardado manual.

### Guardado manual

Botón **"Guardar Imagen"** → selección de carpeta → crea subcarpeta `{nombre_imagen}_{timestamp}/`.

Opciones de guardado (checkboxes):

| Checkbox | Archivo generado |
|---|---|
| (siempre) | `{nombre}_original.png` |
| Histograma | `{nombre}_histograma.png` |
| Canales | `{nombre}_canal_R/G/B.png` o `canal_GRIS.png` |
| Conteo de objetos | `{nombre}_conteo_v4.png`, `_v8.png` |
| Multi-Vista Frecuencial | `{nombre}_multiview.png` + `_espectro.png` |

El checkbox Multi-Vista solo se activa cuando hay un resultado frecuencial vigente en pantalla.

### Paneles colapsables

Los paneles izquierdo (controles) y derecho (información) tienen botones `<` / `>` para colapsar/expandir, ampliando el área del visor central.

### Imágenes responsivas

La vista múltiple frecuencial se redistribuye automáticamente al redimensionar la ventana o mover los separadores.

---

## Dependencias

Además de las dependencias del proyecto base (`opencv-python`, `numpy`, `matplotlib`, `scipy`, `PyQt5`), esta práctica no requiere dependencias adicionales.

Los filtros FFT y la DCT se implementan con `numpy.fft` y operaciones matriciales de NumPy puro, sin librerías externas adicionales.

---

## Imágenes de Prueba

Las imágenes recomendadas provienen de la base de datos HIPR2 (University of Edinburgh):
http://homepages.inf.ed.ac.uk/rbf/HIPR2/alphlib.htm

| Imagen | Uso recomendado |
|---|---|
| `cam1.gif` | FFT comparativa de filtros (lowpass/highpass) |
| `pcb2.gif` | Filtro Notch (ruido periódico en circuito) |
| `art8.gif` | FFT con patrones geométricos claros |
| `fce5.gif` | DCT cuantización (imagen facial) |
| `ape1.gif` | DCT Top-K |
| `grass1_gris.png`, `rocks1_gris.png` | Texturas para análisis frecuencial |

---

## Valores de Referencia (PSNR)

| Rango | Calidad |
|---|---|
| > 40 dB | Excelente — pérdida imperceptible |
| 30 – 40 dB | Buena calidad |
| 20 – 30 dB | Aceptable — pérdida visible |
| < 20 dB | Baja calidad |

---

## Algoritmos Implementados (controlador_imagen.py)

### FFT

- `fft2_imagen_frecuencial(img)` — calcula FFT 2D y espectro de magnitud log-normalizado
- `crear_mascara_fft(shape, cutoff, tipo_filtro, modo)` — máscaras Ideal, Gaussiana, Butterworth
- `crear_mascara_notch(shape, centros, radio)` — máscara Notch con puntos simétricos
- `aplicar_filtro_fft_frecuencial(...)` — función principal; valida imagen monocanal y retorna `{imagen_original, espectro_magnitud, mascara, imagen_filtrada}`

### DCT

- `_dct_matrix(N=8)` / `_C8` — matriz DCT 8×8 precalculada
- `_Q_JPEG` — tabla de cuantización JPEG estándar
- `_dct_bloque_2d(b)` / `_idct_bloque_2d(D)` — transformadas por bloques 8×8
- `_pad_a_multiplo(img, N=8)` — padding a múltiplo de 8
- `_calcular_psnr(original, reconstruida)` — Peak Signal-to-Noise Ratio
- `aplicar_dct_cuantizacion_frecuencial(img, q_factor)` — cuantización JPEG por bloques
- `aplicar_dct_topk_frecuencial(img, k_values)` — selección Top-K por energía

---

## Referencias

- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
- HIPR2 — Hypermedia Image Processing Reference. University of Edinburgh.
  http://homepages.inf.ed.ac.uk/rbf/HIPR2/
- Material didáctico: Unidad 2 — Transformadas de Frecuencia, Análisis de Imágenes, ESCOM-IPN.
