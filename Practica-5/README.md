# Práctica 5: Transformadas de Frecuencia (FFT y DCT)

---

**Institución:** ESCOM - Instituto Politécnico Nacional
**Estudiante:** Rodrigo Arturo Fernández González
**Matricula:**   2009630357
**Carrera:**     Ingeniería en Sistemas Computacionales 
**Grupo:** 7CM3  
**Asignatura:** Análisis de Imágenes  
**Profesora:** M. en C. María Elena Cruz Meza  

---

## 📅 Fecha - Abril 2026

---

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Objetivos](#-objetivos)
- [Dependencias](#-dependencias)
- [Inicio Rápido](#-inicio-rápido)
- [Experimentos](#-experimentos-incluidos)
- [Comandos Detallados](#-comandos-detallados)
- [Flujo de Trabajo](#-flujo-de-trabajo-completo)
- [Formato de Resultados](#-formato-de-resultados)
- [Tips y Troubleshooting](#-tips-y-troubleshooting)
- [Referencias](#-referencias)

---

## 📁 Estructura del Proyecto

```
Practica-5/
├── scripts/
│   ├── practica_frecuencia_ISC.py    # Script principal con extensiones
│   └── extraer_metricas.py                     # Generador de tablas CSV/MD
├── data/                                       # Imágenes de entrada (HIPR2)
│   ├── cam1.gif, pcb2.gif, fce5.gif, ape1.gif
│   └── ... (12 imágenes en total)
├── salidas/                                    # TODOS los resultados aquí
│   ├── 01_ideal_low_005_cam1/
│   │   ├── fft_filtrado.png
│   │   └── dct_reconstruccion.png
│   ├── 02_ideal_low_015_cam1/
│   ├── ... (20 subcarpetas)
│   └── tabla_resultados_YYYYMMDD_HHMM.csv     # Auto-generado
├── reporte/
│   └── practica5-TF.docx                       # Documento final
├── ejecutar_practica5.bat             # Ejecuta todo
├── TABLA_EXPERIMENTOS_PRACTICA5.md             # Referencia de experimentos
└── README.md                                   # Este archivo
```

---

## 🎯 Objetivos

1. **FFT:** Analizar contenido frecuencial y diseñar filtros (Ideal, Gaussiano, Butterworth)
2. **DCT:** Implementar compresión tipo JPEG con cuantización
3. **Métricas:** Evaluar calidad con PSNR
4. **Extensiones (Trabajo Autónomo):**
   - Filtro Notch para eliminar frecuencias específicas
   - Selección Top-K de coeficientes DCT por energía

---

## 🔧 Dependencias

```bash
pip install numpy matplotlib Pillow
```

**Nota:** Este proyecto **NO** usa OpenCV, solo NumPy, Matplotlib y Pillow (PIL).

---

## 🚀 Inicio Rápido

### **1. Ejecutar todos los experimentos (automático)**
```cmd
ejecutar_practica5.bat
```

**¿Qué hace?**
- Ejecuta 20 experimentos (13 FFT + 7 DCT)
- Guarda resultados en `salidas/` con subcarpetas organizadas
- Al final pregunta si generar tabla de métricas
- Abre la carpeta de resultados

**Tiempo estimado:** 5-10 minutos

### **2. Generar tabla de métricas**
```cmd
python scripts/extraer_metricas.py --carpeta salidas
```

**Resultado:**
- `salidas/tabla_resultados_YYYYMMDD_HHMM.csv` (para Excel)
- `salidas/tabla_resultados_YYYYMMDD_HHMM.md` (visualización)

---

## 📊 Experimentos Incluidos

### **PARTE A: Filtrado FFT (13 experimentos)**

| Imagen | Filtros | Configuraciones |
|--------|---------|-----------------|
| **cam1.gif** | Ideal, Gaussiano, Butterworth | Lowpass (cutoff=0.05, 0.15) + Highpass (cutoff=0.05) |
| **pcb2.gif** | Ideal, Gaussiano, Butterworth | Lowpass (cutoff=0.1) + **Notch** (extensión) |

**Total:** 9 experimentos con cam1.gif + 4 con pcb2.gif

### **PARTE B: Compresión DCT (7 experimentos)**

| Método | Imágenes | Parámetros |
|--------|----------|------------|
| **Cuantización JPEG** | fce5.gif | q_factor = 0.3, 0.5, 0.8, 1.0 |
| **Top-K** (extensión) | ape1.gif | k = 5, 10, 20, 30, 40 |
| **Comparación** | blb1.gif, fce1.gif | q_factor = 0.5 |

**Total:** 20 experimentos automatizados

---

## 💻 Comandos Detallados

### **Experimentos Individuales**

#### FFT - Filtro Butterworth:
```cmd
python scripts/practica_frecuencia_ISC.py --imagen data/cam1.gif --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --salidas salidas/prueba_butter
```

#### DCT - Cuantización:
```cmd
python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.5 --salidas salidas/prueba_dct
```

#### Filtro Notch (Extensión):
```cmd
python scripts/practica_frecuencia_ISC.py --modo notch --imagen data/pcb2.gif --notch_centros "60,60;-60,-60" --notch_radio 10 --salidas salidas/prueba_notch
```

#### DCT Top-K (Extensión):
```cmd
python scripts/practica_frecuencia_ISC.py --modo topk --imagen data/ape1.gif --salidas salidas/prueba_topk
```

### **Tabla Comparativa Automática**
```cmd
python scripts/practica_frecuencia_ISC.py --modo tabla_comparativa --imagen data/cam1.gif --salidas salidas/analisis
```

### **Ver Ayuda**
```cmd
python scripts/extraer_metricas.py --help
python scripts/practica_frecuencia_ISC.py --help
```

---

## 🔄 Flujo de Trabajo Completo

### **Paso 1: Preparar el Ambiente**
```cmd
cd Practica-5
# Verificar que existan las carpetas: scripts/, data/, salidas/, reporte/
```

### **Paso 2: Descargar Imágenes**
Descarga manualmente desde [HIPR2](http://homepages.inf.ed.ac.uk/rbf/HIPR2/alphlib.htm):

**Para FFT:** cam1.gif, pcb2.gif, ape1.gif, art4.gif, art8.gif, air1.gif, mon1.gif  
**Para DCT:** fce5.gif, fce1.gif, fce4.gif, ape1.gif, blb1.gif, ply1.gif

Colócalas en `data/`

### **Paso 3: Ejecutar Experimentos**
```cmd
ejecutar_practica5.bat
```
- Responde **"S"** cuando pregunte si generar tabla de métricas

### **Paso 4: Revisar Resultados**
```cmd
explorer salidas
```

Verás 20 subcarpetas, cada una con:
- `fft_filtrado.png` (visualización FFT)
- `dct_reconstruccion.png` (visualización DCT con PSNR)

### **Paso 5: Completar Tabla en Excel**
1. Abre `salidas/tabla_resultados_YYYYMMDD_HHMM.csv` en Excel
2. Para **DCT**: Copia valores PSNR de las figuras
3. Para **FFT**: Agrega observaciones visuales:
   - Ringing (sí/no)
   - Nivel de difuminado
   - Calidad de bordes

### **Paso 6: Crear Reporte Final**
Usa `Plantilla_Reporte_Practica_Frecuencia.docx`:
- Agregar portada estilo IPN/ESCOM (logos incluidos en proyecto)
- Incluir figuras seleccionadas de cada experimento
- Copiar tabla de resultados
- Responder análisis crítico:
  - ¿Qué revela el espectro?
  - ¿Qué filtro fue más efectivo?
  - ¿Ventajas de DCT?
  - ¿Limitaciones observadas?
- Escribir conclusiones

Guardar como: `reporte/practica5-TF.docx`

---

## 📈 Formato de Resultados

### **Timestamp Automático**
Formato: `YYYYMMDD_HHMM`

**Ejemplos:**
- `20260429_1430` = 29 abril 2026, 14:30
- `20260430_0915` = 30 abril 2026, 09:15

**Ventajas:**
- Ordenamiento cronológico automático
- Compatible Windows/Linux/Mac
- Sin caracteres problemáticos

### **Tabla de Métricas Generada**

La tabla CSV contiene:
- Filtro (Ideal, Gaussiano, Butterworth, DCT, Notch, Top-K)
- Parámetros (cutoff, orden, q_factor, k)
- Imagen utilizada
- PSNR (dB) - para DCT
- Observaciones - para FFT

---

## 💡 Tips y Troubleshooting

### **Si algo falla:**
```cmd
# Ver archivos generados
dir salidas /s

# Regenerar solo la tabla
python scripts/extraer_metricas.py --carpeta salidas --salida tabla_nueva
```

### **Ejecutar solo algunos experimentos:**
Edita `ejecutar_practica5.bat` y comenta con `REM` las líneas que no necesites.

### **Personalizar tabla de salida:**
```cmd
python scripts/extraer_metricas.py --carpeta salidas --salida mi_analisis
```
Genera: `mi_analisis_YYYYMMDD_HHMM.csv`

### **Valores PSNR de Referencia (DCT):**
- **PSNR > 40 dB:** Excelente (pérdida imperceptible)
- **PSNR 30-40 dB:** Buena calidad
- **PSNR 20-30 dB:** Aceptable (pérdida visible)
- **PSNR < 20 dB:** Mala calidad

### **Observaciones FFT (no usa PSNR):**
Documenta:
- Presencia de ringing (Ideal lo tiene, Gaussiano no)
- Nivel de suavizado/difuminado
- Preservación de bordes
- Artefactos visuales

---

## 🎓 Extensiones Implementadas

### **1. Filtro Notch (Rechazo de Banda)**
Elimina frecuencias específicas del espectro. Útil para:
- Ruido periódico de escaneo
- Patrones de moiré
- Interferencias en circuitos (pcb2.gif)

**Cómo funciona:**
- Define centros de rechazo en coordenadas (u, v)
- Rechaza automáticamente puntos simétricos
- Radio ajustable de rechazo

### **2. DCT Top-K Coeficientes**
Compresión adaptativa por energía:
- Preserva solo los k coeficientes de mayor magnitud
- Alternativa a cuantización uniforme
- Muestra relación energía-calidad

**Comparación:**
- k=5: Solo componentes principales
- k=10: Más detalles preservados
- k=20: Alta fidelidad
- k=64: Sin compresión (todos los coeficientes)

---

## 📚 Referencias

- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
- HIPR2 - Hypermedia Image Processing Reference. University of Edinburgh.  
  http://homepages.inf.ed.ac.uk/rbf/HIPR2/
- Material didáctico: Unidad 2 - Transformadas de Frecuencia
- Documentación adicional en: `TABLA_EXPERIMENTOS_PRACTICA5.md`

---

## 📦 Entregables

1. ✅ **Código fuente:** Scripts en `scripts/`
2. ✅ **Resultados:** 20 experimentos en `salidas/`
3. ✅ **Tabla de métricas:** CSV y Markdown auto-generados
4. ✅ **Reporte académico:** `reporte/practica5-TF.docx` con:
   - Portada IPN/ESCOM
   - Desarrollo con figuras
   - Tabla de resultados
   - Análisis crítico
   - Conclusiones

---

## 🎯 Resumen de Comandos Clave

| Acción | Comando |
|--------|---------|
| Ejecutar todos los experimentos | `ejecutar_practica5.bat` |
| Generar tabla de métricas | `python scripts/extraer_metricas.py --carpeta salidas` |
| Ver ayuda del extractor | `python scripts/extraer_metricas.py --help` |
| Experimento individual FFT | `python scripts/practica_frecuencia_ISC.py --imagen data/cam1.gif --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --salidas salidas/test` |
| Experimento individual DCT | `python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.5 --salidas salidas/test` |
| Modo Notch | `python scripts/practica_frecuencia_ISC.py --modo notch --imagen data/pcb2.gif --notch_centros "60,60;-60,-60" --salidas salidas/test` |
| Modo Top-K | `python scripts/practica_frecuencia_ISC.py --modo topk --imagen data/ape1.gif --salidas salidas/test` |

---

