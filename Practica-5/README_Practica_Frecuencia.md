# Práctica: Transformaciones en el dominio de la frecuencia (FFT y DCT)
**Carrera:** Ingeniería en Sistemas Computacionales (ISC) — **Semestre:** 7º  
**Duración:** 90 min laboratorio + 180 min trabajo autónomo 
**Docente: María Elena Cruz Meza 

> Esta práctica se alinea con tu guía *"Transformaciones en el Dominio de la Frecuencia"* (FFT y DCT), con énfasis en análisis de espectros, filtrado y compresión por bloques. Revisar el documento con la guía.  

---
## Objetivos de aprendizaje
1. Interpretar el espectro de magnitud y fase de una imagen en el dominio de la frecuencia.  
2. Diseñar y aplicar filtros pasa-bajas y pasa-altas (ideal, gaussiano, Butterworth).  
3. Implementar compresión con DCT por bloques 8×8 y evaluar calidad con PSNR.  

## Requisitos
- Python 3.9+ con: `numpy`, `matplotlib`, `Pillow` (PIL).  
- Editor/IDE o terminal para ejecutar scripts.

## Archivos
- `practica_frecuencia_ISC.py`: script principal de laboratorio.  
- Carpeta sugerida: `data/` con imágenes de prueba (.jpg/.png).  

---
## Plan de los **90 min de laboratorio**
**0–10 min (Preparación):** Presentación de objetivos, repaso rápido de FFT/DCT.  
**10–40 min (Parte A — FFT):**
- Ejecuta:  
  ```bash
  python practica_frecuencia_ISC.py --imagen data/ejemplo.jpg --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2
  ```
- Observa: imagen original, espectros de magnitud y fase, máscara y resultado filtrado.
- Variaciones: cambia `--tipo` a `highpass`; prueba `--filtro ideal` y `gaussiano`; ajusta `--cutoff`.

**40–75 min (Parte B — DCT):**
- Ejecuta:  
  ```bash
  python practica_frecuencia_ISC.py --imagen data/ejemplo.jpg --dct_q 0.5
  ```
- Compara original vs reconstruida; registra PSNR.
- Prueba distintos `--dct_q` (p.ej. 0.3, 0.6, 1.0) y discute impacto.

**75–90 min (Cierre):** Puesta en común de hallazgos y resolución de dudas.

---
## Trabajo autónomo (**180 min**)
1. **Análisis sistemático de filtrado (60 min):**
   - Para una imagen con ruido de alta frecuencia, compara filtros ideal, gaussiano y Butterworth.
   - Entrega una tabla con parámetros (`cutoff`, `orden`) y una breve interpretación del resultado.
2. **Compresión con DCT (70 min):**
   - Implementa tres niveles de cuantización (`q_factor` en 0.3, 0.6, 1.0).
   - Reporta PSNR e inserta capturas.
   - Opcional: calcula también SSIM si cuentas con librería disponible.
3. **Extensión (50 min):**
   - Implementa un **filtro notch** (rechazo de banda) para eliminar patrones periódicos.
   - Explora mantener solo los **k** coeficientes DCT de mayor energía (sin cuantización) y evalúa.

## Rúbrica de evaluación (100%)
- **Ejecución técnica (40%)**: scripts corren; uso correcto de parámetros y funciones.  
- **Análisis y justificación (35%)**: interpreta espectros, discute efectos de filtros y compresión.  
- **Calidad de evidencias (15%)**: figuras legibles, comparación clara, PSNR/SSIM (si aplica).  
- **Presentación y claridad (10%)**: redacción, organización y conclusiones.

## Entregables
- PDF o Markdown con: metodología, parámetros usados, figuras y métricas.  
- Código modificado (si hicieron extensiones) y/o cuaderno Jupyter.

## Preguntas guía
- ¿Qué patrones espaciales se revelan en el espectro que no se perciben en la imagen original?  
- ¿Qué diferencia observaste entre filtros ideal, gaussiano y Butterworth al mismo `cutoff`?  
- ¿Cómo afecta `q_factor` la relación calidad–tasa de compresión en DCT?  

---
## Consejos prácticos
- Normaliza siempre las imágenes a `[0,1]` antes de transformar.
- En FFT, recuerda que el **bajo contenido de frecuencia** está cerca del centro tras `fftshift`.
- En DCT, el desplazamiento a `b-0.5` ayuda a centrar la señal antes de transformar y mejora la reconstrucción.

¡Éxitos en la práctica!  
