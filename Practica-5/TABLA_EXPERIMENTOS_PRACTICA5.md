# Tabla de Experimentos - Práctica 5: Transformadas de Frecuencia

**Rodrigo Arturo Fernández González**  
**Grupo: 7CM3 - ESCOM IPN**

---

## PARTE A: FILTRADO FFT (13 experimentos)

### Tabla 1: Comparación de Filtros con cam1.gif

| # | Filtro | Parámetros | Imagen | PSNR (dB) | Observaciones Esperadas |
|---|--------|------------|--------|-----------|-------------------------|
| 1 | Ideal | Lowpass, cutoff=0.05 | cam1.gif | (calcular) | Ringing visible en bordes, pérdida severa de detalles |
| 2 | Ideal | Lowpass, cutoff=0.15 | cam1.gif | (calcular) | Menor ringing, difuminado moderado |
| 3 | Ideal | Highpass, cutoff=0.05 | cam1.gif | (calcular) | Resalta bordes, elimina componente DC, imagen oscura |
| 4 | Gaussiano | Lowpass, cutoff=0.05 | cam1.gif | (calcular) | Sin ringing, suavizado natural, imagen borrosa |
| 5 | Gaussiano | Lowpass, cutoff=0.15 | cam1.gif | (calcular) | Suavizado moderado, mejor que ideal |
| 6 | Gaussiano | Highpass, cutoff=0.05 | cam1.gif | (calcular) | Realce de bordes suave, sin artefactos |
| 7 | Butterworth | Lowpass, cutoff=0.05, n=2 | cam1.gif | (calcular) | Compromiso entre ideal y gaussiano |
| 8 | Butterworth | Lowpass, cutoff=0.15, n=2 | cam1.gif | (calcular) | Transición controlada, buen balance |
| 9 | Butterworth | Highpass, cutoff=0.05, n=2 | cam1.gif | (calcular) | Realce de detalles sin ringing excesivo |

### Tabla 2: Filtrado en pcb2.gif (patrones periódicos)

| # | Filtro | Parámetros | Imagen | PSNR (dB) | Observaciones Esperadas |
|---|--------|------------|--------|-----------|-------------------------|
| 10 | Ideal | Lowpass, cutoff=0.1 | pcb2.gif | (calcular) | Afecta patrones del circuito, ringing |
| 11 | Gaussiano | Lowpass, cutoff=0.1 | pcb2.gif | (calcular) | Preserva estructura general |
| 12 | Butterworth | Lowpass, cutoff=0.1, n=2 | pcb2.gif | (calcular) | Balance óptimo para circuitos |
| 13 | Notch | Centros=(60,60;-60,-60), r=10 | pcb2.gif | (calcular) | **EXTENSIÓN:** Elimina frecuencias específicas |

---

## PARTE B: COMPRESIÓN DCT (7 experimentos)

### Tabla 3: DCT con Cuantización Variable (fce5.gif)

| # | Método | Parámetros | Imagen | PSNR (dB) | Observaciones Esperadas |
|---|--------|------------|--------|-----------|-------------------------|
| 14 | DCT Cuantización | q_factor=0.3 | fce5.gif | (calcular) | Alta compresión, pérdida visible (efecto bloques) |
| 15 | DCT Cuantización | q_factor=0.5 | fce5.gif | (calcular) | Compresión moderada, calidad aceptable |
| 16 | DCT Cuantización | q_factor=0.8 | fce5.gif | (calcular) | Baja compresión, buena calidad |
| 17 | DCT Cuantización | q_factor=1.0 | fce5.gif | (calcular) | Referencia JPEG estándar |

### Tabla 4: DCT Top-K Coeficientes (EXTENSIÓN)

| # | Método | Parámetros | Imagen | PSNR (dB) | Observaciones Esperadas |
|---|--------|------------|--------|-----------|-------------------------|
| 18 | DCT Top-K | k=5,10,20,30,40 | ape1.gif | (varios) | **EXTENSIÓN:** Compresión por energía |

### Tabla 5: DCT en Diferentes Tipos de Imágenes

| # | Método | Parámetros | Imagen | PSNR (dB) | Observaciones Esperadas |
|---|--------|------------|--------|-----------|-------------------------|
| 19 | DCT Cuantización | q_factor=0.5 | blb1.gif | (calcular) | Áreas suaves vs bordes nítidos |
| 20 | DCT Cuantización | q_factor=0.5 | fce1.gif | (calcular) | Comparación entre rostros diferentes |

---

## RESUMEN DE EXPERIMENTOS

### Por Tipo de Análisis:
- **FFT Lowpass:** 7 experimentos (Ideal, Gaussiano, Butterworth con diferentes cutoff)
- **FFT Highpass:** 3 experimentos (uno por tipo de filtro)
- **FFT Notch (extensión):** 1 experimento
- **DCT Cuantización:** 6 experimentos (diferentes q_factor e imágenes)
- **DCT Top-K (extensión):** 1 experimento comparativo

### Por Imagen:
- **cam1.gif:** 9 experimentos (FFT completo)
- **pcb2.gif:** 4 experimentos (FFT + Notch)
- **fce5.gif:** 4 experimentos (DCT cuantización)
- **ape1.gif:** 1 experimento (DCT Top-K)
- **blb1.gif:** 1 experimento (DCT)
- **fce1.gif:** 1 experimento (DCT)

**TOTAL: 20 experimentos**

---

## INSTRUCCIONES DE USO

1. **Ejecutar el script batch:**
   ```cmd
   ejecutar_practica5.bat
   ```

2. **Esperar a que terminen todos los experimentos** (puede tomar varios minutos)

3. **Revisar la carpeta de resultados** que se crea automáticamente

4. **Llenar los valores PSNR** revisando las figuras generadas (el PSNR aparece en el título de las figuras DCT)

5. **Copiar la tabla al reporte Word** usando la plantilla

---

## NOTAS IMPORTANTES

### Para FFT:
- El **PSNR no aplica directamente** en filtrado FFT porque no hay "compresión" sino transformación
- En su lugar, documenta:
  - Cambios visuales observados
  - Presencia/ausencia de ringing
  - Calidad de preservación de bordes
  - Difuminado/suavizado

### Para DCT:
- El **PSNR SÍ es relevante** porque mide pérdida por compresión
- Valores típicos:
  - **PSNR > 40 dB:** Excelente calidad (pérdida imperceptible)
  - **PSNR 30-40 dB:** Buena calidad (pérdida apenas perceptible)
  - **PSNR 20-30 dB:** Calidad aceptable (pérdida visible)
  - **PSNR < 20 dB:** Mala calidad (pérdida severa)

### Extensiones del Trabajo Autónomo:
- **Filtro Notch (experimento #13):** Elimina frecuencias específicas
- **DCT Top-K (experimento #18):** Selección de coeficientes por energía

---

## ANÁLISIS CRÍTICO (Preguntas de la Plantilla)

Al completar los experimentos, responde en tu reporte:

1. **¿Qué información revela el espectro de una imagen?**
   - Analiza los espectros de magnitud de cam1.gif y pcb2.gif
   - Compara patrones periódicos vs naturales

2. **¿Qué tipo de filtro fue más efectivo?**
   - Compara Ideal vs Gaussiano vs Butterworth
   - Considera ringing, suavidad, preservación de detalles

3. **¿Qué ventajas ofrece la DCT en compresión?**
   - Analiza resultados con diferentes q_factor
   - Compara PSNR vs calidad visual
   - Menciona Top-K como alternativa

4. **¿Qué limitaciones observaste?**
   - Efecto de bloqueo en DCT
   - Ringing en filtros ideales
   - Pérdida de información en altas compresiones

---

**Generado para:** Práctica 5 - Análisis de Imágenes  
**Estudiante:** Rodrigo Arturo Fernández González  
**Profesora:** M. en C. María Elena Cruz Meza  
**Fecha:** Abril 2026
