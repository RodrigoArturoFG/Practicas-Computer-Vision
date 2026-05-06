@echo off
REM ========================================================================
REM Script de ejecución para Práctica 5 - Transformadas de Frecuencia
REM Rodrigo Arturo Fernández González
REM Grupo: 7CM3 - ESCOM IPN
REM ========================================================================

echo.
echo ========================================================================
echo PRACTICA 5 - TRANSFORMADAS DE FRECUENCIA
echo ========================================================================
echo.

REM La carpeta de salidas siempre será "salidas"
set carpeta_salidas=salidas
mkdir %carpeta_salidas% 2>nul

echo Resultados se guardaran en: %carpeta_salidas%
echo.

REM ========================================================================
REM PARTE A: FILTRADO FFT - COMPARACION DE FILTROS
REM ========================================================================
echo.
echo [PARTE A] ANALISIS FFT - COMPARACION DE FILTROS
echo ========================================================================

REM --- Filtro IDEAL ---
echo.
echo [1/13] Filtro Ideal - Lowpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/rocks1_gris.png --filtro ideal --tipo lowpass --cutoff 0.05 --salidas %carpeta_salidas%/01_ideal_low_005_art8

echo [3/13] Filtro Ideal - Highpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/rocks1_gris.png --filtro ideal --tipo highpass --cutoff 0.05 --salidas %carpeta_salidas%/03_ideal_high_005_art8

echo [1/13] Filtro Ideal - Lowpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/sand1_gris.png --filtro ideal --tipo lowpass --cutoff 0.05 --salidas %carpeta_salidas%/01_ideal_low_005_art8

echo [3/13] Filtro Ideal - Highpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/sand1_gris.png --filtro ideal --tipo highpass --cutoff 0.05 --salidas %carpeta_salidas%/03_ideal_high_005_art8

echo [1/13] Filtro Ideal - Lowpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/grass1_gris.png --filtro ideal --tipo lowpass --cutoff 0.05 --salidas %carpeta_salidas%/01_ideal_low_005_art8

echo [3/13] Filtro Ideal - Highpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/grass1_gris.png --filtro ideal --tipo highpass --cutoff 0.05 --salidas %carpeta_salidas%/03_ideal_high_005_art8

REM --- Filtro GAUSSIANO ---
echo [4/13] Filtro Gaussiano - Lowpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro gaussiano --tipo lowpass --cutoff 0.05 --salidas %carpeta_salidas%/04_gauss_low_005_art8

echo [5/13] Filtro Gaussiano - Lowpass cutoff=0.15 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro gaussiano --tipo lowpass --cutoff 0.15 --salidas %carpeta_salidas%/05_gauss_low_015_art8

echo [6/13] Filtro Gaussiano - Highpass cutoff=0.05 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro gaussiano --tipo highpass --cutoff 0.05 --salidas %carpeta_salidas%/06_gauss_high_005_art8

REM --- Filtro BUTTERWORTH ---
echo [7/13] Filtro Butterworth - Lowpass cutoff=0.05, orden=2 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro butterworth --tipo lowpass --cutoff 0.05 --orden 2 --salidas %carpeta_salidas%/07_butter_low_005_art8

echo [8/13] Filtro Butterworth - Lowpass cutoff=0.15, orden=2 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro butterworth --tipo lowpass --cutoff 0.15 --orden 2 --salidas %carpeta_salidas%/08_butter_low_015_art8

echo [9/13] Filtro Butterworth - Highpass cutoff=0.05, orden=2 (art8.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/art8.gif --filtro butterworth --tipo highpass --cutoff 0.05 --orden 2 --salidas %carpeta_salidas%/09_butter_high_005_art8

REM --- Pruebas con PCB (patrones periódicos) ---
echo [10/13] Filtro Ideal - Lowpass cutoff=0.1 (pcb2.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/pcb2.gif --filtro ideal --tipo lowpass --cutoff 0.1 --salidas %carpeta_salidas%/10_ideal_low_01_pcb2

echo [11/13] Filtro Gaussiano - Lowpass cutoff=0.1 (pcb2.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/pcb2.gif --filtro gaussiano --tipo lowpass --cutoff 0.1 --salidas %carpeta_salidas%/11_gauss_low_01_pcb2

echo [12/13] Filtro Butterworth - Lowpass cutoff=0.1, orden=2 (pcb2.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/pcb2.gif --filtro butterworth --tipo lowpass --cutoff 0.1 --orden 2 --salidas %carpeta_salidas%/12_butter_low_01_pcb2

REM --- Filtro NOTCH (extensión) ---
echo [13/13] Filtro Notch - Eliminar frecuencias especificas (pcb2.gif)
python scripts/practica_frecuencia_ISC.py --modo notch --imagen data/pcb2.gif --notch_centros "60,60;-60,-60" --notch_radio 30 --salidas %carpeta_salidas%/13_notch_pcb2

REM ========================================================================
REM PARTE B: COMPRESION DCT
REM ========================================================================
echo.
echo [PARTE B] COMPRESION DCT
echo ========================================================================

REM --- DCT con diferentes factores de cuantización ---
echo.
echo [14/20] DCT q_factor=0.3 (fce5.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.3 --salidas %carpeta_salidas%/14_dct_q03_fce5

echo [15/20] DCT q_factor=0.5 (fce5.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.2 --salidas %carpeta_salidas%/15_dct_q05_fce5

echo [16/20] DCT q_factor=0.8 (fce5.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.1 --salidas %carpeta_salidas%/16_dct_q08_fce5

echo [17/20] DCT q_factor=1.0 (fce5.gif)
python scripts/practica_frecuencia_ISC.py --imagen data/fce5.gif --dct_q 0.05 --salidas %carpeta_salidas%/17_dct_q10_fce5

REM --- DCT Top-K coeficientes (extensión) ---
echo [18/20] DCT Top-K comparacion (ape1.gif)
python scripts/practica_frecuencia_ISC.py --modo topk --imagen data/ape1.gif --salidas %carpeta_salidas%/18_dct_topk_ape1

REM --- DCT en diferentes tipos de imágenes ---
echo [19/20] DCT q_factor=0.05 (blb1.gif - areas suaves)
python scripts/practica_frecuencia_ISC.py --imagen data/blb1.gif --dct_q 0.05 --salidas %carpeta_salidas%/19_dct_q05_blb1

echo [20/20] DCT q_factor=0.2 (fce1.gif - comparacion rostros)
python scripts/practica_frecuencia_ISC.py --imagen data/fce1.gif --dct_q 0.2 --salidas %carpeta_salidas%/20_dct_q05_fce1

REM ========================================================================
REM GENERACION DE TABLA COMPARATIVA AUTOMATICA
REM ========================================================================
echo.
echo [EXTRA] Generando tabla comparativa automatica (art8.gif)
python scripts/practica_frecuencia_ISC.py --modo tabla_comparativa --imagen data/art8.gif --salidas %carpeta_salidas%/tabla_comparativa

REM ========================================================================
REM EXTRACCION DE METRICAS (OPCIONAL)
REM ========================================================================
echo.
echo ========================================================================
echo EXTRACCION DE METRICAS
echo ========================================================================
echo.
echo Deseas generar tabla de metricas automaticamente? (S/N)
set /p generar_tabla="Respuesta: "

if /i "%generar_tabla%"=="S" (
    echo.
    echo Generando tabla de metricas...
    python scripts/extraer_metricas.py --carpeta %carpeta_salidas%
    echo.
)

REM ========================================================================
REM FINALIZADO
REM ========================================================================
echo.
echo ========================================================================
echo TODOS LOS EXPERIMENTOS COMPLETADOS
echo ========================================================================
echo Revisa los resultados en la carpeta: %carpeta_salidas%
echo.
echo Total de experimentos ejecutados: 20
echo   - 13 experimentos FFT (filtrado)
echo   - 7 experimentos DCT (compresion)
echo   - 1 tabla comparativa automatica
echo.
echo Para generar tabla de metricas manualmente:
echo   python scripts/extraer_metricas.py --carpeta salidas
echo.
echo Presiona cualquier tecla para abrir la carpeta de resultados...
pause
explorer %carpeta_salidas%