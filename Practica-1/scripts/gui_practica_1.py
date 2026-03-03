# --------- PRACTICA 1 "CREANDO MI MAPA DE CALOR" ---------
# --------- GUI PRINCIPAL PARA LA PRÁCTICA 1 --------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 19-02-2026

import os
import sys
import cv2
import time
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QComboBox, QStyledItemDelegate, 
    QMessageBox, QSizePolicy, QProgressDialog, QDialog, QLabel
)
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QFont, QMovie
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# Módulos propios para la práctica
# Importar configuraciones y la clase de procesamiento de imágenes pseudocolor.
from config import (
    script_dir, mapas_color, colores_pastel, colores_tierra, 
    colores_pastel_personalizados, nombre_fuente, tamano_fuente
)
from imagen_pseudocolor import ImagenPseudocolor

# Workers para procesamiento en background con señales para actualizar la UI sin bloquearla.
# Las intente implementar para mejorar la experiencia de usuario, pero no logré integrarlos de forma óptima con el loader.
# Igualmente se separaron los procesos pesados en threads para evitar bloqueos, aunque la actualización del loader no se veía fluida.
class WorkerApplyColormap(QThread):
    finished = pyqtSignal(object)  # emitirá (img_rgb_numpy, imagen_gris, colormap_name)

    def __init__(self, imagen_gris, colormap_name, parent=None):
        super().__init__(parent)
        self.imagen_gris = imagen_gris
        self.colormap_name = colormap_name

    def run(self):
        try:
            resultado = ImagenPseudocolor.aplicar_pseudocolor(self.imagen_gris, self.colormap_name)
            img_bgr = np.ascontiguousarray(resultado.imagen, dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

            # también preparar la escala de grises aquí
            imagen_gris = np.ascontiguousarray(self.imagen_gris, dtype=np.uint8)

            self.finished.emit((img_rgb, imagen_gris, self.colormap_name))
        except Exception as e:
            self.finished.emit((None, None, str(e)))

class WorkerCustomizeColormap(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # imagen_gris

    def __init__(self, imagen_gris, parent=None):
        super().__init__(parent)
        self.imagen_gris = imagen_gris

    def run(self):
        # Progreso inicial
        self.progress.emit(30)
        # Aquí no hay cálculo pesado, solo pasamos la imagen
        self.progress.emit(70)
        self.finished.emit(self.imagen_gris)
        # Progreso final
        self.progress.emit(100)

class WorkerCompareColorMap(QThread):
    progress = pyqtSignal(int)   # porcentaje
    finished = pyqtSignal(list)

    def __init__(self, imagen_gris, parent=None):
        super().__init__(parent)
        self.imagen_gris = imagen_gris

    def run(self):
        resultados = []
        nombres_colormaps = list(mapas_color.keys())
        total = len(nombres_colormaps)
        for idx, nombre in enumerate(nombres_colormaps):
            pseudocolor = ImagenPseudocolor.aplicar_pseudocolor(self.imagen_gris, nombre)
            resultados.append((nombre, pseudocolor.imagen))
            progreso = int((idx+1)/total * 20) # hasta 20%
            self.progress.emit(progreso)  # actualizar loader
        self.finished.emit(resultados)

# Extensiones de widgets personalizados para mejorar la experiencia de usuario en la selección de colormaps.
# Para centrar el texto en el combo box y permitir abrir el desplegable al hacer clic en el área del lineEdit, no solo en la flecha.
class CenteredComboBoxDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

class ClickableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        le = self.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)
        # Instalar filtro de eventos en el lineEdit
        le.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.lineEdit() and event.type() == QEvent.MouseButtonPress:
            self.showPopup()
            return True  # consumir el evento
        return super().eventFilter(obj, event)


# clase principal de la GUI, con métodos para mostrar mensajes, loaders y resultados de procesamiento.
class Practica1GUI(QMainWindow):  

    # Métodos para mostrar mensajes de error e información al usuario utilizando QMessageBox.
    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def show_info(self, message: str):
        QMessageBox.information(self, "Información", message)

    # Loader con QProgressDialog --- Funciona bien para la comparación de mapas, 
    # en los demas casos no pude integrarlo de forma óptima ya que no se veía fluido el progreso, 
    # pero lo dejo como ejemplo de implementación de un loader modal con progreso real.
    def show_loader(self, message="Procesando...", title="Cargando"):
        # Crear el diálogo solo cuando se necesite y configurarlo para mostrar progreso real
        self.progress_dialog = QProgressDialog(message, None, 0, 100, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.show()

    def hide_loader(self):
        if self.progress_dialog is not None:
            self.progress_dialog.hide()
            self.progress_dialog = None  # liberar referencia

    # Spinner inline --- Con QMovie para mostrar un GIF animado centrado sobre el canvas.
    # No funciona de forma óptima, pero lo dejo como ejemplo de intento de integración directa con la interfaz
    def _update_spinner_geometry(self):
        # Coloca el spinner centrado sobre el canvas (ajusta márgenes si quieres)
        try:
            canvas_geo = self.canvas.geometry()
            # Hacer que el label ocupe el mismo rect del canvas
            self._inline_spinner.setGeometry(canvas_geo)
            self._inline_spinner.raise_()
        except Exception:
            pass

    def _wrap_canvas_resize(self, original_resize):
        # Devuelve una función que llama al resize original y actualiza la geometría del spinner
        def _resized(event):
            try:
                original_resize(event)
            except Exception:
                pass
            try:
                self._update_spinner_geometry()
            except Exception:
                pass
        return _resized

    def show_inline_spinner(self, gif_path: str = None):
        """
        Muestra el spinner integrado sobre la interfaz (no modal).
        gif_path: ruta al GIF; si es None, intenta usar un QMovie vacío como fallback.
        """
        # Cerrar movie previo si existe
        try:
            if self._inline_spinner_movie is not None:
                try:
                    self._inline_spinner_movie.stop()
                except Exception:
                    pass
                self._inline_spinner_movie = None
        except Exception:
            pass

        # Cargar QMovie
        if gif_path:
            movie = QMovie(gif_path)
        else:
            movie = QMovie()  # fallback; reemplaza con ruta si no tienes recursos

        # Si el movie no es válido, mostrar texto simple
        if not movie.isValid():
            self._inline_spinner.setText("Procesando...")
            self._inline_spinner.setStyleSheet("color: #ffffff; background: rgba(0,0,0,40%);")
        else:
             # después de crear el QMovie
            movie.setCacheMode(QMovie.CacheAll)   # mejora rendimiento de lectura
            movie.setSpeed(180)                   # 180% de la velocidad original
            self._inline_spinner.setText("")
            self._inline_spinner.setStyleSheet("background: rgba(0,0,0,0%);")
            self._inline_spinner.setMovie(movie)
            self._inline_spinner_movie = movie
            movie.start()

        self._update_spinner_geometry()
        self._inline_spinner.setVisible(True)
        QApplication.processEvents()  # asegurar que la animación comience

    def hide_inline_spinner(self):
        """Detiene y oculta el spinner integrado."""
        try:
            if self._inline_spinner_movie is not None:
                try:
                    self._inline_spinner_movie.stop()
                except Exception:
                    pass
                self._inline_spinner_movie = None
            self._inline_spinner.setVisible(False)
            self._inline_spinner.setMovie(None)
            self._inline_spinner.setText("")
        except Exception:
            pass
        QApplication.processEvents()
    
    # Inicialización de la GUI, configuración de la ventana, botones y canvas.
    # Estado interno para manejo de imágenes procesadas y acciones realizadas.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practica 1 - Menú Principal (GUI)")
        # Obtener tamaño de la pantalla principal
        screen = QGuiApplication.primaryScreen().availableGeometry()
        # screen = screen.geometry() # Incluye la barra de tareas

        # Ajustar ventana al tamaño de la pantalla para evitar superposición con la barra de tareas
        self.setGeometry(screen.x(), screen.y(), screen.width()-1000, screen.height()-120)

        # Centrar ventana
        center_point = screen.center()
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(center_point)
        self.move(frame_geom.topLeft())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Estado interno para el canvas / handlers
        self._ax_left = None
        self._ax_right = None
        self._im_left = None
        self._im_right = None
        self._last_draw_cid = None

        # Spinner inline (oculto por defecto)
        self._inline_spinner = QLabel(self.central_widget)
        self._inline_spinner.setAttribute(Qt.WA_TranslucentBackground, True)
        self._inline_spinner.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._inline_spinner.setAlignment(Qt.AlignCenter)
        self._inline_spinner.setStyleSheet("background: rgba(0,0,0,0%);")  # totalmente transparente
        self._inline_spinner.setVisible(False)
        self._inline_spinner_movie = None

        # Posicionar y tamaño: ocupar la zona del canvas para centrar el spinner sobre la figura
        # Ajusta el geometry si tu layout cambia; aquí lo hacemos relativo al canvas
        self._update_spinner_geometry()
        # Conectar redimensionado para mantenerlo centrado
        self.resizeEvent = self._wrap_canvas_resize(self.resizeEvent)

        self.btn_select_image = QPushButton("1. Seleccionar Imagen a Procesar")
        self.btn_select_image.clicked.connect(self.select_image)
        self.layout.addWidget(self.btn_select_image)

        self.btn_apply_colormap = QPushButton("2. Aplicar un Mapa de Color a la Imagen en Escala de Grises")
        self.btn_apply_colormap.clicked.connect(self.apply_colormap_menu)
        self.layout.addWidget(self.btn_apply_colormap)

        self.colormap_combo = ClickableComboBox()
        self.colormap_combo.addItems(list(mapas_color.keys()))
        self.colormap_combo.setPlaceholderText("Selecciona un mapa de color")
        self.colormap_combo.setItemDelegate(CenteredComboBoxDelegate(self.colormap_combo))

        # Mostrar todos los colormaps disponibles en el desplegable
        self.colormap_combo.setMaxVisibleItems(len(mapas_color))
        self.layout.addWidget(self.colormap_combo)

        self.btn_customize_colormap = QPushButton("3. Personalización del Mapa de Color")
        self.btn_customize_colormap.clicked.connect(self.customize_colormap)
        self.layout.addWidget(self.btn_customize_colormap)

        self.btn_compare_colormaps = QPushButton("4. Comparación Visual de Mapas de Color Disponibles en OpenCV")
        self.btn_compare_colormaps.clicked.connect(self.compare_colormaps)
        self.layout.addWidget(self.btn_compare_colormaps)

        self.btn_exit = QPushButton("5. Salir del Programa")
        self.btn_exit.clicked.connect(self.close)
        self.layout.addWidget(self.btn_exit)

        # Figura de matplotlib (canvas)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.canvas)

        self.image = None
        self.img_rgb = None
        self.imagen_path = None
        self.last_action = None # trackea la última opción seleccionada 
        self.last_processed_image = None # trackea la última imagen procesada (OpenCV) para guardado diferido

        # Botón para guardar la imagen procesada o la figura actual
        self.btn_save_image = QPushButton("Guardar Imagen Procesada")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.layout.addWidget(self.btn_save_image)

        # Inicialmente no creamos el diálogo
        self.progress_dialog = None 
        self.worker = None

    # Metodo para seleccionar una imagen utilizando QFileDialog, cargarla con OpenCV y mostrarla en el canvas.
    def select_image(self):
        initial_dir = os.path.join(script_dir, "resources", "input")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", initial_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.imagen_path = file_path
            self.image = cv2.imread(file_path)
            if self.image is None:
                self.show_error("Error al cargar la imagen. Asegúrate de seleccionar un archivo de imagen válido.")
                return
            self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Resetear estado del canvas al cargar nueva imagen (una sola vez)
            self._reset_canvas_state()
            self.last_processed_image = None
            self.last_action = None

            # Mostrar la imagen original en el canvas
            self.show_image()

    # Metodo para mostrar la imagen original en el canvas, con título y sin ejes, 
    # utilizando la configuración de fuente definida en config.py para mantener una apariencia consistente.
    def show_image(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        # Definir fuente y tamaño para el título 
        ax.set_title("Imagen Original", fontname=nombre_fuente, fontsize=tamano_fuente*2)
        ax.axis("off")
        self.canvas.draw_idle()

    # Métodos para mostrar resultados de procesamiento en el canvas, con manejo de errores y actualización del estado interno para guardado diferido.
    def mostrar_pseudocolor(self, imagen_gris, colormap_name):
        try:
            resultado = ImagenPseudocolor.aplicar_pseudocolor(imagen_gris, colormap_name)

            # Limpiar figura y retícula
            self.figure.clear()
            axs = self.figure.subplots(1, 2)

            # Imagen en escala de grises
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Imagen en escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            axs[0].axis('off')
            axs[0].set_aspect('equal')

            # Imagen pseudocolor
            axs[1].imshow(cv2.cvtColor(resultado.imagen, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f'Pseudocolor: {colormap_name}', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            axs[1].axis('off')
            axs[1].set_aspect('equal')

            # Maximizar uso de pantalla con márgenes mínimos
            self.figure.subplots_adjust(
                left=0.01,   # margen mínimo izquierdo
                right=0.99,  # margen mínimo derecho
                top=0.92,    # espacio justo para títulos
                bottom=0.05, # espacio inferior
                wspace=0.02, # separación horizontal mínima
                hspace=0.05  # separación vertical mínima
            )

            # Render estable
            self.canvas.draw_idle()

            # Guardado diferido
            self.last_processed_image = resultado.imagen
            self.last_action = "pseudocolor"

        except ValueError as e:
            self.show_error(str(e))

    def mostrar_personalizacion_mapas_gui(self, imagen_gris):
        # Crear colormaps personalizados
        mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
        mapa_tierra = LinearSegmentedColormap.from_list("TierraMap", colores_tierra, N=256)
        mapa_pastel_personalizado = LinearSegmentedColormap.from_list("PastelPersonalizadoMap", colores_pastel_personalizados, N=256)

        # Limpiar figura y retícula
        self.figure.clear()
        axs = self.figure.subplots(2, 2).flatten()

        # Imagen en escala de grises
        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Imagen en escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[0].axis('off')
        axs[0].set_aspect('equal')

        # Mapa pastel
        axs[1].imshow(imagen_gris, cmap=mapa_pastel)
        axs[1].set_title('Mapa de color pastel', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[1].axis('off')
        axs[1].set_aspect('equal')

        # Mapa tierra
        axs[2].imshow(imagen_gris, cmap=mapa_tierra)
        axs[2].set_title('Mapa de color tierra', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[2].axis('off')
        axs[2].set_aspect('equal')

        # Mapa pastel personalizado
        axs[3].imshow(imagen_gris, cmap=mapa_pastel_personalizado)
        axs[3].set_title('Mapa de color pastel personalizado', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[3].axis('off')
        axs[3].set_aspect('equal')

        # Maximizar uso de pantalla con márgenes mínimos
        self.figure.subplots_adjust(
            left=0.01,   # margen mínimo izquierdo
            right=0.99,  # margen mínimo derecho
            top=0.92,    # espacio justo para títulos
            bottom=0.05, # espacio inferior
            wspace=0.02, # separación horizontal mínima
            hspace=0.25  # separación vertical suficiente para títulos
        )

        # Render estable
        self.canvas.draw_idle()

        # Guardado diferido
        self.last_processed_image = None
        self.last_action = "personalizacion"

    def mostrar_comparacion_mapas_gui(self, resultados):
        total_imgs = len(resultados) + 1
        n_cols = min(5, total_imgs)
        n_rows = int(np.ceil(total_imgs / n_cols))

        self.figure.clear()
        axs = self.figure.subplots(n_rows, n_cols).reshape(-1)

        # Imagen original
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[0].axis('off')

        # Dibujar resultados con progreso 30–95
        total = len(resultados)
        for idx, (nombre, imagen) in enumerate(resultados):
            axs[idx+1].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            axs[idx+1].set_title(nombre, fontname=nombre_fuente, fontsize=tamano_fuente*2)
            axs[idx+1].axis('off')

            # Actualizar progreso paso a paso
            progreso = 20 + int((idx+1)/total * 75)  # hasta ~95%
            self.progress_dialog.setValue(progreso)
            QApplication.processEvents()

            # Dibujar en bloques intermedios con draw_idle
            if (idx+1) % n_cols == 0 and (idx+1) != total:
                self.canvas.draw_idle()
                QApplication.processEvents()

        # Desactivar ejes sobrantes
        for ax in axs[total_imgs:]:
            ax.axis('off')

        # Ajustar layout al final
        self.figure.tight_layout(rect=[0, 0, 1, 1])

        # Render final síncrono
        self.canvas.draw_idle()
        QApplication.processEvents()

        # Finalizar loader justo después del render completo
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        # Guardado diferido
        self.last_action = "comparacion"

    # Método para guardar la imagen procesada o la figura actual dependiendo de la última acción realizada, 
    # con un diálogo para seleccionar la carpeta de destino y nombrado automático con timestamp.
    def save_current_image(self):
        if self.last_action is None:
            self.show_error("Selecciona una opción válida antes de guardar la imagen.")
            return

        # Carpeta inicial de salida en resources
        carpeta_salida = os.path.join(script_dir, "resources\\output")
        os.makedirs(carpeta_salida, exist_ok=True)

        # Abrir selector de carpeta empezando en resources
        carpeta_elegida = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de salida", carpeta_salida
        )
        if not carpeta_elegida:
            return  # si cancela, no guardamos nada

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.last_action == "pseudocolor" and self.last_processed_image is not None:
            # Usar el nombre del colormap en el archivo 
            colormap_name = self.colormap_combo.currentText()
            ruta_proc = os.path.join(carpeta_elegida, f"imagen_pseudocolor_{colormap_name}_{timestamp}.png")
            ruta_comp = os.path.join(carpeta_elegida, f"comparacion_pseudocolor_{colormap_name}_{timestamp}.png")
            cv2.imwrite(ruta_proc, self.last_processed_image)
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

        elif self.last_action == "personalizacion":
            ruta_comp = os.path.join(carpeta_elegida, f"mapas_color_personalizados_{timestamp}.png")
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

        elif self.last_action == "comparacion":
            ruta_comp = os.path.join(carpeta_elegida, f"comparacion_mapas_color_{timestamp}.png")
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

    # Método para resetear el estado del canvas antes de mostrar una nueva imagen, desconectando handlers, 
    # limpiando ejes y forzando un redraw vacío para evitar superposiciones o errores al mostrar nuevas imágenes.
    def _reset_canvas_state(self):
        """
        Limpia el estado interno usado para actualizar el canvas.
        Desconecta handlers, borra ejes y fuerza un redraw vacío.
        """
        try:
            # Desconectar handler de draw_event si existe
            try:
                if getattr(self, "_last_draw_cid", None) is not None:
                    try:
                        self.canvas.mpl_disconnect(self._last_draw_cid)
                    except Exception:
                        pass
                    self._last_draw_cid = None
            except Exception:
                self._last_draw_cid = None

            # Limpiar referencias a ejes e imágenes
            try:
                self._ax_left = None
                self._ax_right = None
                self._im_left = None
                self._im_right = None
            except Exception:
                pass

            # Borrar la figura y forzar un redraw vacío (no bloqueante)
            try:
                self.figure.clear()
                self.canvas.draw_idle()
                QApplication.processEvents()
            except Exception:
                try:
                    # fallback seguro
                    self.figure = Figure()
                    self.canvas.figure = self.figure
                    self.canvas.draw()
                except Exception:
                    pass
        except Exception:
            # Silenciar cualquier error para no romper la UI
            pass

    # Manejo de opciones principales del menú, con validación de estado, carga de imagen en escala de grises,
    # y uso de workers para procesamiento en background con actualización del canvas al finalizar.
    def apply_colormap_menu(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return
        colormap_name = self.colormap_combo.currentText()
        if not colormap_name:
            self.show_error("Selecciona un mapa de color.")
            return

        # Mostrar spinner antes de iniciar el worker
        # Carpeta inicial de salida en resources
        spinner_path = os.path.join(script_dir, "resources\\gui\\liquid_dot_loader.gif")

        # print("T1 show spinner", time.time())
        self.show_inline_spinner(spinner_path)
        # print("T2 after show spinner", time.time())

        QApplication.processEvents()  # forzar inicio de la animación del GIF

        # Limpieza segura del worker anterior (si existe)
        try:
            prev = getattr(self, "worker", None)
            if prev is not None:
                # Si el worker ya terminó, desconectar y soltar la referencia
                if not prev.isRunning():
                    try:
                        prev.finished.disconnect()
                    except Exception:
                        pass
                    try:
                        prev.done.disconnect()
                    except Exception:
                        pass
                    self.worker = None
                else:
                    # Si sigue corriendo, desconectamos señales para que no afecte la UI
                    # pero no forzamos su terminación (puede estar en llamada bloqueante)
                    try:
                        prev.finished.disconnect()
                    except Exception:
                        pass
                    try:
                        prev.done.disconnect()
                    except Exception:
                        pass
                    # dejamos prev corriendo en background; la señal antigua ya no está conectada
                    # y el handler comprobará el sender() para ignorar emisiones antiguas
                    self.worker = None
        except Exception:
            self.worker = None

        self.worker = WorkerApplyColormap(imagen_gris, colormap_name)
        self.worker.finished.connect(self._on_worker_finished_update_canvas)
        # print("T3 before worker start", time.time())
        self.worker.start()
        # print("T4 after worker start", time.time())

    def customize_colormap(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return

        # Loader
        self.show_loader("Generando mapas personalizados...", "Procesando Imagen")

        # Worker
        self.worker = WorkerCustomizeColormap(imagen_gris)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.mostrar_personalizacion_mapas_gui)
        self.worker.finished.connect(self.progress_dialog.close)
        self.worker.start()

    def compare_colormaps(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return

        # Crear loader con rango definido para mostrar progreso real
        self.show_loader("Comparando mapas de color...", "Procesando Imágenes")

        # Crear worker
        self.worker = WorkerCompareColorMap(imagen_gris)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.mostrar_comparacion_mapas_gui)
        self.worker.start()

    # mostrar pseudocolor con worker para no bloquear la UI, y actualizar el canvas al finalizar el procesamiento.
    def _on_worker_finished_update_canvas(self, datos):
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return

        img_rgb, imagen_gris, colormap_name = datos
        if img_rgb is None:
            self.hide_inline_spinner()
            return

        h, w = img_rgb.shape[:2]

        # Crear/recrear ejes si no existen o cambió la forma
        recreate = False
        if not getattr(self, "_ax_left", None):
            recreate = True
        else:
            try:
                existing_shape = self._im_right.get_array().shape
                if existing_shape[:2] != (h, w):
                    recreate = True
            except Exception:
                recreate = True

        if recreate:
            self.figure.clear()
            axs = self.figure.subplots(1, 2)
            self._ax_left, self._ax_right = axs[0], axs[1]
            self._im_left = self._ax_left.imshow(np.zeros((h, w)), cmap='gray', vmin=0, vmax=255)
            self._im_right = self._ax_right.imshow(np.zeros((h, w, 3), dtype=np.uint8))
            self._ax_left.set_title('Imagen en escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            self._ax_left.axis('off')
            self._ax_right.set_title(f'Pseudocolor: {colormap_name}', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            self._ax_right.axis('off')
        else:
            self._ax_right.set_title(f'Pseudocolor: {colormap_name}', fontname=nombre_fuente, fontsize=tamano_fuente*2)

        # Actualizar datos (ya listos desde el worker)
        self._im_left.set_data(imagen_gris)
        self._im_left.set_clim(0, 255)
        self._ax_left.set_aspect('equal')

        self._im_right.set_data(img_rgb)
        self._ax_right.set_aspect('equal')

        # Ajustes de layout
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.05, wspace=0.02, hspace=0.05)

        # Ocultar spinner cuando el canvas termine de dibujar
        def _on_draw(event):
            self.hide_inline_spinner()
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

        cid = self.canvas.mpl_connect('draw_event', _on_draw)

        self.canvas.draw_idle()

        # Guardado diferido
        self.last_processed_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # si quieres guardar en BGR
        self.last_action = "pseudocolor"

        if sender is self.worker:
            self.worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(nombre_fuente, tamano_fuente))
    viewer = Practica1GUI()
    viewer.show()
    sys.exit(app.exec_())
