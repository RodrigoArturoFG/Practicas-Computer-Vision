# --------- PRACTICA 1 "CREANDO MI MAPA DE CALOR" ---------
# --------- GUI PRINCIPAL PARA LA PRÁCTICA 1 --------------
# Autor: Rodrigo Arturo Fernández González
# Fecha: 02-19-2026

import os
import sys
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QComboBox, QStyledItemDelegate, 
    QMessageBox, QSizePolicy, QProgressDialog
)
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QGuiApplication, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# Módulos propios
from config import (
    script_dir, mapas_color, colores_pastel, colores_tierra, 
    colores_pastel_personalizados, nombre_fuente, tamano_fuente
)
from imagen_pseudocolor import ImagenPseudocolor

class WorkerApplyColormap(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(tuple)  # (imagen_gris, colormap_name)
    done = pyqtSignal() # nueva señal para indicar que el proceso ha terminado completamente

    def __init__(self, imagen_gris, colormap_name, parent=None):
        super().__init__(parent)
        self.imagen_gris = imagen_gris
        self.colormap_name = colormap_name

    def run(self):
        # Simular progreso inicial 0–30
        for p in range(0, 21, 5):
            self.progress.emit(p)
            #self.msleep(5)  # 50 ms de pausa para que se vea fluido

        # Procesamiento real
        resultado = ImagenPseudocolor.aplicar_pseudocolor(self.imagen_gris, self.colormap_name)

        # Emitir resultado
        self.finished.emit((self.imagen_gris, self.colormap_name))

        # Simular progreso final 70–100
        for p in range(21, 105, 5):
            self.progress.emit(p)
            #self.msleep(5)
        
        self.done.emit() # aquí sí se cierra el loader

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


class Practica1GUI(QMainWindow):

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def show_info(self, message: str):
        QMessageBox.information(self, "Información", message)

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

            # Ajustar márgenes manualmente (sin tamaño fijo de figura)
            self.figure.subplots_adjust(
                left=0.05, right=0.95, top=0.90, bottom=0.05,
                wspace=0.3, hspace=0.3
            )

            # Render estable
            self.canvas.draw()

            # Guardado diferido
            self.last_processed_image = resultado.imagen
            self.last_action = "pseudocolor"

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
            os.makedirs(ruta_carpeta, exist_ok=True)

            self.ruta_imagen_proc = os.path.join(ruta_carpeta, f"imagen_procesada_{colormap_name}_{timestamp}.png")
            self.ruta_imagen_comp = os.path.join(ruta_carpeta, f"comparacion_pseudocolor_{colormap_name}_{timestamp}.png")

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
            left=0.05,   # casi sin margen izquierdo
            right=0.95,  # casi sin margen derecho
            top=0.95,    # espacio justo para títulos
            bottom=0.05, # espacio justo para títulos inferiores
            wspace=0.05, # mínima separación horizontal
            hspace=0.25  # separación vertical suficiente para títulos
        )

        # Render estable
        self.canvas.draw()

        # Guardado diferido
        self.last_processed_image = None
        self.last_action = "personalizacion"

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
        os.makedirs(ruta_carpeta, exist_ok=True)
        self.ruta_imagen_comp = os.path.join(ruta_carpeta, f"mapas_color_personalizados_{timestamp}.png")
        self.ruta_imagen_proc = None

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
        self.canvas.draw()
        QApplication.processEvents()

        # Finalizar loader justo después del render completo
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        # Guardado diferido
        self.last_action = "comparacion"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
        os.makedirs(ruta_carpeta, exist_ok=True)
        self.ruta_imagen_comp = os.path.join(ruta_carpeta, f"comparacion_mapas_color_{timestamp}.png")
        self.ruta_imagen_proc = None

    def save_current_image(self):
        if self.last_action is None:
            self.show_error("Selecciona una opción válida antes de guardar la imagen.")
            return

        if self.last_action == "pseudocolor" and self.last_processed_image is not None:
            cv2.imwrite(self.ruta_imagen_proc, self.last_processed_image)
            self.figure.savefig(self.ruta_imagen_comp, bbox_inches='tight', pad_inches=0.05)
            self.show_info(f"Imagen guardada en:\n{self.ruta_imagen_proc}\nComparación guardada en:\n{self.ruta_imagen_comp}")

        elif self.last_action in ["personalizacion", "comparacion"]:
            self.figure.savefig(self.ruta_imagen_comp, bbox_inches='tight', pad_inches=0.05)
            self.show_info(f"Figura guardada en:\n{self.ruta_imagen_comp}")


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
            self.show_image()

    def show_image(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        # Definir fuente y tamaño para el título 
        ax.set_title("Imagen Original", fontname=nombre_fuente, fontsize=tamano_fuente*2)
        ax.axis("off")
        self.canvas.draw()

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

        # Loader
        self.show_loader("Aplicando mapa de color...", "Procesando Imagen")

        # Worker
        self.worker = WorkerApplyColormap(imagen_gris, colormap_name)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(lambda datos: self.mostrar_pseudocolor(*datos))
        self.worker.done.connect(self.progress_dialog.close)
        self.worker.start()

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(nombre_fuente, tamano_fuente))
    viewer = Practica1GUI()
    viewer.show()
    sys.exit(app.exec_())
