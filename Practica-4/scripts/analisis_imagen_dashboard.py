# =====================================================================
# PRÁCTICA 4 - "MM BINARIA Y EN LATTICES"
# VERSIÓN 3: Layout 3 Paneles (Dashboard)
# Autor: Rodrigo Arturo Fernández González
# Fecha: 01-04-2026
# UI desarrollada con PyQt5
# =====================================================================

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QScrollArea, QHBoxLayout, QVBoxLayout,
    QFileDialog, QFrame, QSplitter, QSizePolicy, QGroupBox,
    QSpacerItem, QMessageBox, QTabWidget, QGridLayout, QStackedWidget,
    QStyledItemDelegate, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QRect, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Importar lógica existente ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config
    import controlador_imagen as procesadorImagen
    from modelo_imagen import metadataImagen
    from modelo_historial_imagen import metadataHistorialImagen
except ImportError:
    # Stub de config para preview sin los módulos
    class config:
        script_dir_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    class metadataImagen:
        def __init__(self, ruta=""):
            self.ruta = ruta
            self.nombre = os.path.basename(ruta) if ruta else "sin_imagen.jpg"
            self.modelo = "RGB"
            self.datos = None
            self.umbral = None
            self.histograma = {}
    class metadataHistorialImagen:
        def __init__(self, ruta=""):
            self.ruta = ruta
            self.nombre = os.path.basename(ruta) if ruta else "sin_imagen.jpg"
            self.modelo = "RGB"
            self.thumbnail = None
            self.umbral = None
            self.histograma = {}
    class procesadorImagen:
        @staticmethod
        def cargar_imagen_opencv_rgb(m): return {"objeto": m, "error": True, "mensaje": "Módulo no encontrado"}


# ══════════════════════════════════════════════════════════════
#  PALETA  —  Tema "Terminal Verde / Laboratorio"
# ══════════════════════════════════════════════════════════════
C = {
    "bg":           "#080C10",
    "surface":      "#0D1117",
    "surface2":     "#131A22",
    "surface3":     "#192030",
    "border":       "#1E2D3D",
    "border2":      "#2A4060",
    "accent":       "#00D4AA",       # verde-teal
    "accent2":      "#0099CC",       # azul
    "accent_dim":   "#004433",
    "accent2_dim":  "#003355",
    "text":         "#CDD9E5",
    "text2":        "#6B8FA8",
    "text3":        "#2D4A5E",
    "warn":         "#E8A838",
    "danger":       "#FF4455",
    "success":      "#00D4AA",
    "ch_r":         "#FF5555",
    "ch_g":         "#50FA7B",
    "ch_b":         "#8BE9FD",
}

SS = f"""
* {{ font-family: 'Consolas', 'Courier New', monospace; font-size: 13pt; }}
QMainWindow, QWidget {{ background: {C['bg']}; color: {C['text']}; font-size: 13pt; }}

/* ── Sidebar ─────────────────────────────── */
QWidget#sidebar {{
    background: {C['surface']};
    border-right: 1px solid {C['border']};
}}

/* ── Sección de sidebar ─────────────────── */
QWidget#section {{
    background: {C['surface2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
}}

/* ── Botones ─────────────────────────────── */
QPushButton {{
    background: {C['surface3']};
    color: {C['text']};
    border: 1px solid {C['border2']};
    border-radius: 4px;
    padding: 10px 18px;
    text-align: left;
    font-size: 12pt;
}}
QPushButton:hover {{
    background: {C['accent_dim']};
    border-color: {C['accent']};
    color: {C['accent']};
}}
QPushButton:pressed {{ background: {C['accent']}; color: {C['bg']}; }}

QPushButton#primary {{
    background: {C['accent_dim']};
    border: 1px solid {C['accent']};
    color: {C['accent']};
    text-align: center;
    font-size: 13pt;
    font-weight: bold;
    padding: 13px;
}}
QPushButton#primary:hover {{ background: {C['accent']}; color: #ffffff; }}

QPushButton#secondary {{
    background: {C['accent2_dim']};
    border: 1px solid {C['accent2']};
    color: {C['accent2']};
    text-align: center;
    font-size: 12pt;
    padding: 10px;
}}
QPushButton#secondary:hover {{ background: {C['accent2']}; color: #ffffff; }}

QPushButton#danger {{
    background: transparent;
    border: 1px solid {C['danger']};
    color: {C['danger']};
    text-align: center;
    font-size: 12pt;
    padding: 10px;
}}
QPushButton#danger:hover {{ background: {C['danger']}; color: white; }}

/* ── ComboBox ─────────────────────────────── */
QComboBox {{
    background: {C['surface3']};
    color: {C['text']};
    border: 1px solid {C['border2']};
    border-radius: 4px;
    padding: 4px 14px;
    font-size: 12pt;
    min-height: 28px;
}}
QComboBox:focus {{ border-color: {C['accent']}; }}
QComboBox QAbstractItemView {{
    background: {C['surface2']};
    color: {C['text']};
    border: 1px solid {C['accent']};
    selection-background-color: {C['accent_dim']};
    selection-color: {C['accent']};
    outline: none;
    padding: 4px 0px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    width: 10px; height: 10px;
}}

/* ── Slider ─────────────────────────────── */
QSlider::groove:horizontal {{
    height: 3px; background: {C['border2']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['accent']}; border: none;
    width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}

/* ── Scroll ─────────────────────────────── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {C['surface']}; width: 6px; border-radius: 3px; }}
QScrollBar:horizontal {{ background: {C['surface']}; height: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {C['border2']}; border-radius: 3px; min-height: 20px; min-width: 20px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ background: none; border: none; }}

/* ── Tabs ─────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {C['border']};
    background: {C['surface']};
    border-radius: 0 4px 4px 4px;
}}
QTabBar::tab {{
    background: {C['surface2']};
    color: {C['text2']};
    border: 1px solid {C['border']};
    border-bottom: none;
    padding: 6px 16px;
    font-size: 11px;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {C['surface']};
    color: {C['accent']};
    border-color: {C['border2']};
    border-bottom: 1px solid {C['surface']};
}}

/* ── Labels especiales ─────────────────── */
QLabel#section_title {{
    color: {C['accent']};
    font-size: 10pt;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 0 0 4px 0;
}}
QLabel#chip {{
    background: {C['accent_dim']};
    color: {C['accent']};
    border: 1px solid {C['accent']};
    border-radius: 3px;
    padding: 2px 9px;
    font-size: 11pt;
    font-weight: bold;
}}
QLabel#chip_b {{
    background: {C['accent2_dim']};
    color: {C['accent2']};
    border: 1px solid {C['accent2']};
    border-radius: 3px;
    padding: 2px 9px;
    font-size: 11pt;
    font-weight: bold;
}}
QLabel#info {{
    color: {C['text2']};
    font-size: 11pt;
    padding: 2px 0;
}}
QLabel#warn {{
    color: {C['warn']};
    font-size: 11pt;
}}
QLabel#stat_key {{
    color: {C['text2']};
    font-size: 11pt;
}}
QLabel#stat_val {{
    color: {C['accent']};
    font-size: 12pt;
    font-weight: bold;
}}
QLabel#canal_header {{
    color: {C['text']};
    font-size: 12pt;
    font-weight: bold;
    padding: 6px 0 2px 0;
    border-bottom: 1px solid {C['border']};
}}
QFrame#hline {{
    background: {C['border']};
    max-height: 1px;
    min-height: 1px;
}}
QFrame#vline {{
    background: {C['border']};
    max-width: 1px;
    min-width: 1px;
}}
"""


# ══════════════════════════════════════════════════════════════
#  DELEGATE  —  Resalta ítem seleccionado en ComboBox dropdown
# ══════════════════════════════════════════════════════════════
class DarkTextDelegate(QStyledItemDelegate):
    """
    Pinta cada ítem del dropdown con texto claro sobre fondo oscuro,
    coherente con el tema general del dashboard.
    El ítem actualmente seleccionado se resalta con el color accent.
    """
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # Texto siempre claro — visible sobre surface2
        option.palette.setColor(option.palette.Text, QColor(C["text"]))
        # Texto resaltado en accent cuando el mouse pasa por encima
        option.palette.setColor(option.palette.HighlightedText, QColor(C["accent"]))


class CenteredComboBox(QComboBox):
    """
    QComboBox con texto centrado en Windows/PyQt5.
    text-align:center en stylesheet no funciona con el renderizador nativo de Windows,
    por eso se sobreescribe paintEvent para dibujar el texto manualmente centrado.
    """
    def paintEvent(self, _event):
        from PyQt5.QtWidgets import QStylePainter, QStyleOptionComboBox, QStyle
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        # Dibuja el control completo excepto el texto
        opt.currentText = ""
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)

        # Dibuja el texto centrado manualmente
        painter.setPen(QColor(C["text"]))
        font = self.font()
        painter.setFont(font)
        # Área disponible: rect del widget menos el espacio del botón desplegable
        text_rect = self.style().subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self
        )
        painter.drawText(text_rect, Qt.AlignCenter, self.currentText())


# ══════════════════════════════════════════════════════════════
#  WIDGET MINIATURA  (thumbnail clickable del carrusel)
# ══════════════════════════════════════════════════════════════
class ThumbWidget(QWidget):
    """Contenedor clickable para cada entrada del carrusel de historial."""
    clicked = pyqtSignal(int)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)


# ══════════════════════════════════════════════════════════════
#  WIDGET MÉTRICA  (tarjeta pequeña con un valor)
# ══════════════════════════════════════════════════════════════
class MetricCard(QWidget):
    def __init__(self, titulo, valor="—", color=None, parent=None):
        super().__init__(parent)
        self.color = color or C["accent"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.lbl_titulo = QLabel(titulo)
        self.lbl_titulo.setStyleSheet(f"color:{C['text2']}; font-size:10pt; letter-spacing:1px;")
        self.lbl_valor = QLabel(valor)
        self.lbl_valor.setStyleSheet(f"color:{self.color}; font-size:18pt; font-weight:bold;")

        layout.addWidget(self.lbl_titulo)
        layout.addWidget(self.lbl_valor)
        self.setStyleSheet(f"""
            background:{C['surface2']};
            border:1px solid {C['border']};
            border-left: 3px solid {self.color};
            border-radius:4px;
        """)

    def set_valor(self, v):
        self.lbl_valor.setText(v)


# ══════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL  –  3 PANELES
# ══════════════════════════════════════════════════════════════
class VentanaDashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.imagen_metadata   = None
        self.imagen_metadata_b = None  # Imagen secundaria persistente para operaciones entre dos imágenes
        self._datos_originales = None
        # Base de binarización: guarda datos+modelo ANTES de binarizar.
        # Se actualiza dentro de binarizar_manual/otsu cuando la imagen NO es binaria.
        # Se limpia (None) cuando operaciones lógicas producen un resultado binario,
        # para preservar ese resultado ante movimientos del slider.
        self._base_bin_datos  = None
        self._base_bin_modelo = None
        self._fig_conteo      = None  # Figura matplotlib de comparación vecindad-4 vs 8
        self.historial_estados = []  # Lista de metadataHistorialImagen
        self._inicializar_cache()
        self._build_ui()
        self.setStyleSheet(SS)
        self.setWindowTitle("Vision Lab  ·  Análisis de Imágenes  ·  ESCOM-IPN")
        self.setMinimumSize(1000, 800)

    # ──────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_topbar())

        # Tres paneles con splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C['border']}; }}")
        splitter.addWidget(self._make_sidebar())
        splitter.addWidget(self._make_center())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([380, 1050, 500])
        root.addWidget(splitter, 1)

        root.addWidget(self._make_statusbar())

    # ── Top Bar ──────────────────────────────────
    def _make_topbar(self):
        w = QWidget()
        w.setFixedHeight(60)
        w.setStyleSheet(f"background:{C['surface']};")
        layout = QHBoxLayout(w)
        layout.setContentsMargins(20, 0, 20, 0)

        dot_green = QLabel("◉")
        dot_green.setStyleSheet(f"color:{C['accent']}; font-size:14px;")

        titulo = QLabel("VISION LAB")
        titulo.setStyleSheet(f"color:{C['text']}; font-size:18pt; font-weight:bold; letter-spacing:4px;")

        sep = QLabel("  ·  ")
        sep.setStyleSheet(f"color:{C['text3']};")

        subtitulo = QLabel("Explorador de Imagen Digital")
        subtitulo.setStyleSheet(f"color:{C['text2']}; font-size:12pt;")

        self.chip_modelo = QLabel("RGB")
        self.chip_modelo.setObjectName("chip")

        self.chip_archivo = QLabel("sin archivo")
        self.chip_archivo.setObjectName("chip_b")

        lbl_modelo = QLabel("MODELO:")
        lbl_modelo.setStyleSheet(f"color:{C['text2']}; font-size:11pt;")

        layout.addWidget(dot_green)
        layout.addSpacing(8)
        layout.addWidget(titulo)
        layout.addWidget(sep)
        layout.addWidget(subtitulo)
        layout.addStretch()
        layout.addWidget(lbl_modelo)
        layout.addSpacing(6)
        layout.addWidget(self.chip_modelo)
        layout.addSpacing(12)
        layout.addWidget(self.chip_archivo)

        # Contenedor con separador inferior real (ocupa todo el ancho)
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(w)
        sep_line = QFrame()
        sep_line.setObjectName("hline")
        outer_layout.addWidget(sep_line)
        return outer

    # ── Sidebar (panel izquierdo) ─────────────────
    def _make_sidebar(self):
        w = QWidget()
        w.setObjectName("sidebar")
        w.setFixedWidth(380)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Tabs: Preprocesamiento / Segmentación ──
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: {C['surface']};
            }}
            QTabBar::tab {{
                background: {C['surface2']};
                color: {C['text2']};
                border: 1px solid {C['border']};
                border-bottom: none;
                padding: 8px 0px;
                font-size: 10pt;
                letter-spacing: 1px;
                margin-right: 2px;
                border-radius: 4px 4px 0 0;
                min-width: 185px;
            }}
            QTabBar::tab:selected {{
                background: {C['surface']};
                color: {C['accent']};
                border-color: {C['border2']};
                border-bottom: 1px solid {C['surface']};
            }}
            QTabBar::tab:hover:!selected {{
                color: {C['text']};
                background: {C['surface3']};
            }}
        """)
        tabs.addTab(self._make_tab_preprocesamiento(), "PREPROCESAMIENTO")
        tabs.addTab(self._make_tab_segmentacion(),     "SEGMENTACIÓN")
        tabs.addTab(self._make_tab_morfologia(),       "MORFOLOGÍA")
        outer.addWidget(tabs, 1)

        # ── Footer fijo: Guardar imagen ─────────────
        outer.addWidget(self._make_footer_guardar())

        return w

    def _make_tab_preprocesamiento(self):
        """Contenido de la pestaña Preprocesamiento: archivo, modelo, canales, binarización."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border:none; background:transparent;")
        inner_w = QWidget()
        inner_w.setObjectName("sidebar_inner")
        inner_w.setStyleSheet(f"QWidget#sidebar_inner {{ background: {C['surface']}; }}")
        layout = QVBoxLayout(inner_w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(inner_w)

        # ── 1. Cargar imagen ────────────────────
        layout.addWidget(self._section_label("01  ·  ARCHIVO"))
        btn_cargar = QPushButton("⊕  Cargar Imagen...")
        btn_cargar.setObjectName("primary")
        btn_cargar.setCursor(Qt.PointingHandCursor)
        btn_cargar.clicked.connect(self.seleccionar_imagen)
        layout.addWidget(btn_cargar)

        self.lbl_info_archivo = QLabel("Sin imagen seleccionada")
        self.lbl_info_archivo.setObjectName("info")
        self.lbl_info_archivo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_info_archivo.setMaximumWidth(340)
        layout.addWidget(self.lbl_info_archivo)

        layout.addWidget(self._hline())

        # ── 2. Modelo de color ──────────────────
        layout.addWidget(self._section_label("02  ·  MODELO DE COLOR"))
        self.combo_modelo = CenteredComboBox()
        self.combo_modelo.addItems(["RGB", "HSV", "CMY", "YIQ", "HSI", "Escala de Grises"])
        self.combo_modelo.setItemDelegate(DarkTextDelegate(self.combo_modelo))
        self.combo_modelo.setStyleSheet(self._combo_style())
        layout.addWidget(self.combo_modelo)

        btn_modelo = QPushButton("Aplicar Modelo →")
        btn_modelo.setCursor(Qt.PointingHandCursor)
        btn_modelo.clicked.connect(self.aplicar_modelo)
        btn_modelo.setStyleSheet(self._btn_style_accion())
        layout.addWidget(btn_modelo)

        layout.addWidget(self._hline())

        # ── 3. Separar Canales ──────────────────
        layout.addWidget(self._section_label("03  ·  CANALES"))
        btn_capas = QPushButton("◫  Separar y Visualizar Capas")
        btn_capas.setCursor(Qt.PointingHandCursor)
        btn_capas.clicked.connect(self.mostrar_capas)
        btn_capas.setStyleSheet(self._btn_style_accion())
        layout.addWidget(btn_capas)

        layout.addWidget(self._hline())

        # ── 4. Binarización ─────────────────────
        layout.addWidget(self._section_label("04  ·  BINARIZACIÓN"))

        fila_s = QHBoxLayout()
        lbl_0 = QLabel("0"); lbl_0.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lbl_255 = QLabel("255"); lbl_255.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(128)
        self.slider.valueChanged.connect(lambda v: self.lbl_umbral.setText(f"Umbral: {v}"))
        self.slider.valueChanged.connect(lambda v: self.binarizar_manual())
        fila_s.addWidget(lbl_0); fila_s.addWidget(self.slider); fila_s.addWidget(lbl_255)
        layout.addLayout(fila_s)

        btn_menos = QPushButton(" − ")
        btn_menos.setFixedWidth(36)
        btn_menos.setToolTip("Decrementar umbral en 1")
        btn_menos.setCursor(Qt.PointingHandCursor)
        btn_menos.setStyleSheet(f"""
            QPushButton {{ background: {C['accent_dim']}; color: {C['accent']}; border: 1px solid {C['accent']}; border-radius: 4px; font-size: 14pt; font-weight: bold; padding: 0px; }}
            QPushButton:hover {{ background: {C['accent']}; color: #ffffff; }}
        """)
        btn_menos.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value() - 1)))

        btn_mas = QPushButton(" + ")
        btn_mas.setFixedWidth(36)
        btn_mas.setToolTip("Incrementar umbral en 1")
        btn_mas.setCursor(Qt.PointingHandCursor)
        btn_mas.setStyleSheet(f"""
            QPushButton {{ background: {C['accent_dim']}; color: {C['accent']}; border: 1px solid {C['accent']}; border-radius: 4px; font-size: 14pt; font-weight: bold; padding: 0px; }}
            QPushButton:hover {{ background: {C['accent']}; color: #ffffff; }}
        """)
        btn_mas.clicked.connect(lambda: self.slider.setValue(min(255, self.slider.value() + 1)))

        self.lbl_umbral = QLabel("Umbral: 128")
        self.lbl_umbral.setAlignment(Qt.AlignCenter)
        self.lbl_umbral.setStyleSheet(f"color:{C['accent']}; font-size:13pt; font-weight:bold;")

        fila_umbral = QHBoxLayout()
        fila_umbral.addWidget(btn_menos)
        fila_umbral.addWidget(self.lbl_umbral, 1)
        fila_umbral.addWidget(btn_mas)
        layout.addLayout(fila_umbral)

        btn_bin_otsu = QPushButton("⚙  Binarización Automática (Otsu)")
        btn_bin_otsu.setCursor(Qt.PointingHandCursor)
        btn_bin_otsu.clicked.connect(self.binarizar_otsu)
        btn_bin_otsu.setStyleSheet(self._btn_style_accion())
        layout.addWidget(btn_bin_otsu)

        self.lbl_otsu_resultado = QLabel("")
        self.lbl_otsu_resultado.setObjectName("warn")
        self.lbl_otsu_resultado.setVisible(False)
        layout.addWidget(self.lbl_otsu_resultado)

        layout.addStretch()
        return scroll

    def _make_tab_segmentacion(self):
        """Pestaña de Segmentación: ruido, operaciones lógicas/relacionales y conteo de objetos."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border:none; background:transparent;")
        inner_w = QWidget()
        inner_w.setStyleSheet(f"background: {C['surface']};")
        layout = QVBoxLayout(inner_w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(inner_w)

        # Fix: al cambiar de tab el QScrollArea no siempre recalcula el ancho del
        # widget interno. Conectar resizeEvent garantiza que inner_w nunca exceda
        # el ancho del viewport, eliminando el desbordamiento horizontal.
        def _constrain_width(event, _scroll=scroll, _inner=inner_w):
            _inner.setMaximumWidth(_scroll.viewport().width())
            QScrollArea.resizeEvent(_scroll, event)
        scroll.resizeEvent = _constrain_width

        btn_style_menos_mas = f"""
            QPushButton {{ background: {C['accent_dim']}; color: {C['accent']}; border: 1px solid {C['accent']}; border-radius: 4px; font-size: 14pt; font-weight: bold; padding: 0px; }}
            QPushButton:hover {{ background: {C['accent']}; color: #ffffff; }}
        """
        btn_style_accion = self._btn_style_accion()

        # ══════════════════════════════════════════
        # 01 · RUIDO
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("01  ·  RUIDO"))

        # ── Sal ──────────────────────────────────
        layout.addWidget(self._make_sublabel("Sal  (píxeles blancos)"))
        fila_sal = QHBoxLayout()
        lbl_sal_0 = QLabel("1%");   lbl_sal_0.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lbl_sal_20 = QLabel("20%"); lbl_sal_20.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        self.slider_sal = QSlider(Qt.Horizontal)
        self.slider_sal.setRange(1, 20)
        self.slider_sal.setValue(2)
        self.slider_sal.valueChanged.connect(
            lambda v: self.lbl_sal_valor.setText(f"Cantidad: {v}%"))
        fila_sal.addWidget(lbl_sal_0); fila_sal.addWidget(self.slider_sal); fila_sal.addWidget(lbl_sal_20)
        layout.addLayout(fila_sal)

        btn_sal_menos = QPushButton(" − "); btn_sal_menos.setFixedWidth(36)
        btn_sal_menos.setCursor(Qt.PointingHandCursor); btn_sal_menos.setStyleSheet(btn_style_menos_mas)
        btn_sal_menos.clicked.connect(lambda: self.slider_sal.setValue(max(1, self.slider_sal.value() - 1)))
        btn_sal_mas = QPushButton(" + "); btn_sal_mas.setFixedWidth(36)
        btn_sal_mas.setCursor(Qt.PointingHandCursor); btn_sal_mas.setStyleSheet(btn_style_menos_mas)
        btn_sal_mas.clicked.connect(lambda: self.slider_sal.setValue(min(20, self.slider_sal.value() + 1)))
        self.lbl_sal_valor = QLabel("Cantidad: 2%")
        self.lbl_sal_valor.setAlignment(Qt.AlignCenter)
        self.lbl_sal_valor.setStyleSheet(f"color:{C['accent']}; font-size:12pt; font-weight:bold;")
        fila_sal_ctrl = QHBoxLayout()
        fila_sal_ctrl.addWidget(btn_sal_menos); fila_sal_ctrl.addWidget(self.lbl_sal_valor, 1); fila_sal_ctrl.addWidget(btn_sal_mas)
        layout.addLayout(fila_sal_ctrl)

        btn_aplicar_sal = QPushButton("⬜  Aplicar Ruido Sal")
        btn_aplicar_sal.setCursor(Qt.PointingHandCursor)
        btn_aplicar_sal.setStyleSheet(btn_style_accion)
        btn_aplicar_sal.clicked.connect(self._seg_aplicar_ruido_sal)
        layout.addWidget(btn_aplicar_sal)

        layout.addWidget(self._hline())

        # ── Pimienta ──────────────────────────────
        layout.addWidget(self._make_sublabel("Pimienta  (píxeles negros)"))
        fila_pim = QHBoxLayout()
        lbl_pim_0 = QLabel("1%");   lbl_pim_0.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lbl_pim_20 = QLabel("20%"); lbl_pim_20.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        self.slider_pimienta = QSlider(Qt.Horizontal)
        self.slider_pimienta.setRange(1, 20)
        self.slider_pimienta.setValue(2)
        self.slider_pimienta.valueChanged.connect(
            lambda v: self.lbl_pimienta_valor.setText(f"Cantidad: {v}%"))
        fila_pim.addWidget(lbl_pim_0); fila_pim.addWidget(self.slider_pimienta); fila_pim.addWidget(lbl_pim_20)
        layout.addLayout(fila_pim)

        btn_pim_menos = QPushButton(" − "); btn_pim_menos.setFixedWidth(36)
        btn_pim_menos.setCursor(Qt.PointingHandCursor); btn_pim_menos.setStyleSheet(btn_style_menos_mas)
        btn_pim_menos.clicked.connect(lambda: self.slider_pimienta.setValue(max(1, self.slider_pimienta.value() - 1)))
        btn_pim_mas = QPushButton(" + "); btn_pim_mas.setFixedWidth(36)
        btn_pim_mas.setCursor(Qt.PointingHandCursor); btn_pim_mas.setStyleSheet(btn_style_menos_mas)
        btn_pim_mas.clicked.connect(lambda: self.slider_pimienta.setValue(min(20, self.slider_pimienta.value() + 1)))
        self.lbl_pimienta_valor = QLabel("Cantidad: 2%")
        self.lbl_pimienta_valor.setAlignment(Qt.AlignCenter)
        self.lbl_pimienta_valor.setStyleSheet(f"color:{C['accent']}; font-size:12pt; font-weight:bold;")
        fila_pim_ctrl = QHBoxLayout()
        fila_pim_ctrl.addWidget(btn_pim_menos); fila_pim_ctrl.addWidget(self.lbl_pimienta_valor, 1); fila_pim_ctrl.addWidget(btn_pim_mas)
        layout.addLayout(fila_pim_ctrl)

        btn_aplicar_pimienta = QPushButton("⬛  Aplicar Ruido Pimienta")
        btn_aplicar_pimienta.setCursor(Qt.PointingHandCursor)
        btn_aplicar_pimienta.setStyleSheet(btn_style_accion)
        btn_aplicar_pimienta.clicked.connect(self._seg_aplicar_ruido_pimienta)
        layout.addWidget(btn_aplicar_pimienta)

        layout.addWidget(self._hline())

        # ── Gaussiano ─────────────────────────────
        layout.addWidget(self._make_sublabel("Gaussiano  (variaciones de intensidad)"))
        fila_gau = QHBoxLayout()
        lbl_gau_0 = QLabel("1");   lbl_gau_0.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lbl_gau_100 = QLabel("100"); lbl_gau_100.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        self.slider_sigma = QSlider(Qt.Horizontal)
        self.slider_sigma.setRange(1, 100)
        self.slider_sigma.setValue(20)
        self.slider_sigma.valueChanged.connect(
            lambda v: self.lbl_sigma_valor.setText(f"Sigma: {v}"))
        fila_gau.addWidget(lbl_gau_0); fila_gau.addWidget(self.slider_sigma); fila_gau.addWidget(lbl_gau_100)
        layout.addLayout(fila_gau)

        btn_gau_menos = QPushButton(" − "); btn_gau_menos.setFixedWidth(36)
        btn_gau_menos.setCursor(Qt.PointingHandCursor); btn_gau_menos.setStyleSheet(btn_style_menos_mas)
        btn_gau_menos.clicked.connect(lambda: self.slider_sigma.setValue(max(1, self.slider_sigma.value() - 1)))
        btn_gau_mas = QPushButton(" + "); btn_gau_mas.setFixedWidth(36)
        btn_gau_mas.setCursor(Qt.PointingHandCursor); btn_gau_mas.setStyleSheet(btn_style_menos_mas)
        btn_gau_mas.clicked.connect(lambda: self.slider_sigma.setValue(min(100, self.slider_sigma.value() + 1)))
        self.lbl_sigma_valor = QLabel("Sigma: 20")
        self.lbl_sigma_valor.setAlignment(Qt.AlignCenter)
        self.lbl_sigma_valor.setStyleSheet(f"color:{C['accent']}; font-size:12pt; font-weight:bold;")
        fila_gau_ctrl = QHBoxLayout()
        fila_gau_ctrl.addWidget(btn_gau_menos); fila_gau_ctrl.addWidget(self.lbl_sigma_valor, 1); fila_gau_ctrl.addWidget(btn_gau_mas)
        layout.addLayout(fila_gau_ctrl)

        btn_aplicar_gaussiano = QPushButton("〜  Aplicar Ruido Gaussiano")
        btn_aplicar_gaussiano.setCursor(Qt.PointingHandCursor)
        btn_aplicar_gaussiano.setStyleSheet(btn_style_accion)
        btn_aplicar_gaussiano.clicked.connect(self._seg_aplicar_ruido_gaussiano)
        layout.addWidget(btn_aplicar_gaussiano)

        btn_retirar_ruido = QPushButton("✕  Retirar Ruido")
        btn_retirar_ruido.setCursor(Qt.PointingHandCursor)
        btn_retirar_ruido.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {C['danger']};
                color: {C['danger']};
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background: {C['danger']}; color: white; }}
        """)
        btn_retirar_ruido.clicked.connect(self._seg_retirar_ruido)
        layout.addWidget(btn_retirar_ruido)

        layout.addWidget(self._hline())

        # ══════════════════════════════════════════
        # 02 · OPERACIONES ARITMÉTICAS
        # ══════════════════════════════════════════

        # ── Imagen B (compartida por aritméticas y lógicas) ──────
        layout.addWidget(self._make_sublabel("Imagen secundaria (B)"))

        self.thumb_imagen_b = QLabel()
        self.thumb_imagen_b.setFixedSize(70, 56)
        self.thumb_imagen_b.setAlignment(Qt.AlignCenter)
        self.thumb_imagen_b.setStyleSheet(f"""
            border: 1px solid {C['border2']};
            border-radius: 3px;
            background: {C['surface2']};
            padding: 2px;
        """)
        self.thumb_imagen_b.setVisible(False)
        layout.addWidget(self.thumb_imagen_b)

        self.lbl_imagen_b = QLabel("Sin imagen B cargada")
        self.lbl_imagen_b.setObjectName("info")
        self.lbl_imagen_b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_imagen_b.setMaximumWidth(340)
        layout.addWidget(self.lbl_imagen_b)
        btn_cargar_b = QPushButton("⊕  Cargar imagen B...")
        btn_cargar_b.setCursor(Qt.PointingHandCursor)
        btn_cargar_b.setStyleSheet(btn_style_accion)
        btn_cargar_b.clicked.connect(self._seg_cargar_imagen_b)
        layout.addWidget(btn_cargar_b)

        layout.addWidget(self._hline())

        layout.addWidget(self._section_label("02  ·  OPERACIONES ARITMÉTICAS"))
        layout.addWidget(self._make_sublabel("Operan sobre datos en memoria  ·  requieren imagen B"))

        fila_arit = QHBoxLayout()
        fila_arit.setSpacing(6)
        for texto, slot in [("Sumar", self._seg_sumar), ("Restar", self._seg_restar), ("Multiplicar", self._seg_multiplicar)]:
            b = QPushButton(texto)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style_accion)
            b.clicked.connect(slot)
            fila_arit.addWidget(b)
        layout.addLayout(fila_arit)

        layout.addWidget(self._hline())

        # ══════════════════════════════════════════
        # 03 · OPERACIONES LÓGICAS
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("03  ·  OPERACIONES LÓGICAS"))

        # ── Lógicas ───────────────────────────────
        layout.addWidget(self._make_sublabel("Lógicas  (operan sobre imagen binaria)"))

        fila_log1 = QHBoxLayout()
        fila_log1.setSpacing(6)
        for texto, slot in [("AND", self._seg_and), ("OR", self._seg_or), ("XOR", self._seg_xor)]:
            b = QPushButton(texto)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style_accion)
            b.clicked.connect(slot)
            fila_log1.addWidget(b)
        layout.addLayout(fila_log1)

        btn_not = QPushButton("NOT  (invertir imagen A)")
        btn_not.setCursor(Qt.PointingHandCursor)
        btn_not.setStyleSheet(btn_style_accion)
        btn_not.clicked.connect(self._seg_not)
        layout.addWidget(btn_not)

        layout.addWidget(self._hline())

        # ── Relacionales ──────────────────────────
        layout.addWidget(self._make_sublabel("Relacionales  (comparar intensidad contra umbral)"))

        fila_rel_s = QHBoxLayout()
        lbl_rel_0   = QLabel("0");   lbl_rel_0.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        lbl_rel_255 = QLabel("255"); lbl_rel_255.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        self.slider_umbral_rel = QSlider(Qt.Horizontal)
        self.slider_umbral_rel.setRange(0, 255)
        self.slider_umbral_rel.setValue(128)
        self.slider_umbral_rel.valueChanged.connect(
            lambda v: self.lbl_umbral_rel_valor.setText(f"Umbral: {v}"))
        fila_rel_s.addWidget(lbl_rel_0)
        fila_rel_s.addWidget(self.slider_umbral_rel)
        fila_rel_s.addWidget(lbl_rel_255)
        layout.addLayout(fila_rel_s)

        btn_rel_menos = QPushButton(" − "); btn_rel_menos.setFixedWidth(36)
        btn_rel_menos.setCursor(Qt.PointingHandCursor); btn_rel_menos.setStyleSheet(btn_style_menos_mas)
        btn_rel_menos.clicked.connect(
            lambda: self.slider_umbral_rel.setValue(max(0, self.slider_umbral_rel.value() - 1)))
        btn_rel_mas = QPushButton(" + "); btn_rel_mas.setFixedWidth(36)
        btn_rel_mas.setCursor(Qt.PointingHandCursor); btn_rel_mas.setStyleSheet(btn_style_menos_mas)
        btn_rel_mas.clicked.connect(
            lambda: self.slider_umbral_rel.setValue(min(255, self.slider_umbral_rel.value() + 1)))
        self.lbl_umbral_rel_valor = QLabel("Umbral: 128")
        self.lbl_umbral_rel_valor.setAlignment(Qt.AlignCenter)
        self.lbl_umbral_rel_valor.setStyleSheet(f"color:{C['accent']}; font-size:12pt; font-weight:bold;")
        fila_rel_ctrl = QHBoxLayout()
        fila_rel_ctrl.addWidget(btn_rel_menos)
        fila_rel_ctrl.addWidget(self.lbl_umbral_rel_valor, 1)
        fila_rel_ctrl.addWidget(btn_rel_mas)
        layout.addLayout(fila_rel_ctrl)

        fila_rel = QHBoxLayout()
        fila_rel.setSpacing(6)
        for texto, slot in [("mayor >", self._seg_relacional_mayor),
                             ("menor <", self._seg_relacional_menor),
                             ("igual ==", self._seg_relacional_igual)]:
            b = QPushButton(texto)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style_accion)
            b.clicked.connect(slot)
            fila_rel.addWidget(b)
        layout.addLayout(fila_rel)

        layout.addWidget(self._hline())

        # ══════════════════════════════════════════
        # 03 · CONTEO DE OBJETOS
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("04  ·  CONTEO DE OBJETOS"))

        fila_vec = QHBoxLayout()
        fila_vec.setSpacing(6)
        btn_v4 = QPushButton("Vecindad-4")
        btn_v4.setCursor(Qt.PointingHandCursor)
        btn_v4.setStyleSheet(btn_style_accion)
        btn_v4.clicked.connect(self._seg_vecindad_4)
        btn_v8 = QPushButton("Vecindad-8")
        btn_v8.setCursor(Qt.PointingHandCursor)
        btn_v8.setStyleSheet(btn_style_accion)
        btn_v8.clicked.connect(self._seg_vecindad_8)
        fila_vec.addWidget(btn_v4); fila_vec.addWidget(btn_v8)
        layout.addLayout(fila_vec)

        btn_comparar = QPushButton("◫  Comparar Vecindad-4 vs Vecindad-8")
        btn_comparar.setCursor(Qt.PointingHandCursor)
        btn_comparar.setStyleSheet(btn_style_accion)
        btn_comparar.clicked.connect(self._seg_comparar_vecindad)
        layout.addWidget(btn_comparar)

        self.lbl_conteo_v4 = QLabel("")
        self.lbl_conteo_v4.setObjectName("warn")
        self.lbl_conteo_v4.setAlignment(Qt.AlignCenter)
        self.lbl_conteo_v4.setVisible(False)
        layout.addWidget(self.lbl_conteo_v4)

        self.lbl_conteo_v8 = QLabel("")
        self.lbl_conteo_v8.setObjectName("warn")
        self.lbl_conteo_v8.setAlignment(Qt.AlignCenter)
        self.lbl_conteo_v8.setVisible(False)
        layout.addWidget(self.lbl_conteo_v8)

        layout.addStretch()
        return scroll


    def _make_tab_morfologia(self):
        """Pestaña de Morfología: operaciones básicas, binaria avanzada y en laticces."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border:none; background:transparent;")
        inner_w = QWidget()
        inner_w.setStyleSheet(f"background: {C['surface']};")
        layout = QVBoxLayout(inner_w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(inner_w)

        def _constrain_width(event, _scroll=scroll, _inner=inner_w):
            _inner.setMaximumWidth(_scroll.viewport().width())
            QScrollArea.resizeEvent(_scroll, event)
        scroll.resizeEvent = _constrain_width

        btn_style_mm  = self._btn_style_accion()
        btn_style_mmas = f"""
            QPushButton {{ background: {C['accent_dim']}; color: {C['accent']}; border: 1px solid {C['accent']}; border-radius: 4px; font-size: 14pt; font-weight: bold; padding: 0px; }}
            QPushButton:hover {{ background: {C['accent']}; color: #ffffff; }}
        """

        # ── Helper interno: slider con etiquetas y botones +/- ──────────────
        def _slider_row(parent_layout, attr_slider, attr_lbl, rango, defecto, texto_lbl, fmt):
            """
            Crea un slider con botones −/+ y label de valor, y los añade a parent_layout.
            Retorna (slider, lbl_valor).
            """
            fila_s = QHBoxLayout()
            lbl_lo = QLabel(str(rango[0])); lbl_lo.setStyleSheet(f"color:{C['text3']};font-size:10px;")
            lbl_hi = QLabel(str(rango[1])); lbl_hi.setStyleSheet(f"color:{C['text3']};font-size:10px;")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*rango)
            slider.setValue(defecto)
            lbl_val = QLabel(fmt.format(defecto))
            lbl_val.setAlignment(Qt.AlignCenter)
            lbl_val.setStyleSheet(f"color:{C['accent']}; font-size:12pt; font-weight:bold;")
            slider.valueChanged.connect(lambda v, _l=lbl_val, _f=fmt: _l.setText(_f.format(v)))
            fila_s.addWidget(lbl_lo); fila_s.addWidget(slider); fila_s.addWidget(lbl_hi)
            parent_layout.addLayout(fila_s)

            btn_m = QPushButton(" − "); btn_m.setFixedWidth(36)
            btn_m.setCursor(Qt.PointingHandCursor); btn_m.setStyleSheet(btn_style_mmas)
            btn_m.clicked.connect(lambda: slider.setValue(max(rango[0], slider.value() - 1)))
            btn_p = QPushButton(" + "); btn_p.setFixedWidth(36)
            btn_p.setCursor(Qt.PointingHandCursor); btn_p.setStyleSheet(btn_style_mmas)
            btn_p.clicked.connect(lambda: slider.setValue(min(rango[1], slider.value() + 1)))
            fila_ctrl = QHBoxLayout()
            fila_ctrl.addWidget(btn_m); fila_ctrl.addWidget(lbl_val, 1); fila_ctrl.addWidget(btn_p)
            parent_layout.addLayout(fila_ctrl)
            return slider, lbl_val

        # ── Helper: ComboBox forma EE ────────────────────────────────────────
        def _combo_forma(parent_layout):
            lbl = self._make_sublabel("Forma del EE:")
            parent_layout.addWidget(lbl)
            combo = CenteredComboBox()
            combo.addItems(["Disco  (bordes naturales)", "Cuadrado  (más agresivo)"])
            combo.setItemDelegate(DarkTextDelegate(combo))
            combo.setStyleSheet(self._combo_style())
            parent_layout.addWidget(combo)
            return combo

        def _forma_str(combo):
            return "disco" if combo.currentIndex() == 0 else "cuadrado"

        # ══════════════════════════════════════════
        # 01 · MORFOLOGÍA BÁSICA
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("01  ·  MORFOLOGÍA BÁSICA"))
        layout.addWidget(self._make_sublabel("Requiere: BINARIO o GRIS"))

        self.combo_ee_basica = _combo_forma(layout)

        layout.addWidget(self._make_sublabel("Tamaño del EE (kernel):"))
        self.slider_mm_kernel, self.lbl_mm_kernel = _slider_row(
            layout, "slider_mm_kernel", "lbl_mm_kernel", (3, 31), 11, None, "Kernel: {0}×{0}")

        layout.addWidget(self._make_sublabel("Iteraciones:"))
        self.slider_mm_iter, self.lbl_mm_iter = _slider_row(
            layout, "slider_mm_iter", "lbl_mm_iter", (1, 10), 1, None, "Iter: {0}")

        fila_bas = QGridLayout()
        fila_bas.setSpacing(6)
        for col, (texto, slot) in enumerate([
            ("⊖  Erosión",   self._mm_erosion),
            ("⊕  Dilatación", self._mm_dilatacion),
            ("◯  Apertura",  self._mm_apertura),
            ("●  Cierre",    self._mm_cierre),
        ]):
            b = QPushButton(texto)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style_mm)
            b.clicked.connect(slot)
            fila_bas.addWidget(b, col // 2, col % 2)
        layout.addLayout(fila_bas)

        layout.addWidget(self._hline())

        # ══════════════════════════════════════════
        # 02 · MORFOLOGÍA BINARIA AVANZADA
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("02  ·  MORFOLOGÍA BINARIA"))
        layout.addWidget(self._make_sublabel("Requiere: BINARIO"))

        self.combo_ee_binaria = _combo_forma(layout)

        layout.addWidget(self._make_sublabel("Tamaño del EE (frontera / esqueleto):"))
        self.slider_mm_bin_kernel, self.lbl_mm_bin_kernel = _slider_row(
            layout, None, None, (3, 15), 3, None, "Kernel: {0}×{0}")

        # Hit-or-Miss: selector de patrón
        layout.addWidget(self._make_sublabel("Patrón Hit-or-Miss:"))
        self.combo_hitmiss = CenteredComboBox()
        self.combo_hitmiss.addItems([
            "Esquina (4 rotaciones)",
            "Punto aislado",
            "Extremo de línea (4 rotaciones)",
        ])
        self.combo_hitmiss.setItemDelegate(DarkTextDelegate(self.combo_hitmiss))
        self.combo_hitmiss.setStyleSheet(self._combo_style())
        layout.addWidget(self.combo_hitmiss)

        fila_bin = QGridLayout()
        fila_bin.setSpacing(6)
        for col, (texto, slot) in enumerate([
            ("⬚  Frontera",         self._mm_frontera),
            ("⊛  Hit-or-Miss",      self._mm_hit_or_miss),
            ("⇒  Adelgazamiento",   self._mm_adelgazamiento),
            ("☆  Esqueleto",        self._mm_esqueleto),
        ]):
            b = QPushButton(texto)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style_mm)
            b.clicked.connect(slot)
            fila_bin.addWidget(b, col // 2, col % 2)
        layout.addLayout(fila_bin)

        layout.addWidget(self._hline())

        # ══════════════════════════════════════════
        # 03 · MORFOLOGÍA EN LATICCES
        # ══════════════════════════════════════════
        layout.addWidget(self._section_label("03  ·  MORFOLOGÍA EN LATICCES"))
        layout.addWidget(self._make_sublabel("Requiere: GRIS"))

        self.combo_ee_laticces = _combo_forma(layout)

        layout.addWidget(self._make_sublabel("Tamaño del EE:"))
        self.slider_mm_lat_kernel, self.lbl_mm_lat_kernel = _slider_row(
            layout, None, None, (3, 31), 5, None, "Kernel: {0}×{0}")

        # Gradiente: selector de tipo
        layout.addWidget(self._make_sublabel("Tipo de Gradiente:"))
        self.combo_gradiente = CenteredComboBox()
        self.combo_gradiente.addItems([
            "Simétrico  (dilatación − erosión)",
            "Por dilatación  (dilatación − imagen)",
            "Por erosión  (imagen − erosión)",
        ])
        self.combo_gradiente.setItemDelegate(DarkTextDelegate(self.combo_gradiente))
        self.combo_gradiente.setStyleSheet(self._combo_style())
        layout.addWidget(self.combo_gradiente)

        btn_grad = QPushButton("∇  Gradiente Morfológico")
        btn_grad.setCursor(Qt.PointingHandCursor)
        btn_grad.setStyleSheet(btn_style_mm)
        btn_grad.clicked.connect(self._mm_gradiente)
        layout.addWidget(btn_grad)

        layout.addWidget(self._make_sublabel("Top Hat / Bot Hat:"))
        fila_hat = QHBoxLayout()
        fila_hat.setSpacing(6)
        btn_top = QPushButton("▲  Top Hat")
        btn_top.setCursor(Qt.PointingHandCursor)
        btn_top.setStyleSheet(btn_style_mm)
        btn_top.clicked.connect(self._mm_top_hat)
        btn_bot = QPushButton("▼  Bot Hat")
        btn_bot.setCursor(Qt.PointingHandCursor)
        btn_bot.setStyleSheet(btn_style_mm)
        btn_bot.clicked.connect(self._mm_bot_hat)
        fila_hat.addWidget(btn_top); fila_hat.addWidget(btn_bot)
        layout.addLayout(fila_hat)

        layout.addWidget(self._make_sublabel("Orden del filtro de suavizado:"))
        self.combo_suavizado = CenteredComboBox()
        self.combo_suavizado.addItems([
            "Apertura → Cierre  (elimina sal, luego pimienta)",
            "Cierre → Apertura  (elimina pimienta, luego sal)",
        ])
        self.combo_suavizado.setItemDelegate(DarkTextDelegate(self.combo_suavizado))
        self.combo_suavizado.setStyleSheet(self._combo_style())
        layout.addWidget(self.combo_suavizado)

        btn_suav = QPushButton("≈  Suavizado Morfológico")
        btn_suav.setCursor(Qt.PointingHandCursor)
        btn_suav.setStyleSheet(btn_style_mm)
        btn_suav.clicked.connect(self._mm_suavizado)
        layout.addWidget(btn_suav)

        layout.addStretch()
        return scroll

    # ══════════════════════════════════════════════════════════════
    #  ACCIONES — MORFOLOGÍA BÁSICA
    # ══════════════════════════════════════════════════════════════

    def _mm_erosion(self):
        if not self._check(): return
        k = self.slider_mm_kernel.value()
        i = self.slider_mm_iter.value()
        f = "disco" if self.combo_ee_basica.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.erosion(self.imagen_metadata, kernel_size=k, iteraciones=i, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_dilatacion(self):
        if not self._check(): return
        k = self.slider_mm_kernel.value()
        i = self.slider_mm_iter.value()
        f = "disco" if self.combo_ee_basica.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.dilatacion(self.imagen_metadata, kernel_size=k, iteraciones=i, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_apertura(self):
        if not self._check(): return
        k = self.slider_mm_kernel.value()
        i = self.slider_mm_iter.value()
        f = "disco" if self.combo_ee_basica.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.apertura(self.imagen_metadata, kernel_size=k, iteraciones=i, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_cierre(self):
        if not self._check(): return
        k = self.slider_mm_kernel.value()
        i = self.slider_mm_iter.value()
        f = "disco" if self.combo_ee_basica.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.cierre(self.imagen_metadata, kernel_size=k, iteraciones=i, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    # ══════════════════════════════════════════════════════════════
    #  ACCIONES — MORFOLOGÍA BINARIA AVANZADA
    # ══════════════════════════════════════════════════════════════

    def _mm_frontera(self):
        if not self._check(): return
        k = self.slider_mm_bin_kernel.value()
        f = "disco" if self.combo_ee_binaria.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.frontera(self.imagen_metadata, kernel_size=k, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_hit_or_miss(self):
        if not self._check(): return
        tipos = ["esquina", "punto_aislado", "extremo_linea"]
        tipo = tipos[self.combo_hitmiss.currentIndex()]
        resp = procesadorImagen.hit_or_miss(self.imagen_metadata, tipo_ee=tipo)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_adelgazamiento(self):
        if not self._check(): return
        resp = procesadorImagen.adelgazamiento(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_esqueleto(self):
        if not self._check(): return
        k = self.slider_mm_bin_kernel.value()
        f = "disco" if self.combo_ee_binaria.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.esqueleto(self.imagen_metadata, kernel_size=k, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    # ══════════════════════════════════════════════════════════════
    #  ACCIONES — MORFOLOGÍA EN LATICCES
    # ══════════════════════════════════════════════════════════════

    def _mm_gradiente(self):
        if not self._check(): return
        k = self.slider_mm_lat_kernel.value()
        f = "disco" if self.combo_ee_laticces.currentIndex() == 0 else "cuadrado"
        tipos = ["simetrico", "dilatacion", "erosion"]
        tipo = tipos[self.combo_gradiente.currentIndex()]
        resp = procesadorImagen.gradiente_morfologico(self.imagen_metadata, tipo=tipo, kernel_size=k, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_top_hat(self):
        if not self._check(): return
        k = self.slider_mm_lat_kernel.value()
        f = "disco" if self.combo_ee_laticces.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.top_hat(self.imagen_metadata, kernel_size=k, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_bot_hat(self):
        if not self._check(): return
        k = self.slider_mm_lat_kernel.value()
        f = "disco" if self.combo_ee_laticces.currentIndex() == 0 else "cuadrado"
        resp = procesadorImagen.bot_hat(self.imagen_metadata, kernel_size=k, forma=f)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _mm_suavizado(self):
        if not self._check(): return
        k = self.slider_mm_lat_kernel.value()
        f = "disco" if self.combo_ee_laticces.currentIndex() == 0 else "cuadrado"
        orden = "apertura_cierre" if self.combo_suavizado.currentIndex() == 0 else "cierre_apertura"
        resp = procesadorImagen.suavizado_morfologico(self.imagen_metadata, kernel_size=k, forma=f, orden=orden)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _make_sublabel(self, text):
        """Label secundario para subtítulos dentro de una sección."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{C['text2']}; font-size:11pt;")
        return lbl

    def _combo_style(self):
        """Estilo explícito para ComboBox y su dropdown — coherente con el tema del dashboard.
        Se aplica directamente a cada combo para garantizar que Windows no use el estilo nativo."""
        return f"""
            QComboBox {{
                background: {C['surface3']};
                color: {C['text']};
                border: 1px solid {C['border2']};
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 12pt;
                min-height: 28px;
            }}
            QComboBox:focus {{ border-color: {C['accent']}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {C['surface2']};
                color: {C['text']};
                border: 1px solid {C['accent']};
                selection-background-color: {C['accent_dim']};
                selection-color: {C['accent']};
                outline: none;
                padding: 4px 0px;
            }}
        """

    def _btn_style_accion(self):
        """Estilo homologado para botones de acción en ambas pestañas del sidebar."""
        return f"""
            QPushButton {{
                background: {C['surface3']};
                color: {C['text']};
                border: 1px solid {C['border2']};
                border-radius: 4px;
                padding: 10px 12px;
                min-height: 38px;
                text-align: center;
                font-size: 12pt;
            }}
            QPushButton:hover {{
                background: {C['accent_dim']};
                border-color: {C['accent']};
                color: {C['accent']};
            }}
            QPushButton:pressed {{ background: {C['accent']}; color: {C['bg']}; }}
        """



    def _make_footer_guardar(self):
        """Footer fijo debajo de las tabs: siempre visible, independiente de la pestaña activa."""
        footer = QWidget()
        footer.setStyleSheet(f"""
            background: {C['surface2']};
            border-top: 1px solid {C['border2']};
        """)
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        # ── Restablecer ─────────────────────────
        btn_restablecer = QPushButton("↺  Restablecer imagen")
        btn_restablecer.setCursor(Qt.PointingHandCursor)
        btn_restablecer.clicked.connect(self.restablecer_imagen)
        btn_restablecer.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {C['danger']};
                color: {C['danger']};
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background: {C['danger']}; color: white; }}
        """)
        layout.addWidget(btn_restablecer)

        # ── Separador ────────────────────────────
        sep = QFrame()
        sep.setObjectName("hline")
        layout.addWidget(sep)

        # ── Checkboxes + Guardar ─────────────────
        chk_style = f"""
            QCheckBox {{ color: {C['text2']}; font-size: 11pt; spacing: 6px; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {C['border2']};
                border-radius: 3px; background: {C['surface3']}; }}
            QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
        """
        fila_chk = QHBoxLayout()
        fila_chk.setSpacing(16)
        self.chk_histograma = QCheckBox("Histograma")
        self.chk_histograma.setStyleSheet(chk_style)
        self.chk_canales = QCheckBox("Canales")
        self.chk_canales.setStyleSheet(chk_style)
        self.chk_conteo = QCheckBox("Conteo obj.")
        self.chk_conteo.setStyleSheet(chk_style)
        fila_chk.addWidget(self.chk_histograma)
        fila_chk.addWidget(self.chk_canales)
        fila_chk.addWidget(self.chk_conteo)
        fila_chk.addStretch()
        layout.addLayout(fila_chk)

        btn_guardar = QPushButton("💾  Guardar Imagen...")
        btn_guardar.setCursor(Qt.PointingHandCursor)
        btn_guardar.clicked.connect(self.guardar_imagen_actual)
        btn_guardar.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {C['accent2']};
                color: {C['accent2']};
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background: {C['accent2']}; color: white; }}
        """)
        layout.addWidget(btn_guardar)

        return footer


    # ── Panel Central (visor) ─────────────────────
    def _make_center(self):
        w = QWidget()
        w.setStyleSheet(f"background:{C['bg']};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Cabecera del visor
        header = QWidget()
        header.setFixedHeight(46)
        header.setStyleSheet(f"background:{C['surface']}; border-bottom:1px solid {C['border']};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)
        lbl_visor = QLabel("VISOR  ·  Imagen Actual")
        lbl_visor.setStyleSheet(f"color:{C['text']}; font-size:11pt; letter-spacing:2px;")
        self.lbl_dims = QLabel("")
        self.lbl_dims.setStyleSheet(f"color:{C['text']}; font-size:11pt;")
        h_layout.addWidget(lbl_visor)
        h_layout.addStretch()
        h_layout.addWidget(self.lbl_dims)
        layout.addWidget(header)

        # Scroll área para la imagen
        self.scroll_visor = QScrollArea()
        self.scroll_visor.setWidgetResizable(True)
        self.scroll_visor.setAlignment(Qt.AlignCenter)
        self.scroll_visor.setStyleSheet(f"border:none; background:{C['bg']};")
        self.lbl_imagen = QLabel("[ Carga una imagen para comenzar ]")
        self.lbl_imagen.setAlignment(Qt.AlignCenter)
        self.lbl_imagen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_imagen.setStyleSheet(f"color:{C['text2']}; font-size:14pt; background:{C['bg']};")
        self.scroll_visor.setWidget(self.lbl_imagen)
        layout.addWidget(self.scroll_visor, 1)

        # Barra de historial (carrusel)
        carrusel_header = QWidget()
        carrusel_header.setFixedHeight(28)
        carrusel_header.setStyleSheet(f"background:{C['surface']}; border-top:1px solid {C['border']};")
        ch_layout = QHBoxLayout(carrusel_header)
        ch_layout.setContentsMargins(12, 0, 12, 0)
        lbl_h = QLabel("HISTORIAL")
        lbl_h.setStyleSheet(f"color:{C['text2']}; font-size:10pt; letter-spacing:2px;")
        ch_layout.addWidget(lbl_h)
        layout.addWidget(carrusel_header)

        self.scroll_carrusel = QScrollArea()
        self.scroll_carrusel.setFixedHeight(88)
        self.scroll_carrusel.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_carrusel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_carrusel.setStyleSheet(f"border-top:1px solid {C['border']}; border-bottom:none; background:{C['surface']};")
        self.scroll_carrusel.setWidgetResizable(True)
        self.carrusel_inner = QWidget()
        self.carrusel_layout = QHBoxLayout(self.carrusel_inner)
        self.carrusel_layout.setContentsMargins(8, 6, 8, 6)
        self.carrusel_layout.setSpacing(6)
        self.carrusel_layout.setAlignment(Qt.AlignLeft)
        self.scroll_carrusel.setWidget(self.carrusel_inner)
        layout.addWidget(self.scroll_carrusel)

        return w

    # ── Panel Derecho (estadísticas) ──────────────
    def _make_right_panel(self):
        w = QWidget()
        w.setStyleSheet(f"background:{C['surface']}; border-left:1px solid {C['border']};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(46)
        header.setStyleSheet(f"background:{C['surface2']}; border-bottom:1px solid {C['border']};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)
        lbl = QLabel("ANÁLISIS  ·  Estadísticas por Canal")
        lbl.setStyleSheet(f"color:{C['text2']}; font-size:11pt; letter-spacing:2px;")
        h_layout.addWidget(lbl)
        layout.addWidget(header)

        # Grid de métricas resumen (4 tarjetas)
        grid_w = QWidget()
        grid_w.setStyleSheet(f"background:{C['surface2']}; border-bottom:1px solid {C['border']};")
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(6)

        self.card_media   = MetricCard("MEDIA",    color=C["accent"])
        self.card_var     = MetricCard("VARIANZA",  color=C["accent2"])
        self.card_energia = MetricCard("ENERGÍA",   color=C["warn"])
        self.card_entrop  = MetricCard("ENTROPÍA",  color="#AA55FF")

        grid.addWidget(self.card_media,   0, 0)
        grid.addWidget(self.card_var,     0, 1)
        grid.addWidget(self.card_energia, 1, 0)
        grid.addWidget(self.card_entrop,  1, 1)
        layout.addWidget(grid_w)

        # Detalle completo scrollable
        scroll_stats = QScrollArea()
        scroll_stats.setWidgetResizable(True)
        scroll_stats.setStyleSheet("border:none; background:transparent;")
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)
        self.stats_layout.setContentsMargins(14, 12, 14, 12)
        self.stats_layout.setSpacing(4)
        self.stats_layout.setAlignment(Qt.AlignTop)

        placeholder = QLabel("Calcula el histograma para\nver las estadísticas aquí.")
        placeholder.setStyleSheet(f"color:{C['text2']}; font-size:12pt; padding:20px;")
        placeholder.setAlignment(Qt.AlignCenter)
        self.stats_layout.addWidget(placeholder)
        scroll_stats.setWidget(self.stats_widget)
        layout.addWidget(scroll_stats, 1)

        # ── Separador ──────────────────────────
        sep = QFrame(); sep.setObjectName("hline")
        layout.addWidget(sep)

        # ── Canvas embebido del histograma ─────
        self.fig_canvas = Figure(figsize=(4, 2.5), facecolor="#0D1117")
        self.ax_hist = self.fig_canvas.add_subplot(111)
        self.ax_hist.set_facecolor("#0D1117")
        self.ax_hist.tick_params(colors="#6B8FA8", labelsize=7)
        for spine in self.ax_hist.spines.values():
            spine.set_edgecolor("#1E2D3D")
        self.ax_hist.set_xlabel("Intensidad (0–255)", color="#6B8FA8", fontsize=7)
        self.ax_hist.set_ylabel("Frecuencia", color="#6B8FA8", fontsize=7)
        self.ax_hist.grid(True, linestyle="--", alpha=0.2, color="#1E2D3D")
        self.canvas_widget = FigureCanvas(self.fig_canvas)
        self.canvas_widget.setMinimumHeight(200)
        self.canvas_widget.setStyleSheet(f"background:{C['surface']};")
        layout.addWidget(self.canvas_widget)

        # ── Botón Ver Gráfica (ventana externa) ─
        btn_ver = QPushButton("📈  Ver Gráfica")
        btn_ver.setCursor(Qt.PointingHandCursor)
        btn_ver.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {C['accent']};
                color: {C['accent']};
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background: {C['accent']}; color: white; }}
        """)
        btn_ver.clicked.connect(self.mostrar_grafica_histograma)
        layout.addWidget(btn_ver)

        return w

    # ── Status Bar ───────────────────────────────
    def _make_statusbar(self):
        w = QLabel("  ◉  Listo  ·  Sin imagen cargada")
        w.setFixedHeight(36)
        w.setStyleSheet(f"""
            background:{C['surface']};
            border-top:1px solid {C['border']};
            color:{C['text2']};
            font-size:11pt;
            padding: 0 12px;
        """)
        self.statusbar_lbl = w
        return w

    # ── Helpers de layout ────────────────────────
    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section_title")
        return lbl

    def _hline(self):
        f = QFrame()
        f.setObjectName("hline")
        return f

    # ══════════════════════════════════════════════════════════════
    #  ACCIONES
    # ══════════════════════════════════════════════════════════════

    # ── Persistencia de directorios (un solo archivo, dos claves) ───
    _CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultimo_directorio.txt")

    def _inicializar_cache(self):
        """Crea el archivo de caché con claves vacías si no existe."""
        try:
            if not os.path.isfile(self._CACHE_PATH):
                with open(self._CACHE_PATH, "w", encoding="utf-8") as f:
                    f.write("open=\nsave=\n")
        except Exception:
            pass

    def _leer_cache(self, clave):
        """Lee el valor de una clave (open/save) del archivo de caché."""
        try:
            if os.path.isfile(self._CACHE_PATH):
                with open(self._CACHE_PATH, "r", encoding="utf-8") as f:
                    for linea in f:
                        if linea.startswith(f"{clave}="):
                            valor = linea.split("=", 1)[1].strip()
                            if valor and os.path.isdir(valor):
                                return valor
        except Exception:
            pass
        return None

    def _escribir_cache(self, clave, directorio):
        """Actualiza el valor de una clave preservando las demás entradas."""
        try:
            lineas = {}
            if os.path.isfile(self._CACHE_PATH):
                with open(self._CACHE_PATH, "r", encoding="utf-8") as f:
                    for linea in f:
                        if "=" in linea:
                            k, v = linea.split("=", 1)
                            lineas[k.strip()] = v.strip()
            lineas[clave] = directorio
            with open(self._CACHE_PATH, "w", encoding="utf-8") as f:
                for k, v in lineas.items():
                    f.write(f"{k}={v}\n")
        except Exception:
            pass

    # ── Clave "open": último directorio al abrir imagen ─────────────
    def _leer_ultimo_directorio(self):
        default = os.path.join(config.script_dir_parent, 'resources', 'input')
        if not os.path.isdir(default):
            default = config.script_dir_parent
        return self._leer_cache("open") or default

    def _guardar_ultimo_directorio(self, ruta_archivo):
        self._escribir_cache("open", os.path.dirname(ruta_archivo))

    # ── Clave "save": último directorio al guardar ───────────────────
    def _leer_ultimo_directorio_guardado(self):
        default = os.path.join(config.script_dir_parent, 'resources', 'output')
        try:
            os.makedirs(default, exist_ok=True)
        except Exception:
            default = config.script_dir_parent
        return self._leer_cache("save") or default

    def _guardar_ultimo_directorio_guardado(self, carpeta):
        self._escribir_cache("save", carpeta)

    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", self._leer_ultimo_directorio(),
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.webp *.pgm *.ppm)"
        )
        if not ruta: return
        self._guardar_ultimo_directorio(ruta)
        self.imagen_metadata = metadataImagen(ruta)
        resp = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
            return
        # Guardar copia de los datos RGB originales para el slider de binarización
        self._datos_originales = self.imagen_metadata.datos.copy()
        self._base_bin_datos   = self.imagen_metadata.datos.copy()
        self._base_bin_modelo  = self.imagen_metadata.modelo
        nombre = self.imagen_metadata.nombre
        self.lbl_info_archivo.setText(nombre[:42] + "…" if len(nombre) > 42 else nombre)
        self.chip_archivo.setText(nombre[:20] + "…" if len(nombre) > 20 else nombre)
        self._mostrar_imagen()
        self._status(f"✔  {nombre}  cargado correctamente  ·  RGB")
        self.calcular_histograma()

    def aplicar_modelo(self):
        if not self._check(): return
        modelo = self.combo_modelo.currentText()
        mapa = {
            "RGB":              procesadorImagen.cargar_imagen_opencv_rgb,
            "HSV":              procesadorImagen.conversion_imagen_opencv_hsv,
            "CMY":              procesadorImagen.conversion_imagen_opencv_cmy,
            "YIQ":              procesadorImagen.conversion_imagen_opencv_yiq,
            "HSI":              procesadorImagen.conversion_imagen_opencv_hsi,
            "Escala de Grises": procesadorImagen.cargar_imagen_opencv_gris,
        }
        fn = mapa.get(modelo)
        if fn:
            resp = fn(self.imagen_metadata)
            self.imagen_metadata = resp["objeto"]
            if resp["error"]:
                self._status(f"⚠  {resp['mensaje']}", C["warn"])
            else:
                # Conversión de modelo exitosa: actualizar base de binarización
                self._base_bin_datos  = self.imagen_metadata.datos.copy()
                self._base_bin_modelo = self.imagen_metadata.modelo
                self._mostrar_imagen()
                self._status(f"✔  Modelo aplicado: {self.imagen_metadata.modelo}")
                self.calcular_histograma()

    def binarizar_manual(self):
        if not self._check(): return
        if procesadorImagen.es_binaria(self.imagen_metadata):
            if self.imagen_metadata.es_resultado_logico:
                QMessageBox.information(self, "No se puede binarizar",
                    "La imagen es resultado de una operación lógica (AND / OR / XOR / NOT / Relacional) "
                    "y ya es binaria por definición — sus píxeles solo tienen valores 0 o 255.\n\n"
                    "Binarizarla de nuevo no añade información: cualquier umbral entre 0 y 254 "
                    "produce exactamente la misma imagen, porque los píxeles ya están en sus valores extremos.\n\n"
                    "Para explorar distintos umbrales sobre la imagen original, usa ↺ Restablecer."
                )
                return
            elif self._base_bin_datos is not None:
                self.imagen_metadata.datos  = self._base_bin_datos.copy()
                self.imagen_metadata.modelo = self._base_bin_modelo
            else:
                return
        else:
            self._base_bin_datos  = self.imagen_metadata.datos.copy()
            self._base_bin_modelo = self.imagen_metadata.modelo
        u = self.slider.value()
        resp = procesadorImagen.conversion_imagen_opencv_binaria(self.imagen_metadata, u)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen()
            self._status(f"✔  Binarización aplicada  ·  Umbral: {u}")
            self.calcular_histograma()

    def binarizar_otsu(self):
        if not self._check(): return
        if procesadorImagen.es_binaria(self.imagen_metadata):
            if self.imagen_metadata.es_resultado_logico:
                QMessageBox.information(self, "Otsu no aplicable",
                    "La imagen es resultado de una operación lógica (AND / OR / XOR / NOT / Relacional) "
                    "y ya es binaria por definición.\n\n"
                    "El algoritmo de Otsu necesita un histograma continuo de grises para encontrar el umbral óptimo "
                    "que separe dos poblaciones. Con solo dos valores posibles (0 y 255) el histograma tiene "
                    "exactamente dos barras y no hay separación que calcular.\n\n"
                    "Para aplicar Otsu desde la imagen original, usa ↺ Restablecer."
                )
                return
            elif self._base_bin_datos is not None:
                self.imagen_metadata.datos  = self._base_bin_datos.copy()
                self.imagen_metadata.modelo = self._base_bin_modelo
            else:
                return
        else:
            self._base_bin_datos  = self.imagen_metadata.datos.copy()
            self._base_bin_modelo = self.imagen_metadata.modelo

        resp = procesadorImagen.conversion_imagen_opencv_otsu(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            u = self.imagen_metadata.umbral

            # Sincronizar slider con el umbral calculado.
            # blockSignals evita que valueChanged dispare binarizar_manual()
            # encima del resultado de Otsu.
            self.slider.blockSignals(True)
            self.slider.setValue(int(u) if u else 128)
            self.lbl_umbral.setText(f"Umbral: {int(u) if u else 128}")
            self.slider.blockSignals(False)

            self.lbl_otsu_resultado.setText(f"Umbral óptimo encontrado: {int(u) if u else '—'}")
            self.lbl_otsu_resultado.setVisible(True)
            self._mostrar_imagen()
            self._status(f"✔  Otsu  ·  Umbral: {int(u) if u else '—'}")
            self.calcular_histograma()

    def mostrar_capas(self):
        if not self._check(): return
        import matplotlib.pyplot as plt
        conf = procesadorImagen.obtener_config_modelo(self.imagen_metadata.modelo)
        nombres = conf["nombres"]; mapas = conf["cmaps"]
        canales = cv2.split(self.imagen_metadata.datos)
        n = len(canales)
        bg = "#080C10"; fg = "#CDD9E5"
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        fig.patch.set_facecolor(bg)
        if n == 1: axes = [axes]
        for i, (c, ax) in enumerate(zip(canales, axes)):
            ax.set_facecolor(bg)
            nombre = nombres[i] if nombres and i < len(nombres) else f"Canal {i+1}"
            cmap   = mapas[i]   if mapas   and i < len(mapas)   else "gray"
            ax.imshow(c, cmap=cmap)
            ax.set_title(nombre, color=fg, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Canales  ·  {self.imagen_metadata.modelo}", color=fg, fontsize=11)
        plt.tight_layout()
        plt.show()

    def calcular_histograma(self):
        if not self._check(): return
        resp = procesadorImagen.proceso_histograma_completo(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
            return
        self._actualizar_stats(self.imagen_metadata.histograma)
        self._status("✔  Histograma calculado")

    def mostrar_grafica_histograma(self):
        if not self._check(): return
        if not self.imagen_metadata.histograma:
            self._status("⚠  Primero calcula el histograma", C["warn"])
            return
        import matplotlib.pyplot as plt
        conf = procesadorImagen.obtener_config_modelo(self.imagen_metadata.modelo)
        color_map = {
            "Reds":"#FF5555","Greens":"#50FA7B","Blues":"#8BE9FD",
            "hsv":"#FF79C6","gray":"#CDD9E5","Purples":"#BD93F9",
            "YlOrBr":"#FFB86C","coolwarm":"#6272A4",
            "Blues_r":"#8BE9FD","Purples_r":"#BD93F9","YlOrBr_r":"#FFB86C"
        }
        cmaps = conf.get("cmaps") or []
        colores = [color_map.get(c.replace("_r",""), "#CDD9E5") for c in cmaps]
        bg = "#080C10"; fg = "#CDD9E5"; grid_c = "#1E2D3D"
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor("#0D1117")
        for i, (canal, props) in enumerate(self.imagen_metadata.histograma.items()):
            color = colores[i] if i < len(colores) else "#CDD9E5"
            ax.plot(props["histograma_raw"], color=color, label=canal, alpha=0.85, linewidth=1.6)
        ax.set_title(
            f"Histograma Compuesto  ·  {self.imagen_metadata.modelo}  ·  {self.imagen_metadata.nombre}",
            color=fg, fontsize=10, pad=12
        )
        ax.set_xlabel("Intensidad (0–255)", color="#6B8FA8", fontsize=9)
        ax.set_ylabel("Frecuencia", color="#6B8FA8", fontsize=9)
        ax.tick_params(colors="#6B8FA8")
        ax.legend(facecolor="#0D1117", labelcolor=fg, fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.25, color=grid_c)
        for spine in ax.spines.values(): spine.set_edgecolor(grid_c)
        plt.tight_layout()
        plt.show()

    def restablecer_imagen(self):
        if not self._check(): return
        resp = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        self.lbl_otsu_resultado.setText("")
        self.lbl_otsu_resultado.setVisible(False)
        if not resp["error"]:
            self._datos_originales = self.imagen_metadata.datos.copy()
            self._base_bin_datos   = self.imagen_metadata.datos.copy()
            self._base_bin_modelo  = self.imagen_metadata.modelo
            self._mostrar_imagen()
            self._status("✔  Imagen restablecida a RGB original")
            self.calcular_histograma()

    def guardar_imagen_actual(self):
        if not self._check(): return

        carpeta = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de destino",
            self._leer_ultimo_directorio_guardado()
        )
        if not carpeta: return
        self._guardar_ultimo_directorio_guardado(carpeta)

        archivos_guardados = []
        errores = []

        # 1. Guardar imagen principal
        resp = procesadorImagen.guardar_imagen(self.imagen_metadata, carpeta)
        if resp["error"]:
            errores.append(resp["mensaje"])
        else:
            archivos_guardados.extend(resp["archivos"])

        # 2. Guardar histograma si está marcado
        if self.chk_histograma.isChecked():
            resp_h = procesadorImagen.guardar_histograma(self.imagen_metadata, carpeta)
            if resp_h["error"]:
                errores.append(resp_h["mensaje"])
            else:
                archivos_guardados.extend(resp_h["archivos"])

        # 3. Guardar canales si está marcado
        if self.chk_canales.isChecked():
            resp_c = procesadorImagen.guardar_canales(self.imagen_metadata, carpeta)
            if resp_c["error"]:
                errores.append(resp_c["mensaje"])
            else:
                archivos_guardados.extend(resp_c["archivos"])

        # 4. Guardar figura de conteo de objetos si está marcado
        if self.chk_conteo.isChecked():
            resp_co = procesadorImagen.guardar_conteo_vecindad(self.imagen_metadata, carpeta)
            if resp_co["error"]:
                errores.append(resp_co["mensaje"])
            else:
                archivos_guardados.extend(resp_co["archivos"])

        # 4. Reporte de resultado en status bar
        n = len(archivos_guardados)
        carpeta_nombre = os.path.basename(carpeta) or carpeta
        if n > 0 and not errores:
            self._status(f"✔  {n} archivo(s) guardado(s) en: {carpeta_nombre}", C["success"])
        elif n > 0 and errores:
            self._status(f"⚠  {n} guardado(s) · {len(errores)} error(es): {errores[0]}", C["warn"])
        else:
            self._status(f"⚠  {errores[0] if errores else 'No se guardó ningún archivo'}", C["warn"])

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    #  ACCIONES — SEGMENTACIÓN
    # ══════════════════════════════════════════════════════════════

    def _seg_cargar_imagen_b(self):
        """Carga la imagen secundaria B desde disco y la persiste en self.imagen_metadata_b."""
        ruta = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen secundaria (B)",
            self._leer_ultimo_directorio(),
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.webp *.pgm *.ppm)"
        )[0]
        if not ruta:
            return
        self._guardar_ultimo_directorio(ruta)
        self.imagen_metadata_b = metadataImagen(ruta)
        resp = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata_b)
        self.imagen_metadata_b = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  Error al cargar imagen B: {resp['mensaje']}", C["warn"])
            self.imagen_metadata_b = None
            self.lbl_imagen_b.setText("Sin imagen B cargada")
            self.thumb_imagen_b.setVisible(False)
        else:
            # Generar thumbnail con el mismo tamaño que el carrusel (70×56 / 66×52)
            datos = self.imagen_metadata_b.datos
            if len(datos.shape) == 2:
                h, w = datos.shape
                qimg = QImage(datos.data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, ch = datos.shape
                qimg = QImage(datos.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap_b = QPixmap.fromImage(qimg).scaled(
                66, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_imagen_b.setPixmap(pixmap_b)
            self.thumb_imagen_b.setVisible(True)
            _nb = self.imagen_metadata_b.nombre
            self.lbl_imagen_b.setText(_nb[:42] + "…" if len(_nb) > 42 else _nb)
            self._status(f"✔  Imagen B cargada: {self.imagen_metadata_b.nombre}")

    def _seg_check_imagen_b(self):
        """Verifica que imagen B esté cargada. Retorna False y muestra aviso si no."""
        if self.imagen_metadata_b is None or self.imagen_metadata_b.datos is None:
            QMessageBox.warning(self, "Sin imagen B",
                "Carga una imagen secundaria (B) antes de aplicar esta operación.")
            return False
        return True

    # ── Ruido ─────────────────────────────────────────────────────

    def _seg_aplicar_ruido_sal(self):
        if not self._check(): return
        cantidad = self.slider_sal.value() / 100.0
        resp = procesadorImagen.agregar_ruido_sal(self.imagen_metadata, cantidad)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_aplicar_ruido_pimienta(self):
        if not self._check(): return
        cantidad = self.slider_pimienta.value() / 100.0
        resp = procesadorImagen.agregar_ruido_pimienta(self.imagen_metadata, cantidad)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_aplicar_ruido_gaussiano(self):
        if not self._check(): return
        sigma = self.slider_sigma.value()
        resp = procesadorImagen.agregar_ruido_gaussiano(self.imagen_metadata, media=0, sigma=sigma)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False)
            self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            # Gaussiano produce GRIS: actualizar base para que el slider pueda binarizar sobre la imagen con ruido
            self._base_bin_datos  = self.imagen_metadata.datos.copy()
            self._base_bin_modelo = self.imagen_metadata.modelo

    def _seg_retirar_ruido(self):
        """Recarga la imagen desde disco descartando cualquier ruido acumulado en memoria.
        Mantiene el modelo de color y umbral previos si la imagen era binaria."""
        if not self._check(): return
        modelo_previo = self.imagen_metadata.modelo
        umbral_previo = self.imagen_metadata.umbral

        # Recargar RGB limpio desde disco
        resp = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
            return

        # Re-aplicar el modelo que tenía antes del ruido
        mapa = {
            "BINARIO":          lambda m: procesadorImagen.conversion_imagen_opencv_binaria(m, umbral_previo if umbral_previo else 128),
            "GRIS":             procesadorImagen.cargar_imagen_opencv_gris,
            "HSV":              procesadorImagen.conversion_imagen_opencv_hsv,
            "CMY":              procesadorImagen.conversion_imagen_opencv_cmy,
            "YIQ":              procesadorImagen.conversion_imagen_opencv_yiq,
            "HSI":              procesadorImagen.conversion_imagen_opencv_hsi,
        }
        fn = mapa.get(modelo_previo)
        if fn:
            resp2 = fn(self.imagen_metadata)
            self.imagen_metadata = resp2["objeto"]
            if resp2["error"]:
                self._status(f"⚠  No se pudo restaurar el modelo {modelo_previo}: {resp2['mensaje']}", C["warn"])
                return

        # Actualizar base de binarización con los datos limpios
        if self.imagen_metadata.modelo != "BINARIO":
            self._base_bin_datos  = self.imagen_metadata.datos.copy()
            self._base_bin_modelo = self.imagen_metadata.modelo

        self._mostrar_imagen()
        self.calcular_histograma()
        self._status(f"✔  Ruido retirado — imagen restaurada desde disco  ·  {self.imagen_metadata.modelo}")

    # ── Operaciones aritméticas ───────────────────────────────────

    def _seg_sumar(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.sumar_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _seg_restar(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.restar_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    def _seg_multiplicar(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.multiplicar_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")

    # ── Operaciones lógicas ───────────────────────────────────────

    def _seg_and(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.and_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_or(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.or_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_xor(self):
        if not self._check() or not self._seg_check_imagen_b(): return
        resp = procesadorImagen.xor_imagenes(self.imagen_metadata, self.imagen_metadata_b)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_not(self):
        if not self._check(): return
        resp = procesadorImagen.not_imagen(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    # ── Operaciones relacionales ──────────────────────────────────

    def _seg_relacional_mayor(self):
        if not self._check(): return
        umbral = self.slider_umbral_rel.value()
        resp = procesadorImagen.relacional_mayor(self.imagen_metadata, umbral)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_relacional_menor(self):
        if not self._check(): return
        umbral = self.slider_umbral_rel.value()
        resp = procesadorImagen.relacional_menor(self.imagen_metadata, umbral)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    def _seg_relacional_igual(self):
        if not self._check(): return
        umbral = self.slider_umbral_rel.value()
        resp = procesadorImagen.relacional_igual(self.imagen_metadata, umbral)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(es_derivable=False); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._base_bin_datos = None
            self._base_bin_modelo = None

    # ── Conteo de objetos ─────────────────────────────────────────

    def _seg_mostrar_conteo(self, resp4=None, resp8=None):
        """Actualiza los labels de resultado de conteo y abre la figura matplotlib."""
        import matplotlib.pyplot as plt
        import cv2 as _cv2

        if resp4 and not resp4["error"] and resp8 is None:
            # Solo vecindad-4
            n = resp4["num_objetos"]
            self.lbl_conteo_v4.setText(f"Vecindad-4:  {n} objeto(s) detectado(s)")
            self.lbl_conteo_v4.setVisible(True)
            self.lbl_conteo_v8.setVisible(False)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(resp4["labels"], cmap="jet")
            axs[0].set_title(f"Vecindad-4 — etiquetas ({n} obj.)")
            axs[0].axis("off")
            axs[1].imshow(_cv2.cvtColor(resp4["imagen_contornos"], _cv2.COLOR_BGR2RGB))
            axs[1].set_title("Contornos numerados")
            axs[1].axis("off")
            plt.tight_layout()
            self._fig_conteo = fig
            plt.show()

        elif resp8 and not resp8["error"] and resp4 is None:
            # Solo vecindad-8
            n = resp8["num_objetos"]
            self.lbl_conteo_v4.setVisible(False)
            self.lbl_conteo_v8.setText(f"Vecindad-8:  {n} objeto(s) detectado(s)")
            self.lbl_conteo_v8.setVisible(True)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(resp8["labels"], cmap="jet")
            axs[0].set_title(f"Vecindad-8 — etiquetas ({n} obj.)")
            axs[0].axis("off")
            axs[1].imshow(_cv2.cvtColor(resp8["imagen_contornos"], _cv2.COLOR_BGR2RGB))
            axs[1].set_title("Contornos numerados")
            axs[1].axis("off")
            plt.tight_layout()
            self._fig_conteo = fig
            plt.show()

        elif resp4 and resp8:
            # Comparación — mostrar ambos labels apilados
            n4, n8 = resp4["num_objetos"], resp8["num_objetos"]
            self.lbl_conteo_v4.setText(f"Vecindad-4:  {n4} objeto(s)  ·  Δ {abs(n4-n8)}")
            self.lbl_conteo_v8.setText(f"Vecindad-8:  {n8} objeto(s)")
            self.lbl_conteo_v4.setVisible(True)
            self.lbl_conteo_v8.setVisible(True)

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Comparación de vecindad  ·  [{self.imagen_metadata.nombre}]", fontsize=12)
            axs[0,0].imshow(resp4["labels"], cmap="jet")
            axs[0,0].set_title(f"V-4 etiquetas ({n4} obj.)"); axs[0,0].axis("off")
            axs[0,1].imshow(_cv2.cvtColor(resp4["imagen_contornos"], _cv2.COLOR_BGR2RGB))
            axs[0,1].set_title("V-4 contornos"); axs[0,1].axis("off")
            axs[1,0].imshow(resp8["labels"], cmap="jet")
            axs[1,0].set_title(f"V-8 etiquetas ({n8} obj.)"); axs[1,0].axis("off")
            axs[1,1].imshow(_cv2.cvtColor(resp8["imagen_contornos"], _cv2.COLOR_BGR2RGB))
            axs[1,1].set_title("V-8 contornos"); axs[1,1].axis("off")
            plt.tight_layout()
            self._fig_conteo = fig
            plt.show()

    def _seg_vecindad_4(self):
        if not self._check(): return
        resp = procesadorImagen.analizar_vecindad_4(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._seg_mostrar_conteo(resp4=resp)

    def _seg_vecindad_8(self):
        if not self._check(): return
        resp = procesadorImagen.analizar_vecindad_8(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            self._mostrar_imagen(); self.calcular_histograma()
            self._status(f"✔  {resp['mensaje']}")
            self._seg_mostrar_conteo(resp8=resp)

    def _seg_comparar_vecindad(self):
        if not self._check(): return
        resp4 = procesadorImagen.analizar_vecindad_4(self.imagen_metadata)
        resp8 = procesadorImagen.analizar_vecindad_8(self.imagen_metadata)
        # Persistir el estado de vecindad-8 (más informativa) como imagen actual
        self.imagen_metadata = resp8["objeto"]
        if resp4["error"]:
            self._status(f"⚠  V-4: {resp4['mensaje']}", C["warn"]); return
        if resp8["error"]:
            self._status(f"⚠  V-8: {resp8['mensaje']}", C["warn"]); return
        self._mostrar_imagen(); self.calcular_histograma()
        self._status(f"✔  Comparación completada — V4:{resp4['num_objetos']} obj. / V8:{resp8['num_objetos']} obj.")
        self._seg_mostrar_conteo(resp4=resp4, resp8=resp8)

    def _check(self):
        if not self.imagen_metadata or self.imagen_metadata.datos is None:
            QMessageBox.warning(self, "Sin imagen", "Primero carga una imagen.")
            return False
        return True

    def _mostrar_imagen(self, registrar=True, es_derivable=True):
        datos = self.imagen_metadata.datos
        if datos is None: return
        if len(datos.shape) == 2:
            h, w = datos.shape
            qimg = QImage(datos.data, w, h, w, QImage.Format_Grayscale8)
        else:
            if datos.dtype != np.uint8:
                datos = (datos * 255).clip(0, 255).astype(np.uint8)
            h, w, ch = datos.shape
            qimg = QImage(datos.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        visor_size = self.scroll_visor.size() - QSize(4, 4)
        scaled = pixmap.scaled(visor_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_imagen.setPixmap(scaled)
        self.lbl_imagen.setMinimumSize(1, 1)

        h2, w2 = self.imagen_metadata.datos.shape[:2]
        self.lbl_dims.setText(f"{w2} × {h2} px")
        self.chip_modelo.setText(self.imagen_metadata.modelo)

        # Solo agregar al carrusel si es una acción nueva (no una restauración)
        if registrar:
            self._agregar_carrusel(pixmap, self.imagen_metadata.modelo, es_derivable)

    def _agregar_carrusel(self, pixmap, etiqueta, es_derivable=True):
        # Crear y guardar el estado en el historial
        entrada = metadataHistorialImagen(self.imagen_metadata.ruta)
        entrada.modelo       = self.imagen_metadata.modelo
        entrada.umbral       = self.imagen_metadata.umbral
        entrada.histograma   = self.imagen_metadata.histograma.copy()
        entrada.thumbnail    = pixmap
        entrada.es_derivable = es_derivable
        entrada.es_resultado_logico = self.imagen_metadata.es_resultado_logico
        if not es_derivable:
            entrada.datos = self.imagen_metadata.datos.copy()
        self.historial_estados.append(entrada)
        index = len(self.historial_estados) - 1

        self._build_thumb_widget(index, entrada)

        QApplication.processEvents()
        self.scroll_carrusel.horizontalScrollBar().setValue(
            self.scroll_carrusel.horizontalScrollBar().maximum()
        )

    def _build_thumb_widget(self, index, entrada):
        """Construye y agrega al carrusel el widget visual para una entrada del historial."""
        container = ThumbWidget(index)
        container.clicked.connect(self._restaurar_estado)
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(70, 56)
        thumb_lbl.setPixmap(entrada.thumbnail.scaled(66, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setStyleSheet(f"""
            border: 1px solid {C['border2']};
            border-radius: 3px;
            background: {C['surface2']};
            padding: 2px;
        """)

        # Botón × en esquina superior derecha del thumbnail
        btn_del = QPushButton("×", thumb_lbl)
        btn_del.setFixedSize(16, 16)
        btn_del.move(53, 1)   # 70 - 16 - 1, 1
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.setStyleSheet(f"""
            QPushButton {{
                background: {C['surface']};
                color: {C['warn']};
                border: 1px solid {C['border2']};
                border-radius: 3px;
                font-size: 11pt;
                font-weight: bold;
                padding: 0px;
            }}
            QPushButton:hover {{
                background: {C['warn']};
                color: {C['bg']};
            }}
        """)
        btn_del.clicked.connect(lambda _checked=False, i=index: self._eliminar_del_carrusel(i))

        caption = QLabel(entrada.modelo)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(f"color:{C['text3']}; font-size:10pt;")

        col.addWidget(thumb_lbl)
        col.addWidget(caption)
        self.carrusel_layout.addWidget(container)

    def _eliminar_del_carrusel(self, index):
        """Elimina una entrada del historial y reconstruye el carrusel."""
        if index < 0 or index >= len(self.historial_estados): return
        del self.historial_estados[index]
        self._refrescar_carrusel()

    def _refrescar_carrusel(self):
        """Limpia y reconstruye todos los thumbnails del carrusel desde historial_estados."""
        while self.carrusel_layout.count():
            item = self.carrusel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, entrada in enumerate(self.historial_estados):
            self._build_thumb_widget(i, entrada)

    def _restaurar_estado(self, index):
        """Restaura la imagen al estado guardado en el carrusel. No agrega al historial."""
        if not self._check(): return
        if index < 0 or index >= len(self.historial_estados): return

        entrada = self.historial_estados[index]

        # ── Rama rápida: estado no derivable ─────────────────────────────
        if not entrada.es_derivable and entrada.datos is not None:
            self.imagen_metadata.datos             = entrada.datos.copy()
            self.imagen_metadata.modelo            = entrada.modelo
            self.imagen_metadata.umbral            = entrada.umbral
            self.imagen_metadata.es_resultado_logico = entrada.es_resultado_logico
            self._mostrar_imagen(registrar=False)
            self.calcular_histograma()
            self._status(f"↩  Estado restaurado: {entrada.nombre}  ·  {entrada.modelo}")
            return

        # ── Rama estándar: reconstruir desde disco ───────────────────────
        # Si la entrada pertenece a una imagen diferente, recargar desde su ruta
        # y actualizar _datos_originales para que el slider de binarización funcione correctamente
        if entrada.ruta != self.imagen_metadata.ruta:
            self.imagen_metadata = metadataImagen(entrada.ruta)
            resp_rgb = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata)
            self.imagen_metadata = resp_rgb["objeto"]
            if resp_rgb["error"]:
                self._status(f"⚠  No se pudo cargar la imagen base: {resp_rgb['mensaje']}", C["warn"])
                return
            self._datos_originales = self.imagen_metadata.datos.copy()

            # Actualizar chips y label del archivo en la UI
            nombre = self.imagen_metadata.nombre
            self.lbl_info_archivo.setText(nombre[:42] + "…" if len(nombre) > 42 else nombre)
            self.chip_archivo.setText(nombre[:20] + "…" if len(nombre) > 20 else nombre)

        # Partir desde datos RGB limpios antes de aplicar el modelo guardado
        self.imagen_metadata.datos = self._datos_originales.copy()
        self.imagen_metadata.modelo = "RGB"
        self.imagen_metadata.umbral = None

        resp = procesadorImagen.cargar_estado_historial(entrada, self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]

        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
            return

        # Restaurar flag de origen lógico
        self.imagen_metadata.es_resultado_logico = entrada.es_resultado_logico

        # Actualizar el slider si el estado restaurado era una binarización manual
        if entrada.modelo == "BINARIO" and entrada.umbral is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(int(entrada.umbral))
            self.lbl_umbral.setText(f"Umbral: {int(entrada.umbral)}")
            self.slider.blockSignals(False)

        # registrar=False → no duplicar en el carrusel
        self._mostrar_imagen(registrar=False)
        self.calcular_histograma()
        self._status(f"↩  Estado restaurado: {entrada.nombre}  ·  {entrada.modelo}")
    def _actualizar_stats(self, histograma):
        # Limpiar widgets de estadísticas
        for i in reversed(range(self.stats_layout.count())):
            item = self.stats_layout.itemAt(i)
            if item.widget(): item.widget().deleteLater()

        # Actualizar tarjetas de resumen con datos del primer canal
        if histograma:
            primer = list(histograma.values())[0]
            self.card_media.set_valor(f"{primer['Media']:.1f}")
            self.card_var.set_valor(f"{primer['Varianza']:.1f}")
            self.card_energia.set_valor(f"{primer['Energía']:.4f}")
            self.card_entrop.set_valor(f"{primer['Entropía']:.2f}")

        # Detalle por canal
        propiedades_orden = ["Energía", "Entropía", "Asimetría", "Media", "Varianza"]
        for canal, props in histograma.items():
            lbl_canal = QLabel(f"▸  {canal}")
            lbl_canal.setObjectName("canal_header")
            self.stats_layout.addWidget(lbl_canal)

            grid = QGridLayout()
            grid.setContentsMargins(0, 4, 0, 8)
            grid.setSpacing(4)
            row = 0
            for prop in propiedades_orden:
                valor = props.get(prop)
                if valor is None: continue
                lbl_k = QLabel(f"{prop}:")
                lbl_k.setObjectName("stat_key")
                lbl_v = QLabel(f"{valor:.4f}")
                lbl_v.setObjectName("stat_val")
                grid.addWidget(lbl_k, row, 0)
                grid.addWidget(lbl_v, row, 1)
                row += 1
            grid_w = QWidget()
            grid_w.setLayout(grid)
            self.stats_layout.addWidget(grid_w)

            sep = QFrame(); sep.setObjectName("hline")
            self.stats_layout.addWidget(sep)

        # ── Redibujar canvas embebido ──────────
        self._redibujar_canvas(histograma)

    def _redibujar_canvas(self, histograma):
        """Actualiza la gráfica embebida en el panel derecho."""
        conf = procesadorImagen.obtener_config_modelo(self.imagen_metadata.modelo)
        color_map = {
            "Reds": "#FF5555", "Greens": "#50FA7B", "Blues": "#8BE9FD",
            "hsv": "#FF79C6", "gray": "#CDD9E5", "Purples": "#BD93F9",
            "YlOrBr": "#FFB86C", "coolwarm": "#6272A4",
            "Blues_r": "#8BE9FD", "Purples_r": "#BD93F9", "YlOrBr_r": "#FFB86C"
        }
        cmaps = conf.get("cmaps") or []
        colores = [color_map.get(c.replace("_r", ""), "#CDD9E5") for c in cmaps]

        self.ax_hist.clear()
        self.ax_hist.set_facecolor("#0D1117")
        self.ax_hist.tick_params(colors="#6B8FA8", labelsize=7)
        for spine in self.ax_hist.spines.values():
            spine.set_edgecolor("#1E2D3D")
        self.ax_hist.set_xlabel("Intensidad (0–255)", color="#6B8FA8", fontsize=7)
        self.ax_hist.set_ylabel("Frecuencia", color="#6B8FA8", fontsize=7)
        self.ax_hist.grid(True, linestyle="--", alpha=0.2, color="#1E2D3D")

        for i, (canal, props) in enumerate(histograma.items()):
            color = colores[i] if i < len(colores) else "#CDD9E5"
            self.ax_hist.plot(props["histograma_raw"], color=color,
                              label=canal, alpha=0.85, linewidth=1.2)

        self.ax_hist.legend(facecolor="#0D1117", labelcolor="#CDD9E5",
                            fontsize=7, framealpha=0.8)
        self.fig_canvas.tight_layout(pad=0.8)
        self.canvas_widget.draw()

    def _status(self, msg, color=None):
        c = color or C["text2"]
        self.statusbar_lbl.setText(f"  ◉  {msg}")
        self.statusbar_lbl.setStyleSheet(f"""
            background:{C['surface']};
            border-top:1px solid {C['border']};
            color:{c};
            font-size:11pt;
            padding: 0 12px;
        """)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Soporte HiDPI / pantallas de alta resolución ──────────
    # Debe ir ANTES de crear QApplication
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── Fuente global: 14pt garantiza legibilidad en cualquier DPI ──
    fuente_global = QFont("Consolas", 14)
    fuente_global.setStyleHint(QFont.Monospace)
    app.setFont(fuente_global)

    ventana = VentanaDashboard()

    # Tamaño de restauración
    ventana.resize(1280, 720)

    # Centrar la ventana en pantalla para cuando el usuario restaure desde maximizado
    pantalla = app.desktop().availableGeometry()
    x = (pantalla.width()  - 1280) // 2
    y = (pantalla.height() - 720)  // 2
    ventana.move(x, y)

    ventana.showMaximized()
    sys.exit(app.exec_())