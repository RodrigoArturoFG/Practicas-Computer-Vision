# =====================================================================
# PRÁCTICA 2 - "EXPLORANDO LA IMAGEN DIGITAL CON PYTHON"
# VERSIÓN 2: Layout 3 Paneles (Dashboard)
# Autor: Rodrigo Arturo Fernández González
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
    QStyledItemDelegate
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
    padding: 8px 14px;
    font-size: 12pt;
    min-height: 36px;
}}
QComboBox:focus {{ border-color: {C['accent']}; }}
QComboBox QAbstractItemView {{
    background: {C['surface2']};
    color: #000000;
    border: 1px solid {C['accent']};
    selection-background-color: {C['accent_dim']};
    selection-color: {C['accent']};
    outline: none;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}

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
#  DELEGATE  —  Fuerza color de texto en ComboBox dropdown
# ══════════════════════════════════════════════════════════════
class DarkTextDelegate(QStyledItemDelegate):
    """Fuerza el texto a negro en el dropdown del ComboBox (override de Windows)."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.palette.setColor(option.palette.Text, QColor("#111111"))
        option.palette.setColor(option.palette.HighlightedText, QColor("#00D4AA"))


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
        self.imagen_metadata = None
        self._datos_originales = None
        self.historial_pixmaps = []
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
        w.setStyleSheet(f"background:{C['surface']}; border-bottom:1px solid {C['border']};")
        layout = QHBoxLayout(w)
        layout.setContentsMargins(20, 0, 20, 0)

        dot_green = QLabel("◉")
        dot_green.setStyleSheet(f"color:{C['accent']}; font-size:14px;")

        titulo = QLabel("VISION LAB")
        titulo.setStyleSheet(f"color:{C['text']}; font-size:18pt; font-weight:bold; letter-spacing:4px;")

        sep = QLabel("  ·  ")
        sep.setStyleSheet(f"color:{C['text3']};")

        subtitulo = QLabel("Explorador de Imagen Digital  ·  Práctica 1")
        subtitulo.setStyleSheet(f"color:{C['text2']}; font-size:12pt;")

        self.chip_modelo = QLabel("RGB")
        self.chip_modelo.setObjectName("chip")

        self.chip_archivo = QLabel("sin archivo")
        self.chip_archivo.setObjectName("chip_b")

        layout.addWidget(dot_green)
        layout.addSpacing(8)
        layout.addWidget(titulo)
        layout.addWidget(sep)
        layout.addWidget(subtitulo)
        layout.addStretch()
        layout.addWidget(QLabel("MODELO:").setParent(w) or QLabel(""))
        layout.addWidget(self.chip_modelo)
        layout.addSpacing(12)
        layout.addWidget(self.chip_archivo)
        return w

    # ── Sidebar (panel izquierdo) ─────────────────
    def _make_sidebar(self):
        w = QWidget()
        w.setObjectName("sidebar")
        w.setFixedWidth(380)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

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
        outer.addWidget(scroll, 1)

        # ── 1. Cargar imagen ────────────────────
        layout.addWidget(self._section_label("01  ·  ARCHIVO"))
        btn_cargar = QPushButton("⊕  Cargar Imagen...")
        btn_cargar.setObjectName("primary")
        btn_cargar.setCursor(Qt.PointingHandCursor)
        btn_cargar.clicked.connect(self.seleccionar_imagen)
        layout.addWidget(btn_cargar)

        self.lbl_info_archivo = QLabel("Sin imagen seleccionada")
        self.lbl_info_archivo.setObjectName("info")
        self.lbl_info_archivo.setWordWrap(True)
        layout.addWidget(self.lbl_info_archivo)

        layout.addWidget(self._hline())

        # ── 2. Modelo de color ──────────────────
        layout.addWidget(self._section_label("02  ·  MODELO DE COLOR"))
        self.combo_modelo = QComboBox()
        self.combo_modelo.addItems(["RGB", "HSV", "CMY", "YIQ", "HSI", "Escala de Grises"])
        self.combo_modelo.setItemDelegate(DarkTextDelegate(self.combo_modelo))
        layout.addWidget(self.combo_modelo)

        btn_modelo = QPushButton("Aplicar Modelo →")
        btn_modelo.setObjectName("secondary")
        btn_modelo.setCursor(Qt.PointingHandCursor)
        btn_modelo.clicked.connect(self.aplicar_modelo)
        layout.addWidget(btn_modelo)

        layout.addWidget(self._hline())

        # ── 3. Separar Canales ──────────────────
        layout.addWidget(self._section_label("03  ·  CANALES"))
        btn_capas = QPushButton("◫  Separar y Visualizar Capas")
        btn_capas.setCursor(Qt.PointingHandCursor)
        btn_capas.clicked.connect(self.mostrar_capas)
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

        btn_bin_man = QPushButton("⬛  Binarizar (Manual)")
        btn_bin_man.setCursor(Qt.PointingHandCursor)
        btn_bin_man.clicked.connect(self.binarizar_manual)
        layout.addWidget(btn_bin_man)

        btn_bin_otsu = QPushButton("⚙  Binarización Automática (Otsu)")
        btn_bin_otsu.setCursor(Qt.PointingHandCursor)
        btn_bin_otsu.clicked.connect(self.binarizar_otsu)
        layout.addWidget(btn_bin_otsu)

        self.lbl_otsu_resultado = QLabel("")
        self.lbl_otsu_resultado.setObjectName("warn")
        layout.addWidget(self.lbl_otsu_resultado)

        layout.addStretch()
        layout.addWidget(self._hline())

        # ── Reset ────────────────────────────────
        btn_reset = QPushButton("↺  Restablecer Imagen Original")
        btn_reset.setObjectName("danger")
        btn_reset.setCursor(Qt.PointingHandCursor)
        btn_reset.clicked.connect(self.restablecer_imagen)
        layout.addWidget(btn_reset)

        return w

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

    def seleccionar_imagen(self):
        ruta_inicial = os.path.join(config.script_dir_parent, 'resources', 'input')
        # Si la carpeta no existe, abrir desde el directorio del script
        if not os.path.isdir(ruta_inicial):
            ruta_inicial = config.script_dir_parent

        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", ruta_inicial,
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if not ruta: return
        self.imagen_metadata = metadataImagen(ruta)
        resp = procesadorImagen.cargar_imagen_opencv_rgb(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]:
            self._status(f"⚠  {resp['mensaje']}", C["warn"])
            return
        # Guardar copia de los datos RGB originales para el slider de binarización
        self._datos_originales = self.imagen_metadata.datos.copy()
        nombre = self.imagen_metadata.nombre
        self.lbl_info_archivo.setText(nombre)
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
                self._mostrar_imagen()
                self._status(f"✔  Modelo aplicado: {self.imagen_metadata.modelo}")
                self.calcular_histograma()

    def binarizar_manual(self):
        if not self._check(): return
        # Restaurar datos originales antes de binarizar para que el slider pueda moverse libremente
        if hasattr(self, '_datos_originales') and self._datos_originales is not None:
            self.imagen_metadata.datos = self._datos_originales.copy()
            self.imagen_metadata.modelo = "RGB"
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
        resp = procesadorImagen.conversion_imagen_opencv_otsu(self.imagen_metadata)
        self.imagen_metadata = resp["objeto"]
        if resp["error"]: self._status(f"⚠  {resp['mensaje']}", C["warn"])
        else:
            u = self.imagen_metadata.umbral
            self.lbl_otsu_resultado.setText(f"Umbral óptimo encontrado: {int(u) if u else '—'}")
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
        if not resp["error"]:
            self._datos_originales = self.imagen_metadata.datos.copy()
            self._mostrar_imagen()
            self._status("✔  Imagen restablecida a RGB original")
            self.calcular_histograma()

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    def _check(self):
        if not self.imagen_metadata or self.imagen_metadata.datos is None:
            QMessageBox.warning(self, "Sin imagen", "Primero carga una imagen.")
            return False
        return True

    def _mostrar_imagen(self):
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
        self.lbl_imagen.setMinimumSize(1, 1)  # permite que el layout lo maneje libremente

        h2, w2 = self.imagen_metadata.datos.shape[:2]
        self.lbl_dims.setText(f"{w2} × {h2} px")
        self.chip_modelo.setText(self.imagen_metadata.modelo)
        self._agregar_carrusel(pixmap, self.imagen_metadata.modelo)

    def _agregar_carrusel(self, pixmap, etiqueta):
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(70, 56)
        thumb_lbl.setPixmap(pixmap.scaled(66, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setStyleSheet(f"""
            border: 1px solid {C['border2']};
            border-radius: 3px;
            background: {C['surface2']};
            padding: 2px;
        """)

        caption = QLabel(etiqueta)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(f"color:{C['text3']}; font-size:10pt;")

        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(thumb_lbl)
        col.addWidget(caption)
        self.carrusel_layout.addWidget(container)

        QApplication.processEvents()
        self.scroll_carrusel.horizontalScrollBar().setValue(
            self.scroll_carrusel.horizontalScrollBar().maximum()
        )

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