"""Tema oscuro de la interfaz."""

ACCENT = "#ff8a1f"
BG = "#16161c"
PANEL = "#1f1f27"
PANEL2 = "#272732"
TEXT = "#e8e8ee"
MUTED = "#9a9aab"
GOOD = "#4ec27a"
WARN = "#f2c14e"
BAD = "#ef5a5a"

LANE_COLORS = ["#2ecc52", "#e8383d", "#f2d22e", "#3d8ff2", "#ff8c1a"]
OPEN_COLOR = "#b45cff"
SP_COLOR = "#7fe7ff"
REVIEW_COLOR = "#ff4fd8"

STYLE = f"""
* {{ font-family: 'Segoe UI'; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog, QWidget#root {{ background: {BG}; }}
QFrame#panel, QGroupBox {{ background: {PANEL}; border: 1px solid #2e2e3a; border-radius: 10px; }}
QGroupBox {{ margin-top: 14px; padding-top: 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QLabel#title {{ font-size: 18pt; font-weight: 700; }}
QLabel#subtitle {{ color: {MUTED}; }}
QLabel#drop {{ border: 2px dashed #44445a; border-radius: 12px; color: {MUTED}; font-size: 11pt; padding: 18px; }}
QLabel#drop[hover="true"] {{ border-color: {ACCENT}; color: {TEXT}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid #3a3a48; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #666; border-color: #2a2a33; }}
QPushButton#primary {{ background: {ACCENT}; color: #1a1a1a; font-weight: 700; border: none; }}
QPushButton#primary:hover {{ background: #ffa04a; }}
QPushButton#primary:disabled {{ background: #6b4a2a; color: #333; }}
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{ background: #121218; border: 1px solid #33333f;
    border-radius: 6px; padding: 5px; selection-background-color: {ACCENT}; }}
QPlainTextEdit#console {{ font-family: 'Cascadia Mono', 'Consolas'; font-size: 9pt; background: #0e0e12; }}
QTableWidget, QListWidget {{ background: #121218; border: 1px solid #2e2e3a; border-radius: 8px;
    gridline-color: #24242e; alternate-background-color: #17171e; }}
QHeaderView::section {{ background: {PANEL2}; border: none; padding: 6px; color: {MUTED}; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: #3a2a1a; }}
QProgressBar {{ background: #121218; border: 1px solid #33333f; border-radius: 6px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QTabWidget::pane {{ border: 1px solid #2e2e3a; border-radius: 8px; }}
QTabBar::tab {{ background: {PANEL}; padding: 7px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {PANEL2}; color: {ACCENT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QSlider::groove:horizontal {{ height: 6px; background: #2a2a35; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; border-radius: 7px; background: {ACCENT}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3a3a48; border-radius: 5px; min-height: 30px; }}
QToolTip {{ background: {PANEL2}; border: 1px solid {ACCENT}; color: {TEXT}; }}
"""
