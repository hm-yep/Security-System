"""统一的矢量线条图标：内联 SVG + QSvgRenderer，不依赖图片资源。

所有图标都是 24×24 视图框的描边图形，颜色与尺寸可调，按 2 倍像素密度渲染，
在高分屏上保持锐利。主界面、托盘、对话框共用同一套图标，避免 emoji 风格不一致。
"""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer

ACCENT = "#38bdf8"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#f87171"
MUTED = "#7c8ba1"
TEXT = "#e6eefc"

_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
    'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
    'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
)

_BODIES = {
    "target": (
        '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.5"/>'
        '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'
    ),
    "scan": (
        '<path d="M4 8V5a1 1 0 0 1 1-1h3"/><path d="M16 4h3a1 1 0 0 1 1 1v3"/>'
        '<path d="M20 16v3a1 1 0 0 1-1 1h-3"/><path d="M8 20H5a1 1 0 0 1-1-1v-3"/>'
        '<path d="M4 12h16"/>'
    ),
    "gauge": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-5"/><circle cx="12" cy="18" r="1"/>',
    "activity": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
    "shield": (
        '<path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6l7-3z"/>'
        '<path d="M9 12l2 2 4-4"/>'
    ),
    "user": '<circle cx="12" cy="8" r="3.5"/><path d="M5 20c1.5-3.5 4-5 7-5s5.5 1.5 7 5"/>',
    "users": (
        '<circle cx="9" cy="8" r="3"/><path d="M3 19c1.2-3 3.2-4.5 6-4.5s4.8 1.5 6 4.5"/>'
        '<path d="M16 5.5a3 3 0 0 1 0 5.5"/><path d="M18 14.5c2 .5 3.3 2 4 4.5"/>'
    ),
    "key": '<circle cx="8" cy="12" r="4"/><path d="M12 12h8"/><path d="M17 12v3"/><path d="M20 12v2"/>',
    "clipboard": (
        '<rect x="6" y="4" width="12" height="17" rx="2"/>'
        '<path d="M9 4V3h6v1"/><path d="M9 9h6M9 13h6M9 17h4"/>'
    ),
    "camera": (
        '<path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4'
        'a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="3.5"/>'
    ),
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 6"/><path d="M20 5v6h-6"/>',
    "trash": (
        '<path d="M4 7h16"/><path d="M9 7V5h6v2"/>'
        '<path d="M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/>'
    ),
    "sliders": (
        '<path d="M4 7h10M18 7h2"/><circle cx="16" cy="7" r="2"/>'
        '<path d="M4 17h2M10 17h10"/><circle cx="8" cy="17" r="2"/>'
    ),
    "lock": (
        '<rect x="5" y="10" width="14" height="10" rx="2"/>'
        '<path d="M8 10V7a4 4 0 0 1 8 0v3"/><path d="M12 14v2"/>'
    ),
    "power": '<path d="M12 3v9"/><path d="M6.5 6.5a8 8 0 1 0 11 0"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "warning": '<path d="M12 4l9 16H3l9-16z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    "chip": (
        '<rect x="7" y="7" width="10" height="10" rx="2"/>'
        '<path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4"/>'
    ),
    "cube": (
        '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/>'
        '<path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>'
    ),
    "image": (
        '<rect x="3" y="5" width="18" height="14" rx="2"/>'
        '<circle cx="8.5" cy="10" r="1.5"/><path d="M4 17l5-5 4 4 3-3 4 4"/>'
    ),
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "folder": (
        '<path d="M3 6a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v10'
        'a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6z"/>'
    ),
    "play": '<path d="M8 5l11 7-11 7V5z"/>',
    "database": (
        '<ellipse cx="12" cy="6" rx="7" ry="3"/>'
        '<path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/>'
        '<path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>'
    ),
}


def icon_names() -> tuple[str, ...]:
    return tuple(sorted(_BODIES))


def _document(name: str, color: str, size: int, stroke: float) -> str:
    try:
        body = _BODIES[name]
    except KeyError as exc:
        raise KeyError(f"未知图标: {name}") from exc
    return _TEMPLATE.format(size=size, color=color, stroke=stroke, body=body)


def render_pixmap(
    name: str,
    color: str = ACCENT,
    size: int = 24,
    stroke: float = 1.6,
    dpr: int = 2,
) -> QPixmap:
    """把线条图标渲染成 QPixmap；dpr=2 保证高分屏不糊。"""
    physical = max(1, int(size)) * max(1, int(dpr))
    renderer = QSvgRenderer(QByteArray(_document(name, color, physical, stroke).encode("utf-8")))
    pixmap = QPixmap(physical, physical)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(max(1, int(dpr)))
    return pixmap


def make_icon(
    name: str,
    color: str = ACCENT,
    size: int = 20,
    stroke: float = 1.6,
) -> QIcon:
    return QIcon(render_pixmap(name, color, size, stroke))
