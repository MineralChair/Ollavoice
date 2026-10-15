"""Fake PySide6 package so the GUI modules can be import-checked and
construction-smoke-tested in an environment with no usable Qt (no display, or
missing system libraries).

Every class returns permissive stubs: unknown attribute access yields another
stub, and calls yield stubs too - except for the handful of Qt getters whose
*return type* our code depends on (``currentText`` must be a ``str``,
``value()`` an ``int``, ``isChecked()`` a ``bool``...). Those return realistic
values so arithmetic, JSON serialisation and string handling exercise the real
code paths.

This cannot validate Qt semantics, but it does catch the bugs that actually
ship: typos in method/signal names, missing imports, wrong arity on our own
methods, and anything that raises during construction or inside an event handler.
"""

from __future__ import annotations

import sys
import types

#: Qt getters that must hand back a str for our code to behave.
_TEXT_METHODS = {
    "currentText", "text", "toPlainText", "objectName", "description", "whatsThis",
    "toolTip", "statusTip", "windowTitle", "plainText", "displayText", "styleSheet",
    "textOf", "label", "name",
}
#: Qt getters that must hand back an int.
_INT_METHODS = {
    "value", "count", "currentIndex", "width", "height", "size", "x", "y",
    "sampleRate", "channelCount", "maximum", "minimum", "error", "state", "rowCount",
    "columnCount", "length", "bytesFree", "bufferSize", "processedMsecs", "elapsed",
    "sampleFormat", "byteOrder", "maximumChannelCount", "minimumChannelCount",
}
#: Qt getters that must hand back a bool.
_BOOL_METHODS = {
    "isChecked", "isEnabled", "isVisible", "isHidden", "isNull", "isValid",
    "isFormatSupported", "isSystemTrayAvailable", "isRunning", "isFinished",
    "isDown", "isCheckable", "isCheckedAction", "isEmpty", "isEditable",
    "isSignalConnected", "isWidgetType", "isWindow", "isModal", "isPlaying",
}
#: Qt getters that must hand back a float.
_FLOAT_METHODS = {"volume", "opacity", "duration", "position"}


class _Stub:
    """Accepts anything; returns a plausible value for known Qt getters."""

    calls: list[str] = []

    def __init__(self, *args, _name: str = "", **kwargs) -> None:
        self._name = _name
        self.args = args
        self.kwargs = kwargs

    # -- attribute access -------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_child(name)

    def __call__(self, *args, **kwargs):
        base = self._name.rsplit(".", 1)[-1]
        if base in _TEXT_METHODS:
            return "stub-text"
        if base in _INT_METHODS:
            return 0
        if base in _BOOL_METHODS:
            return True
        if base in _FLOAT_METHODS:
            return 1.0
        return _Stub(*args, **kwargs)

    # -- protocol support so real code can do maths/formatting on results --
    def __bool__(self):
        return True

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __index__(self):
        return 0

    def __round__(self, ndigits=None):
        return 0.0 if ndigits else 0

    def __abs__(self):
        return _Stub()

    def __neg__(self):
        return _Stub()

    def __str__(self):
        return "stub"

    def __repr__(self):
        return f"<qt-stub {self._name}>"

    def __getitem__(self, item):
        return _Stub()

    def __contains__(self, item):
        return False

    # -- comparisons (deliberately inert) ---------------------------------
    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return True

    def __hash__(self):
        return id(self)

    # -- arithmetic (flag combining, slider maths, ...) --------------------
    def __or__(self, other):
        return _Stub()

    __ror__ = __or__
    __and__ = __or__
    __rand__ = __or__
    __xor__ = __or__
    __rxor__ = __or__
    __rshift__ = __or__
    __lshift__ = __or__

    def __add__(self, other):
        return _Stub()

    __radd__ = __add__

    def __sub__(self, other):
        return _Stub()

    __rsub__ = __sub__

    def __mul__(self, other):
        return _Stub()

    __rmul__ = __mul__

    def __truediv__(self, other):
        return _Stub()

    __rtruediv__ = __truediv__

    def __floordiv__(self, other):
        return _Stub()

    __rfloordiv__ = __floordiv__

    def __mod__(self, other):
        return _Stub()

    __rmod__ = __mod__

    def __pow__(self, other):
        return _Stub()

    __rpow__ = __pow__


def _make_child(name: str) -> _Stub:
    stub = _Stub(_name=name)
    _Stub.calls.append(name)
    return stub


class _StubClass(type):
    """Metaclass so ``Qt.Key.Key_Space``-style enum lookups also yield stubs."""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_child(f"{cls.__name__}.{name}")


class StubMeta(_Stub, metaclass=_StubClass):
    """Base for every fake Qt class: usable as a class *and* as an instance."""


def _stub_class(name: str) -> type:
    return _StubClass(name, (StubMeta,), {})


MODULE_CLASSES = {
    "QtCore": [
        "QObject", "QThread", "Signal", "Slot", "Qt", "QTimer", "QCoreApplication",
        "QAbstractNativeEventFilter", "QBuffer", "QByteArray", "QIODevice", "QEvent",
        "QMetaObject", "Q_ARG", "QPoint", "QSize", "QRect", "QUrl", "QEasingCurve",
        "Property", "QEventLoop", "QStandardPaths",
    ],
    "QtGui": [
        "QIcon", "QPixmap", "QKeySequence", "QAction", "QGuiApplication", "QFont",
        "QColor", "QPainter", "QShortcut", "QCloseEvent", "QTextCursor", "QKeyEvent",
        "QDesktopServices", "QBrush", "QPen",
    ],
    "QtWidgets": [
        "QApplication", "QMainWindow", "QWidget", "QDialog", "QLabel", "QPushButton",
        "QComboBox", "QLineEdit", "QCheckBox", "QSlider", "QSpinBox", "QDoubleSpinBox",
        "QProgressBar", "QTextBrowser", "QPlainTextEdit", "QTextEdit", "QVBoxLayout",
        "QHBoxLayout", "QFormLayout", "QGridLayout", "QGroupBox", "QTabWidget",
        "QToolBar", "QMenuBar", "QMenu", "QStatusBar", "QSystemTrayIcon", "QMessageBox",
        "QFileDialog", "QInputDialog", "QSizePolicy", "QDialogButtonBox", "QScrollArea",
        "QFrame", "QSplitter", "QListView", "QStackedWidget", "QButtonGroup",
    ],
    "QtMultimedia": [
        "QAudioSource", "QAudioSink", "QAudioFormat", "QMediaDevices", "QAudioDevice",
        "QAudio", "QMediaRecorder", "QAudioOutput", "QSoundEffect",
    ],
}


def install() -> None:
    """Insert the fake PySide6 into ``sys.modules`` (idempotent)."""
    existing = sys.modules.get("PySide6")
    if existing is not None and getattr(existing, "_ollavoice_stub", False):
        return

    def signal_stub(*args, **kwargs):
        """``Signal(...)`` must yield something with connect/emit/disconnect."""
        return _Stub(_name="Signal")

    package = types.ModuleType("PySide6")
    package._ollavoice_stub = True          # type: ignore[attr-defined]
    package.__version__ = "6.0.0-ollavoice-stub"
    package.__path__ = []                   # type: ignore[var-annotated]
    sys.modules["PySide6"] = package

    for module_name, classes in MODULE_CLASSES.items():
        module = types.ModuleType(f"PySide6.{module_name}")
        module.__path__ = []                # type: ignore[var-annotated]
        for class_name in classes:
            if class_name == "Signal":
                setattr(module, class_name, signal_stub)
                continue
            setattr(module, class_name, _stub_class(class_name))
        setattr(package, module_name, module)
        sys.modules[f"PySide6.{module_name}"] = module
