import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":                       # Qt's offscreen plugin finds no fonts on Windows otherwise
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
