# routers/system.py - fachada de compatibilidade
"""Fachada de compatibilidade. Modulo movido para bbsia/app/routers/system.py"""
import sys

from bbsia.app.routers import system as _module
from bbsia.app.routers.system import *  # noqa: F401, F403

sys.modules[__name__] = _module
