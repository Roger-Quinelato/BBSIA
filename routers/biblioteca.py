# routers/biblioteca.py - fachada de compatibilidade
"""Fachada de compatibilidade. Modulo movido para bbsia/app/routers/biblioteca.py"""
import sys

from bbsia.app.routers import biblioteca as _module
from bbsia.app.routers.biblioteca import *  # noqa: F401, F403

sys.modules[__name__] = _module
