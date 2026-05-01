# routers/rag.py - fachada de compatibilidade
"""Fachada de compatibilidade. Modulo movido para bbsia/app/routers/rag.py"""
import sys

from bbsia.app.routers import rag as _module
from bbsia.app.routers.rag import *  # noqa: F401, F403

sys.modules[__name__] = _module
