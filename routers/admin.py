# routers/admin.py - fachada de compatibilidade
"""Fachada de compatibilidade. Modulo movido para bbsia/app/routers/admin.py"""
import sys

from bbsia.app.routers import admin as _module
from bbsia.app.routers.admin import *  # noqa: F401, F403

sys.modules[__name__] = _module
