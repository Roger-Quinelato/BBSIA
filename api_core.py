from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

import time

import os

from api_core import *

from routers import rag, admin, system, biblioteca

app.include_router(rag.router)

app.include_router(admin.router)

app.include_router(system.router)

app.include_router(biblioteca.router)