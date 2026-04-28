from fastapi import APIRouter, Request, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from typing import Any
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from api_core import *

router = APIRouter(prefix='', tags=["Biblioteca"])

