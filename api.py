"""
API REST do Chatbot RAG BBSIA.

Execucao:
  python api.py
ou
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from api_core import app
from routers import admin, biblioteca, rag, system

app.include_router(rag.router)
app.include_router(admin.router)
app.include_router(system.router)
app.include_router(biblioteca.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
