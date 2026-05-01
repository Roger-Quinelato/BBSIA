"""
API REST do Chatbot RAG BBSIA.

Execucao:
  python -m bbsia.app.bootstrap.main
ou
  uvicorn bbsia.app.bootstrap.main:app --host 0.0.0.0 --port 8000
"""

from bbsia.app.runtime.app import app
from bbsia.app.routers import admin, biblioteca, rag, system

app.include_router(rag.router)
app.include_router(admin.router)
app.include_router(system.router)
app.include_router(biblioteca.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bbsia.app.bootstrap.main:app", host="0.0.0.0", port=8000, reload=False)
