from typing import Any

from fastapi import APIRouter, HTTPException

from bbsia.app.core import carregar_biblioteca

router = APIRouter(prefix="", tags=["Biblioteca"])


@router.get("/biblioteca")
def get_biblioteca(
    area: str | None = None,
    tipo: str | None = None,
    ano_min: int | None = None,
    ano_max: int | None = None,
) -> dict[str, Any]:
    biblioteca = carregar_biblioteca()
    docs = biblioteca.get("documentos", [])

    if area:
        area_lower = area.strip().lower()
        docs = [d for d in docs if str(d.get("area_tematica", "")).lower() == area_lower]
    if tipo:
        tipo_lower = tipo.strip().lower()
        docs = [d for d in docs if str(d.get("tipo_documento", "")).lower() == tipo_lower]
    if ano_min is not None:
        docs = [d for d in docs if isinstance(d.get("ano"), int) and d["ano"] >= ano_min]
    if ano_max is not None:
        docs = [d for d in docs if isinstance(d.get("ano"), int) and d["ano"] <= ano_max]

    resumidos = []
    for d in docs:
        resumidos.append(
            {
                "id": d.get("id", ""),
                "titulo": d.get("titulo", ""),
                "autores": d.get("autores", []),
                "ano": d.get("ano"),
                "area_tematica": d.get("area_tematica", "geral"),
                "assuntos": d.get("assuntos", []),
                "tipo_documento": d.get("tipo_documento", "outro"),
                "paginas_total": d.get("paginas_total", 0),
            }
        )

    return {"total": len(resumidos), "documentos": resumidos}


@router.get("/biblioteca/{doc_id}")
def get_biblioteca_doc(doc_id: str) -> dict[str, Any]:
    biblioteca = carregar_biblioteca()
    for doc in biblioteca.get("documentos", []):
        if doc.get("id") == doc_id:
            return doc
    raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' nao encontrado na biblioteca.")


@router.get("/filtros")
def get_filtros() -> dict[str, Any]:
    biblioteca = carregar_biblioteca()
    docs = biblioteca.get("documentos", [])

    areas: set[str] = set()
    tipos: set[str] = set()
    anos: set[int] = set()
    assuntos: set[str] = set()

    for d in docs:
        area_val = d.get("area_tematica", "")
        if area_val:
            areas.add(area_val)
        tipo_val = d.get("tipo_documento", "")
        if tipo_val:
            tipos.add(tipo_val)
        ano = d.get("ano")
        if isinstance(ano, int):
            anos.add(ano)
        for assunto in d.get("assuntos", []):
            if isinstance(assunto, str) and assunto.strip():
                assuntos.add(assunto.strip())

    return {
        "areas": sorted(areas),
        "tipos": sorted(tipos),
        "anos": sorted(anos),
        "assuntos": sorted(assuntos),
    }
