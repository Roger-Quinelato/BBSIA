"""
Classificador inteligente de artigos científicos para o BBSIA.

Fase 1 do plano de metadados:
  - Schema Pydantic do metadado de documento
  - Extração heurística (título, autores, ano, resumo, seções, tipo)
  - Classificação por LLM (Ollama) com fallback
  - Catálogo persistente em data/biblioteca.json
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from config import get_env_str

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

OLLAMA_URL = get_env_str("OLLAMA_URL", "http://localhost:11434")
CLASSIFICADOR_MODEL = get_env_str("CLASSIFICADOR_MODEL", get_env_str("DEFAULT_MODEL", "qwen3.5:7b-instruct"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BIBLIOTECA_FILE = os.path.join(DATA_DIR, "biblioteca.json")

AREAS_VALIDAS = {"ia", "saude", "infraestrutura", "juridico", "tecnologia", "geral"}
TIPOS_VALIDOS = {"artigo_cientifico", "relatorio_tecnico", "manual", "apresentacao", "outro"}
METODOLOGIAS_VALIDAS = {
    "estudo de caso", "revisao-sistematica", "pesquisa-quantitativa",
    "relatorio", "manual", "outro",
}

# ---------------------------------------------------------------------------
# Schema Pydantic (Etapa 1)
# ---------------------------------------------------------------------------


class MetadadoDocumento(BaseModel):
    """Schema de metadado de documento conforme plano_metadados_biblioteca.md."""

    id: str = ""
    titulo: str = ""
    autores: list[str] = Field(default_factory=list)
    ano: int | None = None
    instituicao: str = ""
    tipo_documento: str = "outro"
    resumo: str = ""
    palavras_chave: list[str] = Field(default_factory=list)
    area_tematica: str = "geral"
    assuntos: list[str] = Field(default_factory=list)
    metodologia: str = "outro"
    secoes_detectadas: list[str] = Field(default_factory=list)
    paginas_total: int = 0
    documento_original: str = ""
    data_ingestao: str = ""
    qualidade_extracao: str = "media"


# ---------------------------------------------------------------------------
# Helpers para extração de spans (reutiliza lógica do extrator_pdf_v2)
# ---------------------------------------------------------------------------


def _span_lines(page: fitz.Page) -> list[dict[str, Any]]:
    """Extrai linhas com informações de fonte."""
    try:
        payload = page.get_text("dict")
    except Exception:
        return []

    lines: list[dict[str, Any]] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(str(s.get("text", "")).strip() for s in spans).strip()
            text = re.sub(r"\s{2,}", " ", text)
            if not text:
                continue
            sizes = [float(s.get("size", 0)) for s in spans if s.get("text", "").strip()]
            fonts = " ".join(str(s.get("font", "")) for s in spans)
            lines.append({
                "texto": text,
                "font_size": max(sizes) if sizes else 0.0,
                "is_bold": "bold" in fonts.lower(),
            })
    return lines


# ---------------------------------------------------------------------------
# Etapa 2 — Extração heurística
# ---------------------------------------------------------------------------


# Padrões de cabeçalhos institucionais que NÃO são o título do documento
_HEADER_INSTITUCIONAL = re.compile(
    r"(?i)^(universidade|faculdade|instituto|programa\s+de|centro\s+de|"
    r"departamento|escola|ministério|secretaria|governo|senado|câmara)",
)


def _extrair_titulo(pages_spans: list[list[dict]]) -> str:
    """Título = linha com maior fonte, ignorando headers institucionais.

    Quando todas as fontes têm o mesmo tamanho (comum em teses),
    busca o primeiro bloco bold que não é header institucional,
    localidade ou nome de autor.
    """
    if not pages_spans or not pages_spans[0]:
        return ""

    # Coleta as primeiras 2 páginas (teses têm capa + folha de rosto)
    pool: list[dict] = []
    for page in pages_spans[:2]:
        pool.extend(page)
    if not pool:
        return ""

    sizes = [l["font_size"] for l in pool if l["font_size"] > 0]
    all_same_size = len(set(sizes)) <= 1 if sizes else True

    # --- Estratégia 1: fontes variadas → pega a maior (ignorando ruído) ---
    if not all_same_size:
        ranked = sorted(pool, key=lambda l: l["font_size"], reverse=True)
        for line in ranked:
            candidato = line["texto"].strip()
            if not _is_titulo_candidato(candidato):
                continue
            return candidato

    # --- Estratégia 2: fontes uniformes → busca bold, pula headers ---
    # Concatena linhas bold consecutivas (título pode ter 2-3 linhas)
    titulo_parts: list[str] = []
    collecting = False

    for line in pool:
        text = line["texto"].strip()
        if not text:
            continue

        # Pula headers institucionais e ruído no topo
        if _HEADER_INSTITUCIONAL.match(text):
            continue
        if re.match(r"^(BRASÍLIA|BRASIL|SÃO PAULO|RIO DE JANEIRO|\d{4})$", text, re.IGNORECASE):
            continue
        if re.match(r"^[_\-=]{3,}$", text):
            continue

        # Linha não-bold isolada antes do título (geralmente é o nome do autor)
        if not line["is_bold"] and not collecting:
            continue

        if line["is_bold"] and _is_titulo_candidato(text):
            collecting = True
            titulo_parts.append(text)
        elif collecting:
            # Parou de ser bold ou encontrou ruído → encerra
            break

    if titulo_parts:
        return " ".join(titulo_parts)

    return ""


def _is_titulo_candidato(text: str) -> bool:
    """Verifica se uma linha pode ser título (não é ruído)."""
    if len(text) < 5 or text.isdigit():
        return False
    if _HEADER_INSTITUCIONAL.match(text):
        return False
    if re.match(r"^(BRASÍLIA|BRASIL|SÃO PAULO|RIO DE JANEIRO|\d{4})$", text, re.IGNORECASE):
        return False
    if re.match(r"^[_\-=]{3,}$", text):
        return False
    return True


def _extrair_autores(pages_spans: list[list[dict]]) -> list[str]:
    """Autores nas primeiras 2 páginas via padrões regex."""
    autores: list[str] = []
    text_pool = ""
    for page_spans in pages_spans[:2]:
        text_pool += "\n".join(l["texto"] for l in page_spans) + "\n"

    # Padrão: "Nome Sobrenome¹" ou "Nome Sobrenome, "
    # Captura nomes após o título (linhas com tamanho de fonte menor)
    patterns = [
        # Nome com superscript numérico
        re.compile(r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:de|da|do|dos|das)\s+)?(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s*)+)[¹²³⁴⁵⁶⁷⁸⁹*]"),
        # Nome em linha com vírgula e cargo/instituição
        re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:de|da|do|dos|das)\s+)?(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s*)+),\s*(?:Prof|Dr|Pesquisador|Analista|Coordenador|Mestre|Doutor)", re.MULTILINE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(text_pool):
            nome = match.group(1).strip()
            if 3 < len(nome) < 80 and nome not in autores:
                autores.append(nome)

    # Fallback: busca non-bold lines que parecem nomes de pessoa
    # Valida rigorosamente: 2-6 palavras, todas capitalizadas ou preposições
    if not autores and pages_spans:
        for page in pages_spans[:2]:
            for line in page:
                if line["is_bold"]:
                    continue
                candidato = line["texto"].strip()
                if not _parece_nome_pessoa(candidato):
                    continue
                if _HEADER_INSTITUCIONAL.match(candidato):
                    continue
                autores.append(candidato)
                break
            if autores:
                break

    return autores


def _parece_nome_pessoa(text: str) -> bool:
    """Verifica se o texto parece um nome de pessoa."""
    if not text or len(text) < 5 or len(text) > 80:
        return False
    # Rejeita se contém dígitos, emojis, ou pontuação estranha
    if any(c.isdigit() for c in text):
        return False
    if re.search(r"[^\w\s\-'.àáâãéêíóôõúüçÀÁÂÃÉÊÍÓÔÕÚÜÇ]", text):
        return False
    # Nomes não terminam com ponto
    if text.endswith("."):
        return False

    parts = text.split()
    if not (2 <= len(parts) <= 6):
        return False

    preps = {"de", "da", "do", "dos", "das", "e", "di", "del", "von", "van"}

    # Palavras comuns que não são nomes de pessoa
    _nao_nome = {
        "responsabilidade", "capacitação", "teses", "principal", "básica",
        "estrutura", "alocação", "papéis", "fase", "banco", "brasileiro",
        "soluções", "inteligência", "artificial", "framework", "avaliação",
        "prontidão", "maturidade", "níveis", "contato", "documento",
        "construção", "abertura", "oficina", "versão", "seção", "capítulo",
        "tabela", "figura", "quadro", "gráfico", "anexo", "apêndice",
        "especificação", "equipes", "técnicas", "técnicos", "api", "rest",
        "sistema", "projeto", "plano", "relatório", "análise", "resultado",
        "objetivo", "requisito", "requisitos", "implementação", "arquitetura",
        "gestão", "estratégica", "estratégia", "metodologia", "conclusão",
        "impacto", "ético", "ética", "autoavaliação", "diagnóstico",
        "talentos", "locais", "recursos", "humanos", "competências",
        "processos", "serviços", "dados", "digital", "tecnológica",
        "básicas", "básicos", "avançadas", "avançados", "gerais",
        "específicas", "específicos", "aplicações", "ferramentas",
    }

    non_prep_words = [w for w in parts if w.lower() not in preps]
    # Precisa de pelo menos 2 palavras que não sejam preposição
    if len(non_prep_words) < 2:
        return False

    for word in parts:
        if word.lower() in preps:
            continue
        if not word[0:1].isupper():
            return False
        if len(word) < 2:
            return False
        # Rejeita se qualquer palavra está na blocklist
        if word.lower() in _nao_nome:
            return False

    return True


def _extrair_ano(doc: fitz.Document, pages_spans: list[list[dict]]) -> int | None:
    """Ano via metadados do PDF ou regex nas primeiras páginas."""
    # Tenta metadados do PDF primeiro
    meta = doc.metadata or {}
    for key in ("creationDate", "modDate"):
        val = meta.get(key, "")
        match = re.search(r"(19|20)\d{2}", str(val))
        if match:
            year = int(match.group())
            if 1990 <= year <= 2030:
                return year

    # Regex nas primeiras 3 páginas
    for page_spans in pages_spans[:3]:
        text = " ".join(l["texto"] for l in page_spans)
        matches = re.findall(r"\b((?:19|20)\d{2})\b", text)
        for m in matches:
            year = int(m)
            if 1990 <= year <= 2030:
                return year

    return None


def _extrair_resumo(pages_spans: list[list[dict]]) -> str:
    """Busca bloco após 'Resumo', 'Abstract' ou 'Summary'."""
    keywords = {"resumo", "abstract", "summary"}

    for page_spans in pages_spans[:5]:
        found_keyword = False
        buffer: list[str] = []

        for line in page_spans:
            text_lower = line["texto"].strip().lower().rstrip(":")
            if text_lower in keywords:
                found_keyword = True
                continue

            if found_keyword:
                # Para quando encontrar outra seção
                if line["is_bold"] and len(line["texto"].split()) <= 5:
                    break
                buffer.append(line["texto"])
                # Limita o resumo
                if len(" ".join(buffer)) > 1500:
                    break

        if buffer:
            resumo = " ".join(buffer).strip()
            # Limita a 2000 chars
            return resumo[:2000]

    return ""


# Padrões de ruído que NÃO são seções reais do documento
_NOISE_SECTION = re.compile(
    r"(?i)^(\d{1,3}$"                   # Números de página soltos
    r"|[_\-=]{3,}$"                      # Linhas decorativas
    r"|\d{4}[\.\s]*$"                    # Anos soltos ("2022")
    r"|\(.*\)[\.\s]*$"                   # Referências entre parênteses
    r"|\d+\s+(ago|jan|fev|mar|abr|mai|jun|jul|set|out|nov|dez)\."  # Datas
    r"|.*disponível\s+em"               # Links de referência
    r")",
)

_SECOES_ESTRUTURAIS = {
    "introdução", "introducao", "resumo", "abstract", "summary",
    "metodologia", "método", "metodo", "resultados", "discussão", "discussao",
    "conclusão", "conclusao", "referências", "referencias",
    "referências bibliográficas", "referencias bibliograficas",
    "agradecimentos", "sumário", "sumario", "lista de figuras",
    "lista de tabelas", "apêndice", "apendice", "anexo", "anexos",
}


def _extrair_secoes(pages_spans: list[list[dict]]) -> list[str]:
    """Extrai seções estruturais de alto nível do documento."""
    import statistics

    all_sizes = [l["font_size"] for page in pages_spans for l in page if l["font_size"] > 0]
    if not all_sizes:
        return []
    median_size = statistics.median(all_sizes)

    secoes: list[str] = []
    seen: set[str] = set()

    for page_spans in pages_spans:
        for line in page_spans:
            text = line["texto"].strip()
            size = line["font_size"]
            is_bold = line["is_bold"]

            if not text or len(text) > 80 or len(text) < 3:
                continue
            # Filtra ruído
            if _NOISE_SECTION.match(text):
                continue
            # Ignora linhas que terminam com pontuação de frase
            if text.endswith((".", ",", ";")) and len(text.split()) > 6:
                continue

            # Padrão numerado: "1. Introdução" ou "1.2 Metodologia"
            numbered = bool(re.match(r"^(\d+(\.\d+)*|[A-Z]\.)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ].{2,}", text))
            bigger = median_size > 0 and size >= median_size + 1.5
            is_known = text.lower().strip(":") in _SECOES_ESTRUTURAIS

            if is_known or numbered or (bigger and is_bold and len(text.split()) <= 8):
                normalized = text.lower()
                if normalized not in seen:
                    seen.add(normalized)
                    secoes.append(text)

    return secoes



def _inferir_tipo_documento(secoes: list[str], titulo: str) -> str:
    """Heurística para tipo de documento baseado nas seções detectadas."""
    secoes_lower = {s.lower() for s in secoes}
    titulo_lower = titulo.lower()

    # Indicadores de artigo científico
    indicadores_artigo = {"metodologia", "método", "resultados", "conclusão",
                          "referências", "abstract", "resumo", "introdução",
                          "revisão da literatura", "referencial teórico"}
    match_artigo = len(indicadores_artigo & secoes_lower)

    if match_artigo >= 3:
        return "artigo_cientifico"

    # Indicadores de manual
    if any(k in titulo_lower for k in ("manual", "guia", "tutorial")):
        return "manual"

    # Indicadores de relatório
    if any(k in titulo_lower for k in ("relatório", "relatorio", "fase", "mvp", "framework")):
        return "relatorio_tecnico"

    # Indicadores de apresentação
    if any(k in titulo_lower for k in ("apresentação", "dinâmica", "oficina")):
        return "apresentacao"

    # Se tem pelo menos 1 seção de artigo, classifica como tal
    if match_artigo >= 1:
        return "artigo_cientifico"

    return "relatorio_tecnico"


def _gerar_id(filename: str, ano: int | None) -> str:
    """Gera um ID estável a partir do nome do arquivo."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Remove caracteres especiais
    clean = re.sub(r"[^\w\s-]", "", base).strip()
    clean = re.sub(r"\s+", "_", clean)
    if ano:
        if not clean.startswith(str(ano)):
            clean = f"{ano}_{clean}"
    return clean


def _pages_spans_from_payload(pages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Converte payload extraido em spans simplificados para heuristicas."""
    pages_spans: list[list[dict[str, Any]]] = []

    for page in pages:
        spans: list[dict[str, Any]] = []
        elements = page.get("elementos", [])
        if isinstance(elements, list):
            for element in elements:
                kind = str(element.get("tipo", "text"))
                text = str(element.get("texto", "")).strip()
                if not text:
                    continue
                is_heading = kind == "section"
                for line in text.splitlines():
                    candidate = line.strip()
                    if not candidate:
                        continue
                    spans.append(
                        {
                            "texto": candidate,
                            "font_size": 14.0 if is_heading else 12.0,
                            "is_bold": bool(is_heading),
                        }
                    )

        if not spans:
            raw_text = str(page.get("texto", "")).strip()
            for line in raw_text.splitlines():
                candidate = line.strip()
                if candidate:
                    spans.append({"texto": candidate, "font_size": 12.0, "is_bold": False})

        pages_spans.append(spans)

    return pages_spans


def _extrair_instituicao_pdf(pdf_path: str | None) -> str:
    """Tenta obter instituicao via metadados do PDF, quando caminho existe."""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        instituicao = str(meta.get("author", "") or meta.get("creator", "") or "").strip()
        doc.close()
        return instituicao
    except Exception:
        return ""


def _extrair_ano_de_payload(pages: list[dict[str, Any]], pdf_path: str | None = None) -> int | None:
    """Extrai ano do payload com fallback em metadados do PDF."""
    if pdf_path and os.path.exists(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            doc.close()
            for key in ("creationDate", "modDate"):
                val = meta.get(key, "")
                match = re.search(r"(19|20)\d{2}", str(val))
                if match:
                    year = int(match.group())
                    if 1990 <= year <= 2030:
                        return year
        except Exception:
            pass

    for page in pages[:3]:
        text = str(page.get("texto", ""))
        matches = re.findall(r"\b((?:19|20)\d{2})\b", text)
        for m in matches:
            year = int(m)
            if 1990 <= year <= 2030:
                return year

    return None


def _extrair_resumo_de_payload(pages: list[dict[str, Any]], pages_spans: list[list[dict[str, Any]]]) -> str:
    """Busca resumo no payload via secoes; fallback para heuristica por spans."""
    keywords = {"resumo", "abstract", "summary"}

    for page in pages[:5]:
        elements = page.get("elementos", [])
        if not isinstance(elements, list):
            continue

        found_keyword = False
        buffer: list[str] = []
        for element in elements:
            kind = str(element.get("tipo", "text"))
            text = str(element.get("texto", "")).strip()
            if not text:
                continue

            if kind == "section":
                label = text.lower().rstrip(":").strip()
                if label in keywords:
                    found_keyword = True
                    continue
                if found_keyword:
                    break

            if found_keyword and kind in {"text", "ocr_text", "table"}:
                buffer.append(text)
                if len(" ".join(buffer)) > 1500:
                    break

        if buffer:
            resumo = " ".join(buffer).strip()
            return resumo[:2000]

    return _extrair_resumo(pages_spans)


def _extrair_secoes_de_payload(pages: list[dict[str, Any]], pages_spans: list[list[dict[str, Any]]]) -> list[str]:
    """Extrai secoes detectadas diretamente do payload; fallback para spans."""
    secoes: list[str] = []
    seen: set[str] = set()

    for page in pages:
        elements = page.get("elementos", [])
        if not isinstance(elements, list):
            continue
        for element in elements:
            if str(element.get("tipo", "text")) != "section":
                continue
            text = str(element.get("texto", "")).strip()
            if not text:
                continue
            normalized = text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            secoes.append(text)

    if secoes:
        return secoes

    return _extrair_secoes(pages_spans)


def extrair_heuristicas(pdf_path: str) -> MetadadoDocumento:
    """
    Camada 1: extração heurística sem LLM.

    Retorna um MetadadoDocumento preenchido com o que for possível
    extrair do PDF via heurísticas de fonte, regex e estrutura.
    """
    doc = fitz.open(pdf_path)

    # Coleta spans de todas as páginas
    pages_spans: list[list[dict]] = []
    for page_num in range(len(doc)):
        pages_spans.append(_span_lines(doc[page_num]))

    titulo = _extrair_titulo(pages_spans)
    autores = _extrair_autores(pages_spans)
    ano = _extrair_ano(doc, pages_spans)
    resumo = _extrair_resumo(pages_spans)
    secoes = _extrair_secoes(pages_spans)
    tipo_doc = _inferir_tipo_documento(secoes, titulo)
    doc_id = _gerar_id(pdf_path, ano)

    # Instituição via metadados do PDF
    meta = doc.metadata or {}
    instituicao = str(meta.get("author", "") or meta.get("creator", "") or "").strip()

    total_paginas = len(doc)
    doc.close()

    return MetadadoDocumento(
        id=doc_id,
        titulo=titulo,
        autores=autores,
        ano=ano,
        instituicao=instituicao,
        tipo_documento=tipo_doc,
        resumo=resumo,
        palavras_chave=[],
        area_tematica="geral",
        assuntos=["geral"],
        metodologia="outro",
        secoes_detectadas=secoes,
        paginas_total=total_paginas,
        documento_original=os.path.relpath(pdf_path, BASE_DIR).replace("\\", "/"),
        data_ingestao=datetime.now(timezone.utc).isoformat(),
        qualidade_extracao="media",
    )


# ---------------------------------------------------------------------------
# Etapa 3 — Classificação por LLM (Ollama) + fallback
# ---------------------------------------------------------------------------

_LLM_PROMPT = """Você receberá o título e resumo de um documento.
Responda APENAS com um JSON válido no seguinte formato:
{{
  "area_tematica": "<ia|saude|infraestrutura|juridico|tecnologia|geral>",
  "assuntos": ["<lista de 3 a 6 tags em português, minúsculas, sem acentos>"],
  "palavras_chave": ["<3 a 8 palavras-chave do próprio texto>"],
  "metodologia": "<estudo de caso|revisao-sistematica|pesquisa-quantitativa|relatorio|manual|outro>",
  "tipo_documento": "<artigo_cientifico|relatorio_tecnico|manual|apresentacao|outro>"
}}

Título: {titulo}

Resumo: {resumo}
"""


def _query_ollama_json(titulo: str, resumo: str, timeout: int = 60) -> dict | None:
    """Envia título+resumo ao Ollama e tenta parsear o JSON de retorno."""
    import requests

    prompt = _LLM_PROMPT.format(titulo=titulo, resumo=resumo or "(sem resumo disponível)")

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CLASSIFICADOR_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                    "num_ctx": 8192,
                },
            },
            timeout=(10, timeout),
        )
        resp.raise_for_status()
        payload = resp.json()
        raw = payload.get("response", "")

        # Tenta extrair o JSON da resposta
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return None


def _validar_campo(valor: str, validos: set[str], default: str) -> str:
    """Valida um campo de classificação contra o conjunto de valores válidos."""
    if not valor or not isinstance(valor, str):
        return default
    normalizado = valor.strip().lower()
    return normalizado if normalizado in validos else default


def enriquecer_com_llm(metadado: MetadadoDocumento) -> MetadadoDocumento:
    """
    Camada 2: enriquecimento via LLM.

    Tenta classificar via Ollama. Se falhar ou retornar JSON inválido,
    mantém os valores heurísticos e marca qualidade_extracao='baixa'.
    """
    resultado = _query_ollama_json(metadado.titulo, metadado.resumo)

    if resultado is None or not isinstance(resultado, dict):
        metadado.qualidade_extracao = "baixa"
        return metadado

    # Aplica resultados validados
    area = _validar_campo(
        resultado.get("area_tematica", ""), AREAS_VALIDAS, metadado.area_tematica
    )
    tipo = _validar_campo(
        resultado.get("tipo_documento", ""), TIPOS_VALIDOS, metadado.tipo_documento
    )
    metodologia = _validar_campo(
        resultado.get("metodologia", ""), METODOLOGIAS_VALIDAS, metadado.metodologia
    )

    assuntos = resultado.get("assuntos", [])
    if isinstance(assuntos, list) and assuntos:
        assuntos = [str(a).strip().lower() for a in assuntos if str(a).strip()]
        if assuntos:
            metadado.assuntos = assuntos[:6]

    palavras_chave = resultado.get("palavras_chave", [])
    if isinstance(palavras_chave, list) and palavras_chave:
        palavras_chave = [str(p).strip() for p in palavras_chave if str(p).strip()]
        if palavras_chave:
            metadado.palavras_chave = palavras_chave[:8]

    metadado.area_tematica = area
    metadado.tipo_documento = tipo
    metadado.metodologia = metodologia
    metadado.qualidade_extracao = "alta"

    return metadado


# ---------------------------------------------------------------------------
# Função principal: classificar documento
# ---------------------------------------------------------------------------


def classificar(pdf_path: str, usar_llm: bool = True) -> MetadadoDocumento:
    """
    Pipeline completo de classificação de um documento PDF.

    1. Extrai metadados via heurísticas
    2. Enriquece via LLM (se disponível e habilitado)

    Retorna MetadadoDocumento preenchido.
    """
    metadado = extrair_heuristicas(pdf_path)

    if usar_llm and (metadado.titulo or metadado.resumo):
        metadado = enriquecer_com_llm(metadado)

    return metadado


def classificar_de_payload(
    documento: dict[str, Any],
    pdf_path: str | None = None,
    usar_llm: bool = True,
) -> MetadadoDocumento:
    """
    Classifica documento a partir do payload estruturado do extrator.

    Evita reabrir/reprocessar PDF quando texto, elementos e metadados de pagina
    ja estao disponiveis em memoria.
    """
    pages = documento.get("paginas", [])
    if not isinstance(pages, list) or not pages:
        if pdf_path:
            return classificar(pdf_path, usar_llm=usar_llm)
        raise ValueError("Payload de documento invalido: campo 'paginas' ausente ou vazio.")

    pages_spans = _pages_spans_from_payload(pages)
    titulo = _extrair_titulo(pages_spans)
    autores = _extrair_autores(pages_spans)
    ano = _extrair_ano_de_payload(pages, pdf_path=pdf_path)
    resumo = _extrair_resumo_de_payload(pages, pages_spans)
    secoes = _extrair_secoes_de_payload(pages, pages_spans)
    tipo_doc = _inferir_tipo_documento(secoes, titulo)

    documento_original = str(documento.get("documento", "")).strip().replace("\\", "/")
    if not documento_original and pdf_path:
        documento_original = os.path.relpath(pdf_path, BASE_DIR).replace("\\", "/")

    doc_id = _gerar_id(pdf_path or documento_original or "documento", ano)
    instituicao = _extrair_instituicao_pdf(pdf_path)

    metadado = MetadadoDocumento(
        id=doc_id,
        titulo=titulo,
        autores=autores,
        ano=ano,
        instituicao=instituicao,
        tipo_documento=tipo_doc,
        resumo=resumo,
        palavras_chave=[],
        area_tematica="geral",
        assuntos=["geral"],
        metodologia="outro",
        secoes_detectadas=secoes,
        paginas_total=len(pages),
        documento_original=documento_original,
        data_ingestao=datetime.now(timezone.utc).isoformat(),
        qualidade_extracao="media",
    )

    if usar_llm and (metadado.titulo or metadado.resumo):
        metadado = enriquecer_com_llm(metadado)

    return metadado


# ---------------------------------------------------------------------------
# Catálogo persistente: data/biblioteca.json
# ---------------------------------------------------------------------------


def carregar_biblioteca() -> dict:
    """Carrega o catálogo de documentos, ou retorna estrutura vazia."""
    if not os.path.exists(BIBLIOTECA_FILE):
        return {"versao": 1, "atualizado_em": "", "documentos": []}

    try:
        with open(BIBLIOTECA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("documentos"), list):
            return data
    except Exception:
        pass

    return {"versao": 1, "atualizado_em": "", "documentos": []}


def salvar_biblioteca(biblioteca: dict) -> None:
    """Salva o catálogo no disco."""
    os.makedirs(DATA_DIR, exist_ok=True)
    biblioteca["atualizado_em"] = datetime.now(timezone.utc).isoformat()
    with open(BIBLIOTECA_FILE, "w", encoding="utf-8") as f:
        json.dump(biblioteca, f, ensure_ascii=False, indent=2)


def atualizar_biblioteca(metadado: MetadadoDocumento) -> None:
    """Atualiza ou insere um documento no catálogo biblioteca.json."""
    biblioteca = carregar_biblioteca()
    docs = biblioteca.get("documentos", [])

    # Remove entrada anterior do mesmo documento (por id ou documento_original)
    docs = [
        d for d in docs
        if d.get("id") != metadado.id
        and d.get("documento_original") != metadado.documento_original
    ]

    docs.append(metadado.model_dump())
    biblioteca["documentos"] = docs
    salvar_biblioteca(biblioteca)


def classificar_e_registrar(pdf_path: str, usar_llm: bool = True) -> MetadadoDocumento:
    """
    Classifica um PDF e atualiza o catálogo biblioteca.json.

    Conveniência para uso no pipeline de extração.
    """
    metadado = classificar(pdf_path, usar_llm=usar_llm)
    atualizar_biblioteca(metadado)
    return metadado


# ---------------------------------------------------------------------------
# CLI standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python classificador_artigo.py <caminho_pdf> [--no-llm]")
        sys.exit(1)

    caminho = sys.argv[1]
    usar_llm = "--no-llm" not in sys.argv

    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        sys.exit(1)

    print(f"Classificando: {caminho}")
    print(f"LLM habilitado: {usar_llm}\n")

    resultado = classificar_e_registrar(caminho, usar_llm=usar_llm)

    output = json.dumps(resultado.model_dump(), ensure_ascii=False, indent=2)
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    print(f"\nBiblioteca atualizada em: {BIBLIOTECA_FILE}")
