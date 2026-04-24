"""
Testes para o classificador_artigo.py (Fase 1 do plano de metadados).

Cobre:
  - Extração de título, autores e ano da 1ª página
  - Fallback quando LLM retorna JSON inválido
  - Detecção de tipo de documento (artigo vs relatório)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import classificador_artigo as clf


def _make_span(texto: str, font_size: float = 12.0, is_bold: bool = False) -> dict:
    """Cria um span simulado para testes."""
    return {"texto": texto, "font_size": font_size, "is_bold": is_bold}


# -----------------------------------------------------------------------
# Extração de título
# -----------------------------------------------------------------------


class TestExtracaoTitulo:
    """Testes para extração de título."""

    def test_titulo_maior_fonte(self):
        """Título = linha com maior fonte, ignorando headers institucionais."""
        pages_spans = [[
            _make_span("Universidade de Brasília", font_size=14.0, is_bold=True),
            _make_span("Adoção de IA em Serviços Públicos", font_size=18.0, is_bold=True),
            _make_span("Marina de Alencar Coutinho", font_size=12.0),
            _make_span("Resumo", font_size=14.0, is_bold=True),
        ]]
        titulo = clf._extrair_titulo(pages_spans)
        assert titulo == "Adoção de IA em Serviços Públicos"

    def test_titulo_bold_quando_fontes_uniformes(self):
        """Fontes uniformes → usa primeira linha bold não-institucional."""
        pages_spans = [[
            _make_span("Universidade de Brasília", font_size=12.0, is_bold=True),
            _make_span("Framework de Ética em IA", font_size=12.0, is_bold=True),
            _make_span("Autor Exemplo", font_size=12.0),
        ]]
        titulo = clf._extrair_titulo(pages_spans)
        assert "Framework" in titulo

    def test_titulo_vazio_sem_spans(self):
        """Retorna string vazia se não houver spans."""
        assert clf._extrair_titulo([]) == ""
        assert clf._extrair_titulo([[]]) == ""


# -----------------------------------------------------------------------
# Extração de autores
# -----------------------------------------------------------------------


class TestExtracaoAutores:
    """Testes para extração de autores."""

    def test_autores_com_superscript(self):
        """Detecta autores com marcador de nota de rodapé."""
        pages_spans = [[
            _make_span("Título do Artigo", font_size=18.0, is_bold=True),
            _make_span("Marina de Alencar Araripe Coutinho¹", font_size=12.0),
        ]]
        autores = clf._extrair_autores(pages_spans)
        assert len(autores) >= 1
        assert any("Coutinho" in a for a in autores)

    def test_autores_fallback_nome_pessoa(self):
        """Fallback detecta nomes próprios em linhas não-bold."""
        pages_spans = [[
            _make_span("Título do Artigo Científico", font_size=18.0, is_bold=True),
            _make_span("João Carlos Silva", font_size=12.0),
        ]]
        autores = clf._extrair_autores(pages_spans)
        assert len(autores) >= 1
        assert "João Carlos Silva" in autores


# -----------------------------------------------------------------------
# Extração de ano
# -----------------------------------------------------------------------


class TestExtracaoAno:
    """Testes para extração de ano."""

    def test_ano_via_metadados_pdf(self):
        """Extrai ano dos metadados do PDF."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"creationDate": "D:20220315"}
        pages_spans = [[_make_span("Texto qualquer", font_size=12.0)]]

        ano = clf._extrair_ano(mock_doc, pages_spans)
        assert ano == 2022

    def test_ano_via_regex_pagina(self):
        """Extrai ano via regex do texto da página."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        pages_spans = [[
            _make_span("Publicado em 2023", font_size=12.0),
        ]]

        ano = clf._extrair_ano(mock_doc, pages_spans)
        assert ano == 2023


# -----------------------------------------------------------------------
# Detecção de tipo de documento
# -----------------------------------------------------------------------


class TestDeteccaoTipo:
    """Testes para detecção de tipo de documento."""

    def test_artigo_cientifico_por_secoes(self):
        """Documento com seções acadêmicas → artigo_cientifico."""
        secoes = ["Introdução", "Metodologia", "Resultados", "Conclusão", "Referências"]
        tipo = clf._inferir_tipo_documento(secoes, "Análise de dados em saúde")
        assert tipo == "artigo_cientifico"

    def test_relatorio_tecnico_por_titulo(self):
        """Documento com 'relatório' no título → relatorio_tecnico."""
        secoes = ["Visão Geral", "Escopo"]
        tipo = clf._inferir_tipo_documento(secoes, "Relatório de Infraestrutura BBSIA")
        assert tipo == "relatorio_tecnico"

    def test_manual_por_titulo(self):
        """Documento com 'manual' no título → manual."""
        secoes = []
        tipo = clf._inferir_tipo_documento(secoes, "Manual de Uso da Plataforma")
        assert tipo == "manual"

    def test_fallback_relatorio(self):
        """Sem indicadores claros → relatorio_tecnico."""
        secoes = ["Escopo"]
        tipo = clf._inferir_tipo_documento(secoes, "Documento Genérico")
        assert tipo == "relatorio_tecnico"


# -----------------------------------------------------------------------
# Fallback quando LLM falha
# -----------------------------------------------------------------------


class TestFallbackLLM:
    """Testes para fallback quando LLM falha ou retorna JSON inválido."""

    def test_fallback_llm_retorna_none(self):
        """Se LLM retorna None → qualidade='baixa', mantém heurísticas."""
        metadado = clf.MetadadoDocumento(
            titulo="Teste",
            area_tematica="geral",
            assuntos=["geral"],
            qualidade_extracao="media",
        )
        with patch.object(clf, "_query_ollama_json", return_value=None):
            resultado = clf.enriquecer_com_llm(metadado)

        assert resultado.qualidade_extracao == "baixa"
        assert resultado.area_tematica == "geral"

    def test_fallback_llm_json_invalido(self):
        """Se LLM retorna campo inválido → usa default via validação."""
        metadado = clf.MetadadoDocumento(
            titulo="Teste IA",
            tipo_documento="outro",
            area_tematica="geral",
            qualidade_extracao="media",
        )
        fake_result = {
            "area_tematica": "ia",
            "tipo_documento": "formato_inexistente",  # valor inválido
            "assuntos": ["teste"],
            "palavras_chave": ["ia"],
            "metodologia": "outro",
        }
        with patch.object(clf, "_query_ollama_json", return_value=fake_result):
            resultado = clf.enriquecer_com_llm(metadado)

        assert resultado.qualidade_extracao == "alta"
        assert resultado.area_tematica == "ia"
        # tipo_documento inválido → mantém o valor original ("outro")
        assert resultado.tipo_documento == "outro"

    def test_enriquecimento_bem_sucedido(self):
        """Se LLM retorna JSON válido → qualidade='alta', campos atualizados."""
        metadado = clf.MetadadoDocumento(
            titulo="Adoção de IA",
            area_tematica="geral",
            assuntos=["geral"],
            qualidade_extracao="media",
        )
        fake_result = {
            "area_tematica": "ia",
            "tipo_documento": "artigo_cientifico",
            "assuntos": ["etica", "governanca", "lgpd"],
            "palavras_chave": ["inteligência artificial", "setor público"],
            "metodologia": "estudo de caso",
        }
        with patch.object(clf, "_query_ollama_json", return_value=fake_result):
            resultado = clf.enriquecer_com_llm(metadado)

        assert resultado.qualidade_extracao == "alta"
        assert resultado.area_tematica == "ia"
        assert resultado.tipo_documento == "artigo_cientifico"
        assert resultado.assuntos == ["etica", "governanca", "lgpd"]
        assert resultado.metodologia == "estudo de caso"
