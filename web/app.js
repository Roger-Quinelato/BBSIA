/* ───────────────────────────────────────────────────────────────
   ChatBBSIA — Frontend (Fase 4: Biblioteca + Filtros Dinâmicos)
   ─────────────────────────────────────────────────────────────── */

const state = {
  loading: false,
  defaultModel: "llama3.1:8b",
  selectedArea: "",
  messages: [],
  biblioteca: [],        // cache de /biblioteca
  filtros: {},           // cache de /filtros
  selectedDocId: null,   // doc selecionado na biblioteca para filtrar
};

const elements = {
  messages:      document.getElementById("messages"),
  welcome:       document.getElementById("welcome"),
  welcomeTitle:  document.getElementById("welcomeTitle"),
  chatForm:      document.getElementById("chatForm"),
  question:      document.getElementById("question"),
  sendBtn:       document.getElementById("sendBtn"),
  clearBtn:      document.getElementById("clearBtn"),
  newChatBtn:    document.getElementById("newChatBtn"),
  uploadPdfBtn:  document.getElementById("uploadPdfBtn"),
  pdfUpload:     document.getElementById("pdfUpload"),
  areaSelect:    document.getElementById("areaSelect"),
  agentList:     document.getElementById("agentList"),
  chatTitle:     document.getElementById("chatTitle"),
  chatDate:      document.getElementById("chatDate"),
  chatCount:     document.getElementById("chatCount"),
  toggleSidebar: document.getElementById("toggleSidebar"),
  sidebar:       document.getElementById("sidebar"),
  bibCount:      document.getElementById("bibCount"),
  bibList:       document.getElementById("bibList"),
  areasPanel:    document.getElementById("areasPanel"),
  bibliotecaPanel: document.getElementById("bibliotecaPanel"),
  docModal:      document.getElementById("docModal"),
  modalBody:     document.getElementById("modalBody"),
  modalClose:    document.getElementById("modalClose"),
};

/* ─── Helpers ────────────────────────────────────────────────── */

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const AREA_LABELS = {
  tecnologia: "Tecnologia", ia: "IA", saude: "Saúde",
  infraestrutura: "Infraestrutura", juridico: "Jurídico", geral: "Geral",
};

function labelForArea(area) {
  return AREA_LABELS[area] || area || "Todas as áreas";
}

const TIPO_ICONS = {
  artigo_cientifico: "🔬", relatorio_tecnico: "📋",
  manual: "📖", apresentacao: "📊", outro: "📄",
};

function iconForTipo(tipo) {
  return TIPO_ICONS[tipo] || "📄";
}

function buildHeaders(isJson = true) {
  const headers = {};
  if (isJson) headers["Content-Type"] = "application/json";
  return headers;
}

function formatSourceLabel(item) {
  const autores = item.doc_autores || [];
  const ano = item.doc_ano;
  const titulo = item.doc_titulo || "";
  const documento = item.documento || "desconhecido";

  if (autores.length > 0 && autores[0]) {
    const partes = autores[0].trim().split(" ");
    const sobrenome = partes[partes.length - 1];
    let label = sobrenome;
    if (ano) label += `, ${ano}`;
    if (titulo) label += ` — "${titulo}"`;
    return label;
  }

  const baseName = documento.replace(/\.[^.]+$/, "");
  return ano ? `${baseName} (${ano})` : baseName;
}

/* ─── UI Updates ─────────────────────────────────────────────── */

function updateWelcome() {
  elements.welcome.hidden = state.messages.length > 0;
}

function showApiOffline() {
  if (elements.welcome) {
    elements.welcome.hidden = false;
    elements.welcomeTitle.textContent = "API indisponível";
    const desc = elements.welcome.querySelector("p");
    if (desc) {
      desc.textContent =
        "Não foi possível conectar ao servidor. Verifique se a API está rodando em localhost:8000.";
    }
  }
}

function updateChatSummary(text) {
  const title = text ? text.slice(0, 44) : "Nova conversa RAG";
  elements.chatTitle.textContent = title.length === 44 ? `${title}...` : title;
  elements.chatDate.textContent = new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  }).format(new Date());
  elements.chatCount.textContent = state.messages.length > 0 ? "1" : "0";
}

function setLoading(loading) {
  state.loading = loading;
  elements.sendBtn.disabled = loading;
  elements.sendBtn.textContent = loading ? "…" : "↗";
}

/* ─── 4.3 Rich Sources ───────────────────────────────────────── */

function renderRichSources(resultados) {
  if (!resultados || resultados.length === 0) return "";

  const cards = resultados.map((r, i) => {
    const label = escapeHtml(formatSourceLabel(r));
    const pag = r.pagina || "?";
    const area = escapeHtml(r.area || "geral");
    const tipo = escapeHtml((r.content_type || "text").replace("_", " "));

    // data-doc-name para abrir modal ao clicar
    const docName = escapeHtml(r.documento || "");

    return `<button class="source-card" type="button" data-doc-name="${docName}" title="Ver detalhes">
      <div class="source-label">[${i + 1}] ${label}</div>
      <div class="source-meta">
        <span>p. ${pag}</span>
        <span>${tipo}</span>
        <span>área: ${area}</span>
      </div>
    </button>`;
  }).join("");

  return `<div class="sources">
    <div class="sources-header">📎 Fontes</div>
    ${cards}
  </div>`;
}

/* ─── Messages ───────────────────────────────────────────────── */

function addMessage(role, content, resultados = [], fontes = []) {
  state.messages.push({ role, content, resultados });

  const wrapper = document.createElement("article");
  wrapper.className = `msg ${role}`;

  const roleText = role === "user" ? "Você" : "BBSIA";
  const htmlContent = escapeHtml(content).replaceAll("\n", "<br>");

  wrapper.innerHTML = `
    <div class="role">${roleText}</div>
    <div class="content">${htmlContent}</div>
  `;

  if (role === "assistant" && resultados.length > 0) {
    const sourcesHtml = renderRichSources(resultados);
    if (sourcesHtml) {
      const div = document.createElement("div");
      div.innerHTML = sourcesHtml;
      wrapper.appendChild(div.firstElementChild);
    }
  }

  elements.messages.appendChild(wrapper);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  updateWelcome();

  if (role === "user") {
    updateChatSummary(content);
  }
}

/* ─── Area Selection ─────────────────────────────────────────── */

function setSelectedArea(area) {
  state.selectedArea = area || "";
  elements.areaSelect.value = state.selectedArea;

  elements.agentList.querySelectorAll(".agent-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.area === state.selectedArea);
  });
}

/* ─── 4.2 Dynamic Filters ────────────────────────────────────── */

async function loadFilters() {
  try {
    const resp = await fetch("/filtros", { headers: buildHeaders(false) });
    if (!resp.ok) return;
    state.filtros = await resp.json();

    // Populate sidebar area buttons
    const areas = state.filtros.areas || [];
    const list = elements.agentList;
    list.innerHTML = `<button class="agent-item active" type="button" data-area="">Todas as áreas</button>`;
    areas.forEach((area) => {
      const btn = document.createElement("button");
      btn.className = "agent-item";
      btn.type = "button";
      btn.dataset.area = area;
      btn.textContent = labelForArea(area);
      list.appendChild(btn);
    });

    // Populate composer area select
    const sel = elements.areaSelect;
    sel.innerHTML = `<option value="">Todas as áreas</option>`;
    areas.forEach((area) => {
      const opt = document.createElement("option");
      opt.value = area;
      opt.textContent = labelForArea(area);
      sel.appendChild(opt);
    });
  } catch {
    // Fallback silencioso — os hardcoded do HTML ficam
  }
}

/* ─── 4.1 Biblioteca Panel ───────────────────────────────────── */

async function loadBiblioteca() {
  try {
    const resp = await fetch("/biblioteca", { headers: buildHeaders(false) });
    if (!resp.ok) return;
    const data = await resp.json();
    state.biblioteca = data.documentos || [];
    elements.bibCount.textContent = state.biblioteca.length;
    renderBiblioteca();
  } catch {
    // silencioso
  }
}

function renderBiblioteca() {
  const list = elements.bibList;
  list.innerHTML = "";

  if (state.biblioteca.length === 0) {
    list.innerHTML = `<p style="font-size:11px;color:var(--muted);padding:8px">Nenhum documento catalogado.</p>`;
    return;
  }

  state.biblioteca.forEach((doc) => {
    const autores = (doc.autores || []);
    const primeiroAutor = autores[0] || "";
    const partes = primeiroAutor.split(" ");
    const sobrenome = partes[partes.length - 1] || doc.id;
    const anoStr = doc.ano ? `, ${doc.ano}` : "";
    const titulo = doc.titulo || "";
    const assuntos = (doc.assuntos || []).slice(0, 4);
    const icon = iconForTipo(doc.tipo_documento);
    const isSelected = state.selectedDocId === doc.id;

    const card = document.createElement("button");
    card.className = `bib-card${isSelected ? " selected" : ""}`;
    card.type = "button";
    card.dataset.docId = doc.id;
    card.innerHTML = `
      <div class="bib-card-header">
        <span class="bib-card-icon">${icon}</span>
        <span class="bib-card-author">${escapeHtml(sobrenome)}${anoStr}</span>
      </div>
      <div class="bib-card-title">${escapeHtml(titulo.length > 40 ? titulo.slice(0, 40) + "…" : titulo)}</div>
      <div class="bib-card-tags">${assuntos.map(a => `<span class="bib-tag">${escapeHtml(a)}</span>`).join("")}</div>
    `;

    card.addEventListener("click", () => {
      if (state.selectedDocId === doc.id) {
        state.selectedDocId = null;
      } else {
        state.selectedDocId = doc.id;
        // Auto-set area filter
        if (doc.area_tematica) setSelectedArea(doc.area_tematica);
      }
      renderBiblioteca();
    });

    // Double-click to open modal
    card.addEventListener("dblclick", () => openDocModal(doc.id));

    list.appendChild(card);
  });
}

/* ─── Tab Switching (Chats ↔ Biblioteca) ─────────────────────── */

function switchTab(tabName) {
  document.querySelectorAll(".nav-item[data-tab]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  elements.areasPanel.hidden = tabName !== "chats";
  elements.bibliotecaPanel.hidden = tabName !== "biblioteca";
}

/* ─── 4.4 Document Detail Modal ──────────────────────────────── */

async function openDocModal(docId) {
  try {
    const resp = await fetch(`/biblioteca/${encodeURIComponent(docId)}`, { headers: buildHeaders(false) });
    if (!resp.ok) {
      alert("Documento não encontrado.");
      return;
    }
    const doc = await resp.json();
    renderDocModal(doc);
    elements.docModal.hidden = false;
  } catch {
    alert("Erro ao carregar documento.");
  }
}

function renderDocModal(doc) {
  const autores = (doc.autores || []).join(", ") || "Não identificado";
  const resumo = doc.resumo || "Sem resumo disponível.";
  const palavras = (doc.palavras_chave || []);
  const secoes = (doc.secoes_detectadas || []);
  const ano = doc.ano || "—";
  const inst = doc.instituicao || "—";
  const tipo = (doc.tipo_documento || "outro").replace(/_/g, " ");
  const paginas = doc.paginas_total || "—";
  const qualidade = doc.qualidade_extracao || "—";
  const pdfPath = doc.documento_original || "";

  elements.modalBody.innerHTML = `
    <h2 id="modalTitle">${escapeHtml(doc.titulo || doc.id)}</h2>
    <span class="modal-tipo">${escapeHtml(tipo)}</span>

    <div class="modal-field">
      <div class="modal-field-label">Autores</div>
      <div class="modal-field-value">${escapeHtml(autores)}</div>
    </div>

    <div class="modal-field">
      <div class="modal-field-label">Ano · Instituição</div>
      <div class="modal-field-value">${escapeHtml(String(ano))} · ${escapeHtml(inst)}</div>
    </div>

    <div class="modal-field">
      <div class="modal-field-label">Resumo</div>
      <div class="modal-field-value">${escapeHtml(resumo)}</div>
    </div>

    ${palavras.length > 0 ? `
    <div class="modal-field">
      <div class="modal-field-label">Palavras-chave</div>
      <div class="modal-tags">${palavras.map(p => `<span class="modal-tag">${escapeHtml(p)}</span>`).join("")}</div>
    </div>` : ""}

    ${secoes.length > 0 ? `
    <div class="modal-field">
      <div class="modal-field-label">Seções detectadas</div>
      <ul class="modal-secoes">${secoes.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
    </div>` : ""}

    <div class="modal-field">
      <div class="modal-field-label">Detalhes</div>
      <div class="modal-field-value">${escapeHtml(String(paginas))} páginas · qualidade: ${escapeHtml(qualidade)}</div>
    </div>

    ${pdfPath ? `<a class="modal-pdf-link" href="/${escapeHtml(pdfPath)}" target="_blank">📄 Abrir PDF original</a>` : ""}
  `;
}

function closeDocModal() {
  elements.docModal.hidden = true;
}

/* ─── Chat / API ─────────────────────────────────────────────── */

async function loadBootstrapData() {
  try {
    const response = await fetch("/modelos", { headers: buildHeaders(false) });
    if (!response.ok) return;
    const data = await response.json();
    state.defaultModel = data.default || data.modelos?.[0] || state.defaultModel;
  } catch {
    state.defaultModel = "llama3.1:8b";
  }
}

async function sendQuestion(event) {
  event.preventDefault();
  if (state.loading) return;

  const pergunta = elements.question.value.trim();
  if (!pergunta) return;

  addMessage("user", pergunta);
  elements.question.value = "";
  setLoading(true);

  try {
    const payload = {
      pergunta,
      modelo: state.defaultModel,
      top_k: 5,
      filtro_area: state.selectedArea ? [state.selectedArea] : [],
      filtro_assunto: [],
    };

    let response;
    try {
      response = await fetch("/chat", {
        method: "POST",
        headers: buildHeaders(true),
        body: JSON.stringify(payload),
      });
    } catch (networkError) {
      addMessage("assistant",
        "Não foi possível conectar ao servidor. Verifique se a API está rodando e tente novamente."
      );
      return;
    }

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Falha ao consultar o chatbot.");
    }

    addMessage("assistant", data.resposta || "Sem resposta.", data.resultados || [], data.fontes || []);
  } catch (error) {
    addMessage("assistant", `Erro: ${error.message || error}`);
  } finally {
    setLoading(false);
  }
}

function clearChat() {
  state.messages = [];
  elements.messages.querySelectorAll(".msg").forEach((message) => message.remove());
  updateWelcome();
  updateChatSummary("");
  elements.question.focus();
}

async function uploadSelectedPdfs(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length || state.loading) return;

  setLoading(true);
  const area = state.selectedArea || "ia";
  addMessage("assistant", `Recebi ${files.length} PDF(s). Atualizando a base ${labelForArea(area)}...`);

  try {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("area", area);
    formData.append("assuntos", `${area},pdf,ia`);

    let uploadResponse;
    try {
      uploadResponse = await fetch("/upload", {
        method: "POST",
        headers: buildHeaders(false),
        body: formData,
      });
    } catch (networkError) {
      addMessage("assistant",
        "Não foi possível conectar ao servidor para enviar os PDFs. Verifique se a API está rodando."
      );
      return;
    }
    const uploadData = await uploadResponse.json();

    if (!uploadResponse.ok) {
      throw new Error(uploadData.detail || "Falha ao enviar PDFs.");
    }

    let reprocessResponse;
    try {
      reprocessResponse = await fetch("/reprocessar", {
        method: "POST",
        headers: buildHeaders(false),
      });
    } catch (networkError) {
      addMessage("assistant",
        `${uploadData.total} PDF(s) enviados, mas não foi possível iniciar o reprocessamento.`
      );
      return;
    }
    const reprocessData = await reprocessResponse.json();

    if (!reprocessResponse.ok) {
      throw new Error(reprocessData.detail || "Falha ao atualizar a base RAG.");
    }

    addMessage("assistant",
      `${uploadData.total} PDF(s) adicionados e indexados. A base ${labelForArea(area)} já pode ser consultada.`
    );

    // Reload biblioteca after upload
    await loadBiblioteca();
  } catch (error) {
    addMessage("assistant", `Erro: ${error.message || error}`);
  } finally {
    elements.pdfUpload.value = "";
    setLoading(false);
  }
}

/* ─── Event Binding ──────────────────────────────────────────── */

function bindEvents() {
  elements.chatForm.addEventListener("submit", sendQuestion);
  elements.clearBtn.addEventListener("click", clearChat);
  elements.newChatBtn.addEventListener("click", clearChat);
  elements.uploadPdfBtn.addEventListener("click", () => elements.pdfUpload.click());
  elements.pdfUpload.addEventListener("change", uploadSelectedPdfs);

  elements.areaSelect.addEventListener("change", (e) => setSelectedArea(e.target.value));

  elements.agentList.addEventListener("click", (e) => {
    const button = e.target.closest(".agent-item");
    if (!button) return;
    setSelectedArea(button.dataset.area || "");
  });

  elements.toggleSidebar.addEventListener("click", () => {
    elements.sidebar.classList.toggle("open");
  });

  // Tab switching
  document.querySelectorAll(".nav-item[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Modal close
  elements.modalClose.addEventListener("click", closeDocModal);
  elements.docModal.addEventListener("click", (e) => {
    if (e.target === elements.docModal) closeDocModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !elements.docModal.hidden) closeDocModal();
  });

  // Source card click → open document modal
  elements.messages.addEventListener("click", (e) => {
    const card = e.target.closest(".source-card");
    if (!card) return;
    const docName = card.dataset.docName;
    if (!docName) return;
    // Find doc in biblioteca by documento_original basename match
    const match = state.biblioteca.find((d) => {
      const orig = d.documento_original || "";
      return orig.endsWith(docName) || docName.endsWith(d.id);
    });
    if (match) {
      openDocModal(match.id);
    }
  });
}

/* ─── Init ───────────────────────────────────────────────────── */

async function checkApiHealth() {
  try {
    const response = await fetch("/status", { headers: buildHeaders(false) });
    if (!response.ok) {
      showApiOffline();
      return;
    }
  } catch {
    showApiOffline();
  }
}

async function init() {
  bindEvents();
  setSelectedArea("");
  updateChatSummary("");
  await checkApiHealth();
  await Promise.all([loadBootstrapData(), loadFilters(), loadBiblioteca()]);
}

init();
