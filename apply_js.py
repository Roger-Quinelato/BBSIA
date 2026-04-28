import os

def update_js():
    path = "web/app.js"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_send = """async function sendQuestion(event) {
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
}"""

    new_send = """async function sendQuestion(event) {
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

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Falha ao consultar o chatbot.");
    }

    // Criar placeholder para a mensagem
    state.messages.push({ role: "assistant", content: "", resultados: [] });
    const msgIndex = state.messages.length - 1;

    const wrapper = document.createElement("article");
    wrapper.className = "msg assistant";
    wrapper.innerHTML = `
      <div class="role">BBSIA</div>
      <div class="content"><span class="streaming-cursor">▌</span></div>
    `;
    elements.messages.appendChild(wrapper);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    const contentDiv = wrapper.querySelector(".content");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let textBuffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\\n");
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.type === "metadata") {
            state.messages[msgIndex].resultados = data.resultados || [];
            const sourcesHtml = renderRichSources(data.resultados || []);
            if (sourcesHtml) {
              const div = document.createElement("div");
              div.innerHTML = sourcesHtml;
              wrapper.appendChild(div.firstElementChild);
            }
            elements.messages.scrollTop = elements.messages.scrollHeight;
          } else if (data.type === "token") {
            textBuffer += data.token;
            state.messages[msgIndex].content = textBuffer;
            contentDiv.innerHTML = escapeHtml(textBuffer).replaceAll("\\n", "<br>") + '<span class="streaming-cursor">▌</span>';
            elements.messages.scrollTop = elements.messages.scrollHeight;
          } else if (data.type === "error") {
            textBuffer += "\\n\\n[Erro: " + data.message + "]";
            contentDiv.innerHTML = escapeHtml(textBuffer).replaceAll("\\n", "<br>");
            elements.messages.scrollTop = elements.messages.scrollHeight;
          }
        } catch (e) {
          console.error("Parse error on chunk:", line);
        }
      }
    }
    // Remove o cursor
    contentDiv.innerHTML = escapeHtml(textBuffer).replaceAll("\\n", "<br>");
  } catch (error) {
    addMessage("assistant", `Erro: ${error.message || error}`);
  } finally {
    setLoading(false);
  }
}"""

    content = content.replace(old_send, new_send)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

update_js()
