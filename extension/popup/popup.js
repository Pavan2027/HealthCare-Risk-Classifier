/**
 * HealthGuard - Popup Logic
 * Handles UI interactions, API health checks, manual claim checking,
 * and rendering of attention highlights, LLM explanations, and probability bars.
 */

document.addEventListener("DOMContentLoaded", () => {
  const statusDot      = document.getElementById("status-dot");
  const statusText     = document.getElementById("status-text");
  const toggleScan     = document.getElementById("toggle-scan");
  const useLlmToggle   = document.getElementById("use-llm-toggle");
  const manualInput    = document.getElementById("manual-input");
  const checkBtn       = document.getElementById("check-btn");
  const btnLoader      = document.getElementById("btn-loader");
  const resultCard     = document.getElementById("result-card");
  const disagreeEl     = document.getElementById("disagree-badge");
  const explanationEl  = document.getElementById("result-explanation");
  const rescanBtn      = document.getElementById("rescan-btn");
  const settingsBtn    = document.getElementById("settings-btn");
  const settingsPanel  = document.getElementById("settings-panel");
  const apiUrlInput    = document.getElementById("api-url-input");
  const saveSettings   = document.getElementById("save-settings");

  // ── Initialize ───────────────────────────────────────────────────

  // Check API health on open
  chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (isHealthy) => {
    if (isHealthy) {
      statusDot.classList.add("connected");
      statusText.textContent = "Connected";
    } else {
      statusDot.classList.add("disconnected");
      statusText.textContent = "Offline";
    }
  });

  // Load saved settings
  chrome.storage.local.get(["enabled", "apiUrl", "useLlm"], (result) => {
    toggleScan.checked = result.enabled !== false;
    apiUrlInput.value  = result.apiUrl || "http://localhost:8000";
    useLlmToggle.checked = result.useLlm === true;
  });

  // Get current tab stats
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "GET_STATS" }, (stats) => {
        if (stats) {
          document.getElementById("stat-harmful").textContent    = stats.harmful    || 0;
          document.getElementById("stat-misleading").textContent = stats.misleading || 0;
          document.getElementById("stat-verified").textContent   = stats.verified   || 0;
          document.getElementById("stat-irrelevant").textContent = stats.irrelevant || 0;
        }
      });
    }
  });

  // ── Toggle Scanning ──────────────────────────────────────────────

  toggleScan.addEventListener("change", () => {
    const enabled = toggleScan.checked;
    chrome.storage.local.set({ enabled });
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "TOGGLE_SCAN", enabled });
      }
    });
  });

  // Persist LLM preference
  useLlmToggle.addEventListener("change", () => {
    chrome.storage.local.set({ useLlm: useLlmToggle.checked });
  });

  // ── Manual Check ─────────────────────────────────────────────────

  checkBtn.addEventListener("click", async () => {
    const text = manualInput.value.trim();
    if (!text || text.length < 10) {
      manualInput.classList.add("input-error");
      setTimeout(() => manualInput.classList.remove("input-error"), 800);
      return;
    }

    checkBtn.disabled = true;
    btnLoader.style.display = "inline-flex";
    document.querySelector(".btn-text").style.display = "none";
    resultCard.style.display = "none";

    chrome.runtime.sendMessage(
      { type: "CLASSIFY_TEXT", text, useLlm: useLlmToggle.checked },
      (result) => {
        checkBtn.disabled = false;
        btnLoader.style.display = "none";
        document.querySelector(".btn-text").style.display = "";

        if (!result) {
          showError("Could not connect to API. Check Settings.");
          return;
        }

        showResult(result);
      }
    );
  });

  // Enter key to submit
  manualInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      checkBtn.click();
    }
  });

  // ── Result Rendering ─────────────────────────────────────────────

  const LABEL_META = {
    Harmful:    { icon: "🚨", color: "#e74c3c" },
    Misleading: { icon: "⚠️",  color: "#f39c12" },
    Verified:   { icon: "✅", color: "#2ecc71" },
    Irrelevant: { icon: "🔘", color: "#95a5a6" },
  };

  function showError(msg) {
    resultCard.style.display = "block";
    document.getElementById("result-icon").textContent  = "❌";
    const labelEl = document.getElementById("result-label");
    labelEl.textContent = "Error";
    labelEl.className   = "result-label";
    document.getElementById("result-confidence").textContent = "";
    document.getElementById("result-words").innerHTML = `<span class="error-msg">${msg}</span>`;
    document.getElementById("result-probs").innerHTML = "";
    disagreeEl.style.display    = "none";
    explanationEl.style.display = "none";
  }

  function showResult(result) {
    resultCard.style.display = "block";
    const meta = LABEL_META[result.label] || { icon: "❓", color: "#aaa" };

    document.getElementById("result-icon").textContent = meta.icon;

    const labelEl = document.getElementById("result-label");
    labelEl.textContent = result.label;
    labelEl.className   = `result-label ${result.label.toLowerCase()}`;

    const conf = Math.round(result.confidence * 100);
    document.getElementById("result-confidence").textContent = `${conf}%`;

    // ── Disagreement badge
    if (result.disagreement) {
      disagreeEl.style.display = "flex";
      disagreeEl.textContent   = `⚠ DeBERTa says "${result.label}" · LLM says "${result.llm_label}"`;
    } else {
      disagreeEl.style.display = "none";
    }

    // ── Important words
    const wordsEl = document.getElementById("result-words");
    if (result.important_words && result.important_words.length > 0) {
      wordsEl.innerHTML = result.important_words
        .slice(0, 6)
        .map((w) => {
          const word  = w.word || w;
          const score = w.score != null ? Math.round(w.score * 100) : null;
          const tip   = score != null ? ` title="Attention: ${score}%"` : "";
          return `<span class="word-chip"${tip}>${word}</span>`;
        })
        .join("");
    } else {
      wordsEl.innerHTML = "";
    }

    // ── LLM Explanation
    if (result.explanation) {
      explanationEl.style.display = "block";
      explanationEl.textContent   = result.explanation;
    } else {
      explanationEl.style.display = "none";
    }

    // ── Probability bars
    renderProbBars(result.all_probabilities, result.label);
  }

  function renderProbBars(probs, predictedLabel) {
    const probsEl = document.getElementById("result-probs");
    if (!probs || Object.keys(probs).length === 0) {
      probsEl.innerHTML = "";
      return;
    }

    const ORDER = ["Harmful", "Misleading", "Verified", "Irrelevant"];
    const colors = {
      Harmful:    "#e74c3c",
      Misleading: "#f39c12",
      Verified:   "#2ecc71",
      Irrelevant: "#95a5a6",
    };

    probsEl.innerHTML = ORDER.map((label) => {
      const val = probs[label] ?? 0;
      const pct = Math.round(val * 100);
      const color = colors[label] || "#aaa";
      const active = label === predictedLabel ? "prob-row-active" : "";
      return `
        <div class="prob-row ${active}">
          <span class="prob-label">${label}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${pct}%; background:${color};"></div>
          </div>
          <span class="prob-pct">${pct}%</span>
        </div>`;
    }).join("");
  }

  // ── Rescan ───────────────────────────────────────────────────────

  rescanBtn.addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "RESCAN" });
        // Reset stats display
        ["stat-harmful", "stat-misleading", "stat-verified", "stat-irrelevant"]
          .forEach((id) => (document.getElementById(id).textContent = "0"));
      }
    });
  });

  // ── Settings ─────────────────────────────────────────────────────

  settingsBtn.addEventListener("click", () => {
    const isVisible = settingsPanel.style.display !== "none";
    settingsPanel.style.display = isVisible ? "none" : "block";
  });

  saveSettings.addEventListener("click", () => {
    const url = apiUrlInput.value.trim();
    if (url) {
      chrome.storage.local.set({ apiUrl: url });
      settingsPanel.style.display = "none";
      chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (isHealthy) => {
        statusDot.className  = "status-dot " + (isHealthy ? "connected" : "disconnected");
        statusText.textContent = isHealthy ? "Connected" : "Offline";
      });
    }
  });
});
