/**
 * HealthGuard - Popup Logic
 * Handles UI interactions, API health checks, and manual claim checking.
 */

document.addEventListener("DOMContentLoaded", () => {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const toggleScan = document.getElementById("toggle-scan");
  const manualInput = document.getElementById("manual-input");
  const checkBtn = document.getElementById("check-btn");
  const btnLoader = document.getElementById("btn-loader");
  const resultCard = document.getElementById("result-card");
  const rescanBtn = document.getElementById("rescan-btn");
  const settingsBtn = document.getElementById("settings-btn");
  const settingsPanel = document.getElementById("settings-panel");
  const apiUrlInput = document.getElementById("api-url-input");
  const saveSettings = document.getElementById("save-settings");

  // ── Initialize ───────────────────────────────────────────────────

  // Check API health
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
  chrome.storage.local.get(["enabled", "apiUrl"], (result) => {
    toggleScan.checked = result.enabled !== false;
    apiUrlInput.value = result.apiUrl || "http://localhost:8000";
  });

  // Get current tab stats
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "GET_STATS" }, (stats) => {
        if (stats) {
          document.getElementById("stat-harmful").textContent = stats.harmful || 0;
          document.getElementById("stat-misleading").textContent = stats.misleading || 0;
          document.getElementById("stat-verified").textContent = stats.verified || 0;
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

  // ── Manual Check ─────────────────────────────────────────────────

  checkBtn.addEventListener("click", async () => {
    const text = manualInput.value.trim();
    if (!text || text.length < 10) {
      manualInput.style.borderColor = "rgba(231, 76, 60, 0.5)";
      return;
    }

    manualInput.style.borderColor = "";
    checkBtn.disabled = true;
    btnLoader.style.display = "inline";
    resultCard.style.display = "none";

    chrome.runtime.sendMessage({ type: "CLASSIFY_TEXT", text }, (result) => {
      checkBtn.disabled = false;
      btnLoader.style.display = "none";

      if (!result) {
        resultCard.style.display = "block";
        document.getElementById("result-icon").textContent = "❌";
        document.getElementById("result-label").textContent = "Error";
        document.getElementById("result-label").className = "result-label";
        document.getElementById("result-confidence").textContent = "";
        document.getElementById("result-words").innerHTML = "Could not connect to API. Check settings.";
        document.getElementById("result-probs").innerHTML = "";
        return;
      }

      showResult(result);
    });
  });

  // Enter key to submit
  manualInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      checkBtn.click();
    }
  });

  function showResult(result) {
    resultCard.style.display = "block";

    const icons = { Harmful: "🚨", Misleading: "⚠️", Verified: "✅", Irrelevant: "🔘" };
    document.getElementById("result-icon").textContent = icons[result.label] || "❓";

    const labelEl = document.getElementById("result-label");
    labelEl.textContent = result.label;
    labelEl.className = `result-label ${result.label.toLowerCase()}`;

    const conf = Math.round(result.confidence * 100);
    document.getElementById("result-confidence").textContent = `${conf}%`;

    // Important words
    const wordsEl = document.getElementById("result-words");
    if (result.important_words && result.important_words.length > 0) {
      const words = result.important_words.slice(0, 5);
      wordsEl.innerHTML = "Key words: " + words.map(w => `<span>${w.word || w}</span>`).join("");
    } else {
      wordsEl.innerHTML = "";
    }

    // Probability bars
    const probsEl = document.getElementById("result-probs");
    if (result.all_probabilities) {
      probsEl.innerHTML = Object.entries(result.all_probabilities)
        .map(([label, prob]) => {
          const pct = Math.round(prob * 100);
          return `<div class="prob-bar">${label}<br>${pct}%</div>`;
        }).join("");
    } else {
      probsEl.innerHTML = "";
    }
  }

  // ── Rescan ───────────────────────────────────────────────────────

  rescanBtn.addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "RESCAN" });
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
      // Re-check health
      chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (isHealthy) => {
        statusDot.className = "status-dot " + (isHealthy ? "connected" : "disconnected");
        statusText.textContent = isHealthy ? "Connected" : "Offline";
      });
    }
  });
});
