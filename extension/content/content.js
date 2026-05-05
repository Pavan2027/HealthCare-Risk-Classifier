/**
 * HealthGuard - Content Script
 * Extracts text from web pages, classifies it, and overlays risk indicators.
 */

(function () {
  "use strict";

  if (window.__healthGuardLoaded) return;
  window.__healthGuardLoaded = true;

  const MIN_TEXT_LENGTH = 30;
  const MAX_TEXT_LENGTH = 2000;
  const SCAN_DELAY = 2000;
  const BATCH_SIZE = 10;

  let isEnabled = true;
  let scanResults = { harmful: 0, misleading: 0, verified: 0, irrelevant: 0 };

  // ── Platform-Specific Selectors ────────────────────────────────────

  function getTextElements() {
    const url = window.location.hostname;
    let selectors = [];

    if (url.includes("twitter.com") || url.includes("x.com")) {
      selectors = ['[data-testid="tweetText"]'];
    } else if (url.includes("youtube.com")) {
      selectors = ["#content-text", "yt-formatted-string.ytd-comment-renderer"];
    } else if (url.includes("reddit.com")) {
      selectors = [
        '[data-testid="post-container"] p',
        ".md p",
        "shreddit-comment p",
        '[slot="text-body"] p',
      ];
    } else {
      selectors = ["article p", "main p", ".post-content p", ".entry-content p", "p"];
    }

    const elements = [];
    for (const sel of selectors) {
      try {
        document.querySelectorAll(sel).forEach((el) => {
          if (!el.dataset.hgScanned && el.textContent.trim().length >= MIN_TEXT_LENGTH) {
            elements.push(el);
          }
        });
      } catch (e) { /* ignore invalid selectors */ }
      if (elements.length > 0) break;
    }
    return elements.slice(0, 50);
  }

  // ── Classification ─────────────────────────────────────────────────

  async function classifySingle(text) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "CLASSIFY_TEXT", text }, resolve);
    });
  }

  async function classifyBatch(texts) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "CLASSIFY_BATCH", texts }, resolve);
    });
  }

  // ── Overlay UI ─────────────────────────────────────────────────────

  function applyOverlay(element, result) {
    if (!result || !result.label) return;

    element.dataset.hgScanned = "true";
    element.dataset.hgLabel = result.label;
    element.dataset.hgConfidence = result.confidence;

    const label = result.label;
    scanResults[label.toLowerCase()] = (scanResults[label.toLowerCase()] || 0) + 1;

    if (label === "Verified" || label === "Irrelevant") return;

    // Add risk class
    element.classList.add("hg-risk", `hg-risk-${label.toLowerCase()}`);

    // Create tooltip
    const tooltip = document.createElement("div");
    tooltip.className = `hg-tooltip hg-tooltip-${label.toLowerCase()}`;

    const icon = label === "Harmful" ? "🚨" : "⚠️";
    const conf = Math.round(result.confidence * 100);

    tooltip.innerHTML = `
      <div class="hg-tooltip-header">
        <span class="hg-tooltip-icon">${icon}</span>
        <span class="hg-tooltip-label">${label}</span>
        <span class="hg-tooltip-conf">${conf}%</span>
      </div>
      <div class="hg-tooltip-body">
        ${result.important_words && result.important_words.length > 0
          ? `<div class="hg-tooltip-words">Key: ${result.important_words.slice(0, 4).map(w => w.word || w).join(", ")}</div>`
          : ""
        }
        <div class="hg-tooltip-cta">Click for details</div>
      </div>
    `;

    // Position tooltip on hover
    element.style.position = element.style.position || "relative";
    element.appendChild(tooltip);

    // Click to expand
    element.addEventListener("click", () => {
      tooltip.classList.toggle("hg-tooltip-expanded");
    });
  }

  // ── Floating Badge ─────────────────────────────────────────────────

  function updateFloatingBadge() {
    let badge = document.getElementById("hg-floating-badge");
    const total = scanResults.harmful + scanResults.misleading;

    if (total === 0) {
      if (badge) badge.style.display = "none";
      return;
    }

    if (!badge) {
      badge = document.createElement("div");
      badge.id = "hg-floating-badge";
      document.body.appendChild(badge);
    }

    badge.style.display = "flex";
    badge.innerHTML = `
      <div class="hg-badge-icon">🛡️</div>
      <div class="hg-badge-content">
        <div class="hg-badge-title">HealthGuard</div>
        <div class="hg-badge-stats">
          ${scanResults.harmful > 0 ? `<span class="hg-stat-harmful">${scanResults.harmful} harmful</span>` : ""}
          ${scanResults.misleading > 0 ? `<span class="hg-stat-misleading">${scanResults.misleading} misleading</span>` : ""}
        </div>
      </div>
      <button class="hg-badge-close" id="hg-badge-close">&times;</button>
    `;

    const closeBtn = document.getElementById("hg-badge-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        badge.style.display = "none";
      });
    }

    // Update extension badge
    chrome.runtime.sendMessage({
      type: "UPDATE_BADGE",
      counts: { harmful: scanResults.harmful, misleading: scanResults.misleading },
    });
  }

  // ── Main Scan ──────────────────────────────────────────────────────

  async function scanPage() {
    if (!isEnabled) return;

    const elements = getTextElements();
    if (elements.length === 0) return;

    // Process in batches
    for (let i = 0; i < elements.length; i += BATCH_SIZE) {
      const batch = elements.slice(i, i + BATCH_SIZE);
      const texts = batch.map((el) => el.textContent.trim().slice(0, MAX_TEXT_LENGTH));

      try {
        const batchResult = await classifyBatch(texts);
        if (batchResult && batchResult.predictions) {
          batchResult.predictions.forEach((result, idx) => {
            if (batch[idx]) applyOverlay(batch[idx], result);
          });
        }
      } catch {
        // Fallback: classify individually
        for (const el of batch) {
          try {
            const result = await classifySingle(el.textContent.trim().slice(0, MAX_TEXT_LENGTH));
            if (result) applyOverlay(el, result);
          } catch { /* skip */ }
        }
      }
    }

    updateFloatingBadge();
  }

  // ── Initialization ─────────────────────────────────────────────────

  // Check if enabled
  chrome.storage.local.get(["enabled"], (result) => {
    isEnabled = result.enabled !== false;
    if (isEnabled) {
      setTimeout(scanPage, SCAN_DELAY);

      // Watch for dynamic content
      const observer = new MutationObserver(() => {
        clearTimeout(window.__hgRescanTimer);
        window.__hgRescanTimer = setTimeout(scanPage, 3000);
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }
  });

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "TOGGLE_SCAN") {
      isEnabled = msg.enabled;
      if (isEnabled) scanPage();
      sendResponse({ ok: true });
    }
    if (msg.type === "RESCAN") {
      scanResults = { harmful: 0, misleading: 0, verified: 0, irrelevant: 0 };
      document.querySelectorAll("[data-hg-scanned]").forEach((el) => {
        delete el.dataset.hgScanned;
        el.classList.remove("hg-risk", "hg-risk-harmful", "hg-risk-misleading");
        el.querySelector(".hg-tooltip")?.remove();
      });
      scanPage();
      sendResponse({ ok: true });
    }
    if (msg.type === "GET_STATS") {
      sendResponse(scanResults);
    }
  });
})();
