/**
 * HealthGuard - Background Service Worker
 * Handles API communication, badge updates, and message passing.
 */

const DEFAULT_API_URL = "http://localhost:8000";
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// In-memory cache for recent classifications
const classificationCache = new Map();

// Badge colors per risk level
const BADGE_COLORS = {
  Harmful: "#e74c3c",
  Misleading: "#f39c12",
  Verified: "#2ecc71",
  Irrelevant: "#95a5a6",
};

// ── API Communication ──────────────────────────────────────────────────────

async function getApiUrl() {
  const result = await chrome.storage.local.get(["apiUrl"]);
  return result.apiUrl || DEFAULT_API_URL;
}

async function classifyText(text) {
  // Check cache
  const cacheKey = text.slice(0, 100);
  const cached = classificationCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.result;
  }

  const apiUrl = await getApiUrl();

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, explain: true, use_llm: false }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const result = await response.json();

    // Cache result
    classificationCache.set(cacheKey, { result, timestamp: Date.now() });

    return result;
  } catch (error) {
    console.error("HealthGuard API error:", error);
    return null;
  }
}

async function classifyBatch(texts) {
  const apiUrl = await getApiUrl();
  try {
    const response = await fetch(`${apiUrl}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: texts.slice(0, 32), explain: false }),
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("HealthGuard batch API error:", error);
    return null;
  }
}

async function checkApiHealth() {
  const apiUrl = await getApiUrl();
  try {
    const response = await fetch(`${apiUrl}/health`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}

// ── Message Handling ───────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CLASSIFY_TEXT") {
    classifyText(message.text).then(sendResponse);
    return true; // Keep channel open for async
  }

  if (message.type === "CLASSIFY_BATCH") {
    classifyBatch(message.texts).then(sendResponse);
    return true;
  }

  if (message.type === "CHECK_HEALTH") {
    checkApiHealth().then(sendResponse);
    return true;
  }

  if (message.type === "UPDATE_BADGE") {
    const { harmful, misleading } = message.counts;
    const total = harmful + misleading;
    if (total > 0) {
      chrome.action.setBadgeText({ text: String(total), tabId: sender.tab.id });
      chrome.action.setBadgeBackgroundColor({
        color: harmful > 0 ? BADGE_COLORS.Harmful : BADGE_COLORS.Misleading,
        tabId: sender.tab.id,
      });
    } else {
      chrome.action.setBadgeText({ text: "", tabId: sender.tab.id });
    }
    sendResponse({ ok: true });
    return false;
  }

  if (message.type === "GET_SCAN_RESULTS") {
    sendResponse({ cache: Object.fromEntries(classificationCache) });
    return false;
  }
});

// Clean old cache entries periodically
setInterval(() => {
  const now = Date.now();
  for (const [key, value] of classificationCache) {
    if (now - value.timestamp > CACHE_TTL) {
      classificationCache.delete(key);
    }
  }
}, 60 * 1000);
