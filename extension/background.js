const API_BASE = "http://127.0.0.1:8000";
const CACHE_TTL_MS = 30 * 1000;
const MAX_CACHE_ENTRIES = 100;

const cache = new Map();

function getCachedResult(url) {
  const entry = cache.get(url);

  if (!entry) {
    return null;
  }

  if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
    cache.delete(url);
    return null;
  }

  return entry.data;
}

function saveCachedResult(url, data) {
  cache.set(url, {
    timestamp: Date.now(),
    data
  });

  while (cache.size > MAX_CACHE_ENTRIES) {
    const oldestKey = cache.keys().next().value;
    cache.delete(oldestKey);
  }
}

async function fetchJson(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message =
        typeof data.detail === "string"
          ? data.detail
          : `API returned HTTP ${response.status}`;

      throw new Error(message);
    }

    return data;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function getApiHealth() {
  try {
    const data = await fetchJson(`${API_BASE}/health`, {}, 3000);

    return {
      online: data.status === "ok" && data.model_loaded === true,
      message:
        data.status === "ok" && data.model_loaded === true
          ? "API พร้อมใช้งาน"
          : "API ทำงาน แต่ Model ไม่พร้อมใช้งาน"
    };
  } catch {
    return {
      online: false,
      message: "เชื่อมต่อ API ไม่ได้"
    };
  }
}

async function analyzeUrl(url) {
  const cached = getCachedResult(url);

  if (cached) {
    return {
      ...cached,
      cached: true
    };
  }

  const result = await fetchJson(
    `${API_BASE}/api/v1/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url })
    },
    10000
  );

  saveCachedResult(url, result);

  return {
    ...result,
    cached: false
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "API_HEALTH") {
    getApiHealth()
      .then(sendResponse)
      .catch(() => sendResponse({
        online: false,
        message: "ตรวจสอบ API ไม่สำเร็จ"
      }));

    return true;
  }

  if (message?.type === "ANALYZE_URL") {
    const url = message.url;

    if (typeof url !== "string" || url.length === 0 || url.length > 8192) {
      sendResponse({
        ok: false,
        error: "URL ไม่ถูกต้องหรือยาวเกินกำหนด"
      });

      return false;
    }

    analyzeUrl(url)
      .then((result) => sendResponse({
        ok: true,
        result
      }))
      .catch((error) => sendResponse({
        ok: false,
        error: error?.message || "ไม่สามารถวิเคราะห์ URL ได้"
      }));

    return true;
  }

  return false;
});
