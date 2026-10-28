const currentUrlElement = document.getElementById("currentUrl");
const apiDot = document.getElementById("apiDot");
const apiStatus = document.getElementById("apiStatus");
const resultCard = document.getElementById("resultCard");
const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const resultDescription = document.getElementById("resultDescription");
const riskScore = document.getElementById("riskScore");
const scoreBar = document.getElementById("scoreBar");
const reasonsList = document.getElementById("reasons");
const scanButton = document.getElementById("scanButton");
const modelName = document.getElementById("modelName");
const processingTime = document.getElementById("processingTime");

let activeTabUrl = "";

function setApiStatus(online, message) {
  apiDot.className = online ? "dot online" : "dot offline";
  apiStatus.textContent = message;
}

function setPending(message = "กำลังวิเคราะห์ URL...") {
  resultCard.className = "result-card pending";
  resultIcon.textContent = "⏳";
  resultTitle.textContent = "กำลังวิเคราะห์";
  resultDescription.textContent = message;
  riskScore.textContent = "--%";
  scoreBar.style.width = "0%";
  reasonsList.replaceChildren();

  const item = document.createElement("li");
  item.textContent = "กำลังรอผลจาก AI Model...";
  reasonsList.appendChild(item);

  modelName.textContent = "Model: --";
  processingTime.textContent = "เวลา: --";
}

function setError(message) {
  resultCard.className = "result-card error";
  resultIcon.textContent = "⚠️";
  resultTitle.textContent = "ตรวจสอบไม่สำเร็จ";
  resultDescription.textContent = message;
  riskScore.textContent = "--%";
  scoreBar.style.width = "0%";

  reasonsList.replaceChildren();

  const item = document.createElement("li");
  item.textContent =
    "ไม่สามารถยืนยันความปลอดภัยได้ กรุณาลองใหม่เมื่อ API พร้อมใช้งาน";
  reasonsList.appendChild(item);

  modelName.textContent = "Model: --";
  processingTime.textContent = "เวลา: --";
}

function displayResult(data) {
  const score = Math.max(
    0,
    Math.min(1, Number(data.risk_score) || 0)
  );

  const percent = Math.round(score * 100);
  riskScore.textContent = `${percent}%`;
  scoreBar.style.width = `${percent}%`;

  if (data.prediction === "phishing") {
    resultCard.className =
      data.risk_level === "high"
        ? "result-card high"
        : "result-card medium";

    resultIcon.textContent =
      data.risk_level === "high" ? "🚨" : "⚠️";

    resultTitle.textContent =
      data.risk_level === "high"
        ? "พบความเสี่ยงสูง"
        : "พบความเสี่ยงที่ควรระวัง";

    resultDescription.textContent =
      "AI พบรูปแบบ URL ที่อาจเกี่ยวข้องกับ Phishing";
  } else {
    resultCard.className =
      data.risk_level === "low"
        ? "result-card safe"
        : "result-card medium";

    resultIcon.textContent =
      data.risk_level === "low" ? "✅" : "⚠️";

    resultTitle.textContent =
      data.risk_level === "low"
        ? "ความเสี่ยงที่ประเมินได้ต่ำ"
        : "ยังมีข้อควรระวัง";

    resultDescription.textContent =
      "ไม่พบรูปแบบที่ผ่านเกณฑ์ Phishing ของโมเดล";
  }

  reasonsList.replaceChildren();

  const reasons = Array.isArray(data.reasons) ? data.reasons : [];

  for (const reason of reasons) {
    const item = document.createElement("li");
    item.textContent = String(reason);
    reasonsList.appendChild(item);
  }

  if (reasons.length === 0) {
    const item = document.createElement("li");
    item.textContent = "ไม่มีเหตุผลเพิ่มเติมจาก API";
    reasonsList.appendChild(item);
  }

  modelName.textContent = `Model: ${data.model_name || "unknown"}`;

  const elapsed = Number(data.processing_time_ms);
  processingTime.textContent = Number.isFinite(elapsed)
    ? `เวลา: ${elapsed.toFixed(0)} ms`
    : "เวลา: --";

  if (data.cached) {
    processingTime.textContent += " (cache)";
  }
}

function sendBackgroundMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }

      resolve(response);
    });
  });
}

async function refreshApiHealth() {
  try {
    const health = await sendBackgroundMessage({
      type: "API_HEALTH"
    });

    setApiStatus(Boolean(health?.online), health?.message || "ไม่ทราบสถานะ");
  } catch {
    setApiStatus(false, "ติดต่อ Background Service ไม่ได้");
  }
}

async function scanCurrentUrl() {
  scanButton.disabled = true;

  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    const url = tab?.url || "";
    activeTabUrl = url;
    currentUrlElement.textContent = url || "ไม่พบ URL";

    if (!url) {
      setError("ไม่สามารถอ่าน URL ของแท็บนี้ได้");
      return;
    }

    const lowerUrl = url.toLowerCase();

    if (
      lowerUrl.startsWith("chrome://") ||
      lowerUrl.startsWith("chrome-extension://") ||
      lowerUrl.startsWith("edge://") ||
      lowerUrl.startsWith("about:") ||
      lowerUrl.startsWith("devtools://")
    ) {
      setError("Chrome ไม่อนุญาตให้ตรวจสอบหน้าระบบภายในเหล่านี้");
      return;
    }

    if (!/^https?:\/\//i.test(url)) {
      setError("รองรับเฉพาะเว็บไซต์ที่ใช้ HTTP หรือ HTTPS");
      return;
    }

    setPending();

    await refreshApiHealth();

    const response = await sendBackgroundMessage({
      type: "ANALYZE_URL",
      url
    });

    if (!response?.ok) {
      setError(response?.error || "API ไม่สามารถวิเคราะห์ URL ได้");
      return;
    }

    displayResult(response.result);
  } catch (error) {
    setError(error?.message || "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ");
  } finally {
    scanButton.disabled = false;
  }
}

scanButton.addEventListener("click", scanCurrentUrl);

document.addEventListener("DOMContentLoaded", () => {
  scanCurrentUrl();
});
