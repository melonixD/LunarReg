const byId = (id) => document.getElementById(id);

const state = {
  fixed: null,
  moving: null,
  connected: false,
  outputUrl: null
};

const els = {
  fixedInput: byId("fixed-input"), movingInput: byId("moving-input"),
  fixedPreview: byId("fixed-preview"), movingPreview: byId("moving-preview"),
  fixedBox: byId("fixed-box"), movingBox: byId("moving-box"),
  fixedState: byId("fixed-state"), movingState: byId("moving-state"),
  run: byId("start-matching"), loadSample: byId("load-sample"),
  engine: byId("engine-select"), message: byId("run-message"),
  results: byId("results"), canvas: byId("registered-canvas"),
  resultImage: byId("registered-image"), resultTitle: byId("result-title"),
  resultNote: byId("result-note"), download: byId("download-result")
};

async function checkStatus() {
  try {
    const response = await fetch("/api/model-status");
    const data = await response.json();
    state.connected = Boolean(data.connected);
    byId("status-dot").classList.add("online");
    byId("model-status").textContent = state.connected
      ? "Remote SIFT + MAGSAC service connected"
      : "Browser baseline ready · registration model not trained";
  } catch {
    byId("status-dot").classList.add("online");
    byId("model-status").textContent = "Browser baseline ready · registration model not trained";
  }
}

function setFile(kind, blob, name) {
  state[kind] = { blob, name };
  const preview = kind === "fixed" ? els.fixedPreview : els.movingPreview;
  const box = kind === "fixed" ? els.fixedBox : els.movingBox;
  const label = kind === "fixed" ? els.fixedState : els.movingState;
  preview.src = URL.createObjectURL(blob);
  box.classList.add("has-image");
  label.textContent = name.length > 22 ? `${name.slice(0, 19)}…` : name;
  label.classList.add("ready");
  els.message.classList.remove("error");
  els.message.textContent = state.fixed && state.moving ? "Pair ready to register." : "Select the second image.";
}

els.fixedInput.addEventListener("change", () => {
  const file = els.fixedInput.files?.[0];
  if (file) setFile("fixed", file, file.name);
});
els.movingInput.addEventListener("change", () => {
  const file = els.movingInput.files?.[0];
  if (file) setFile("moving", file, file.name);
});

els.loadSample.addEventListener("click", async () => {
  setBusy(true, "Loading sample…");
  try {
    const [fixedResponse, movingResponse] = await Promise.all([
      fetch("/samples/reference.jpg"), fetch("/samples/moving.jpg")
    ]);
    if (!fixedResponse.ok || !movingResponse.ok) throw new Error("Sample files unavailable");
    setFile("fixed", await fixedResponse.blob(), "LRO-reference.jpg");
    setFile("moving", await movingResponse.blob(), "LRO-shifted.jpg");
    els.message.textContent = "Included LRO demonstration pair loaded.";
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

function setBusy(busy, text = "Processing…") {
  els.run.disabled = busy;
  els.loadSample.disabled = busy;
  els.run.textContent = busy ? text : "Run registration →";
}

function showError(message) {
  els.message.textContent = message;
  els.message.classList.add("error");
}

function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(blob);
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not decode an uploaded image")); };
    image.src = url;
  });
}

function grayscale(image, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const raw = new Float32Array(width * height);
  let mean = 0;
  for (let i = 0; i < raw.length; i++) {
    const p = i * 4;
    raw[i] = .299 * pixels[p] + .587 * pixels[p + 1] + .114 * pixels[p + 2];
    mean += raw[i];
  }
  mean /= raw.length;
  let variance = 0;
  for (const value of raw) variance += (value - mean) ** 2;
  const sd = Math.sqrt(variance / raw.length) || 1;
  const normalized = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) normalized[i] = (raw[i] - mean) / sd;
  return { raw, normalized };
}

function shiftScore(fixed, moving, width, height, dx, dy, stride) {
  const x0 = Math.max(0, dx), x1 = Math.min(width, width + dx);
  const y0 = Math.max(0, dy), y1 = Math.min(height, height + dy);
  if (x1 - x0 < width * .45 || y1 - y0 < height * .45) return Infinity;
  let sum = 0, count = 0;
  for (let y = y0; y < y1; y += stride) {
    for (let x = x0; x < x1; x += stride) {
      const difference = fixed[y * width + x] - moving[(y - dy) * width + (x - dx)];
      sum += difference * difference;
      count++;
    }
  }
  return sum / Math.max(count, 1);
}

function estimateTranslation(fixed, moving, width, height) {
  const limitX = Math.floor(width * .26), limitY = Math.floor(height * .26);
  let best = { dx: 0, dy: 0, score: Infinity };
  for (let dy = -limitY; dy <= limitY; dy += 4) {
    for (let dx = -limitX; dx <= limitX; dx += 4) {
      const score = shiftScore(fixed, moving, width, height, dx, dy, 3);
      if (score < best.score) best = { dx, dy, score };
    }
  }
  const coarse = best;
  for (let dy = coarse.dy - 5; dy <= coarse.dy + 5; dy++) {
    for (let dx = coarse.dx - 5; dx <= coarse.dx + 5; dx++) {
      const score = shiftScore(fixed, moving, width, height, dx, dy, 2);
      if (score < best.score) best = { dx, dy, score };
    }
  }
  return best;
}

function calculateMetrics(fixed, moving, width, height, dx, dy) {
  const errors = [];
  let squared = 0, inliers = 0;
  const threshold = 28;
  const x0 = Math.max(0, dx), x1 = Math.min(width, width + dx);
  const y0 = Math.max(0, dy), y1 = Math.min(height, height + dy);
  for (let y = y0; y < y1; y += 2) {
    for (let x = x0; x < x1; x += 2) {
      const error = Math.abs(fixed[y * width + x] - moving[(y - dy) * width + (x - dx)]);
      errors.push(error); squared += error * error;
      if (error <= threshold) inliers++;
    }
  }
  errors.sort((a, b) => a - b);
  const count = Math.max(errors.length, 1);
  return {
    rmse: Math.sqrt(squared / count), inliers,
    inlier_ratio: inliers / count,
    median_error: errors[Math.floor(errors.length / 2)] || 0,
    max_error: errors.at(-1) || 0,
    coverage: ((x1 - x0) * (y1 - y0)) / (width * height)
  };
}

async function runBrowserRegistration() {
  const [fixedImage, movingImage] = await Promise.all([
    blobToImage(state.fixed.blob), blobToImage(state.moving.blob)
  ]);
  const analysisSize = 192;
  const fixedData = grayscale(fixedImage, analysisSize, analysisSize);
  const movingData = grayscale(movingImage, analysisSize, analysisSize);
  await new Promise((resolve) => requestAnimationFrame(resolve));
  const shift = estimateTranslation(fixedData.normalized, movingData.normalized, analysisSize, analysisSize);
  const metrics = calculateMetrics(fixedData.raw, movingData.raw, analysisSize, analysisSize, shift.dx, shift.dy);

  const maxSide = 1400;
  const scale = Math.min(1, maxSide / Math.max(fixedImage.naturalWidth, fixedImage.naturalHeight));
  const width = Math.round(fixedImage.naturalWidth * scale);
  const height = Math.round(fixedImage.naturalHeight * scale);
  const scaleX = width / analysisSize, scaleY = height / analysisSize;
  els.canvas.width = width; els.canvas.height = height;
  const context = els.canvas.getContext("2d");
  context.fillStyle = "#11151c"; context.fillRect(0, 0, width, height);
  context.drawImage(fixedImage, 0, 0, width, height);
  context.globalAlpha = .52;
  context.globalCompositeOperation = "screen";
  context.drawImage(movingImage, shift.dx * scaleX, shift.dy * scaleY, width, height);
  context.globalAlpha = 1; context.globalCompositeOperation = "source-over";
  els.canvas.style.display = "block"; els.resultImage.style.display = "none";
  state.outputUrl = null;

  showResults(metrics, "Browser translation baseline", `Estimated translation: Δx ${(shift.dx * scaleX).toFixed(1)} px, Δy ${(shift.dy * scaleY).toFixed(1)} px. RMSE and inliers are intensity-based for this browser baseline; they are not feature reprojection error or proof of sub-pixel accuracy.`);
}

async function runBackendRegistration() {
  const form = new FormData();
  form.append("fixed", state.fixed.blob, state.fixed.name);
  form.append("moving", state.moving.blob, state.moving.name);
  const response = await fetch("/api/register", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.detail || "Registration service failed");
  const src = data.registered_image?.startsWith("data:") ? data.registered_image : `data:image/png;base64,${data.registered_image}`;
  els.resultImage.src = src;
  els.resultImage.style.display = "block"; els.canvas.style.display = "none";
  state.outputUrl = src;
  showResults(data.metrics, "SIFT + MAGSAC registration", data.note || "Metrics use inlier feature reprojection errors. Review match coverage before accepting a result.");
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function showResults(metrics, title, note) {
  els.resultTitle.textContent = title;
  byId("metric-rmse").textContent = `${formatNumber(metrics.rmse)} px`;
  byId("metric-inliers").textContent = Number.isFinite(Number(metrics.inliers)) ? Math.round(metrics.inliers).toLocaleString() : "—";
  byId("metric-ratio").textContent = formatNumber(metrics.inlier_ratio, 3);
  byId("metric-median").textContent = `${formatNumber(metrics.median_error)} px`;
  byId("metric-max").textContent = `${formatNumber(metrics.max_error)} px`;
  byId("metric-coverage").textContent = `${formatNumber(Number(metrics.coverage) * 100, 1)}%`;
  els.resultNote.textContent = note;
  els.results.classList.remove("hidden");
  els.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

els.run.addEventListener("click", async () => {
  if (!state.fixed || !state.moving) return showError("Select both a fixed and a moving image first.");
  els.message.classList.remove("error");
  setBusy(true, "Registering…");
  try {
    const choice = els.engine.value;
    if (choice === "backend" && !state.connected) throw new Error("The SIFT service is not connected. Choose Browser baseline or configure INFERENCE_API_URL.");
    if (choice === "backend" || (choice === "auto" && state.connected)) await runBackendRegistration();
    else await runBrowserRegistration();
    els.message.textContent = "Registration completed with measured output.";
  } catch (error) {
    if (els.engine.value === "auto" && state.connected) {
      els.message.textContent = "Remote service unavailable; completed with browser fallback.";
      await runBrowserRegistration();
    } else showError(error.message);
  } finally {
    setBusy(false);
  }
});

els.download.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "lunarreg-registered-output.png";
  link.href = state.outputUrl || els.canvas.toDataURL("image/png");
  link.click();
});

const craterInput = byId("crater-input");
const craterButton = byId("detect-craters");
const craterMessage = byId("crater-message");
const craterResult = byId("crater-result");

craterInput.addEventListener("change", () => {
  const file = craterInput.files?.[0];
  byId("crater-file").textContent = file ? file.name : "Choose image";
});

craterButton.addEventListener("click", async () => {
  const file = craterInput.files?.[0];
  if (!file) { craterMessage.textContent = "Choose a lunar image first."; return; }
  if (!state.connected) { craterMessage.textContent = "Inference service is not connected. Deploy it and set INFERENCE_API_URL in Vercel."; return; }
  craterButton.disabled = true;
  craterButton.textContent = "Detecting…";
  craterMessage.textContent = "Running the supplied YOLOv8n model…";
  try {
    const form = new FormData();
    form.append("image", file, file.name);
    const response = await fetch("/api/detect-craters", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || "Crater detection failed");
    craterResult.src = data.annotated_image;
    craterResult.style.display = "block";
    craterMessage.textContent = `${data.count} candidate craters detected. ${data.note}`;
  } catch (error) {
    craterMessage.textContent = error.message;
  } finally {
    craterButton.disabled = false;
    craterButton.textContent = "Detect craters";
  }
});

checkStatus();
