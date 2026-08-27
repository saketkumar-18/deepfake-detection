/**
 * main.js — orchestrates upload → frame sampling → face detection → scoring → verdict.
 * Everything runs on-device; no network calls except model/CDN fetches.
 */
import { DeepfakeScreener } from "./inference.js";
import { getFaceDetector, cropLargestFace } from "./facedetect.js";

const NUM_FRAMES = 16;

const screener = new DeepfakeScreener();
let faceDetector = null;

const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const statusCard = $("statusCard");
const statusText = $("statusText");
const progressFill = $("progressFill");
const analysisCard = $("analysisCard");
const videoEl = $("videoEl");
const verdictEl = $("verdict");
const scoreValue = $("scoreValue");
const scoreFill = $("scoreFill");
const detailRows = $("detailRows");
const framesStrip = $("framesStrip");
const againBtn = $("againBtn");

function setStatus(text, frac) {
  statusCard.hidden = false;
  statusText.textContent = text;
  if (frac !== undefined) progressFill.style.width = `${Math.round(frac * 100)}%`;
}

function hideStatus() {
  statusCard.hidden = true;
}

// ---------- upload wiring ----------
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("video/")) handleFile(f);
});
againBtn.addEventListener("click", () => {
  analysisCard.hidden = true;
  againBtn.hidden = true;
  framesStrip.innerHTML = "";
  fileInput.value = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ---------- core pipeline ----------
async function handleFile(file) {
  try {
    setStatus("Loading model…", 0.05);
    await screener.init((msg, f) => setStatus(msg, f));
    if (!faceDetector) {
      faceDetector = await getFaceDetector((msg, f) => setStatus(msg, f));
    }
    hideStatus();

    const url = URL.createObjectURL(file);
    videoEl.src = url;
    await new Promise((res, rej) => {
      videoEl.onloadedmetadata = res;
      videoEl.onerror = () => rej(new Error("Could not decode this video in the browser."));
    });

    analysisCard.hidden = false;
    verdictEl.textContent = "Analyzing…";
    verdictEl.className = "verdict uncertain";
    scoreValue.textContent = "…";
    scoreFill.style.width = "0%";
    detailRows.innerHTML = "";
    framesStrip.innerHTML = "";
    againBtn.hidden = true;

    setStatus("Sampling frames…", 0.1);
    statusCard.hidden = false;
    const frames = await sampleFrames(videoEl, NUM_FRAMES);
    if (frames.length === 0) throw new Error("No decodable frames found in this video.");

    setStatus("Detecting faces…", 0.35);
    const crops = [];
    const cropMeta = [];
    for (let i = 0; i < frames.length; i++) {
      const crop = cropLargestFace(frames[i].canvas, faceDetector, 0.35);
      if (crop) {
        crops.push(crop);
        cropMeta.push({ index: frames[i].index, time: frames[i].time, canvas: crop });
      }
      setStatus(`Detecting faces… ${i + 1}/${frames.length}`, 0.35 + 0.25 * ((i + 1) / frames.length));
    }
    if (crops.length === 0) {
      throw new Error("No face found in any sampled frame. Try a video with a clear, visible face.");
    }

    setStatus("Scoring face crops…", 0.65);
    const scores = [];
    for (let i = 0; i < crops.length; i++) {
      const s = await screener.scoreCrop(crops[i]);
      scores.push(s);
      setStatus(`Scoring face crops… ${i + 1}/${crops.length}`, 0.65 + 0.3 * ((i + 1) / crops.length));
    }

    const videoScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    renderResult(videoScore, scores, cropMeta, frames.length);
    hideStatus();
  } catch (err) {
    hideStatus();
    verdictEl.textContent = "Error";
    verdictEl.className = "verdict uncertain";
    scoreValue.textContent = "—";
    detailRows.innerHTML = `<div class="row"><span>${escapeHtml(err.message || String(err))}</span></div>`;
    analysisCard.hidden = false;
    againBtn.hidden = false;
  }
}

async function sampleFrames(video, n) {
  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) return [];
  const W = Math.min(video.videoWidth, 480);
  const scale = W / video.videoWidth;
  const H = Math.round(video.videoHeight * scale);

  const times = [];
  for (let i = 0; i < n; i++) {
    // avoid the very first/last instants (often black/transition frames)
    times.push(((i + 0.5) / n) * duration);
  }

  const frames = [];
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  for (let i = 0; i < times.length; i++) {
    const t = times[i];
    try {
      await seekTo(video, t);
      ctx.drawImage(video, 0, 0, W, H);
      const snap = document.createElement("canvas");
      snap.width = W;
      snap.height = H;
      snap.getContext("2d").drawImage(canvas, 0, 0);
      frames.push({ canvas: snap, time: t, index: i });
    } catch {
      // skip undecodable seek positions
    }
  }
  return frames;
}

function seekTo(video, t) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error("seek timeout")), 4000);
    const onSeeked = () => {
      clearTimeout(to);
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Math.min(t, Math.max(0, video.duration - 0.05));
  });
}

function renderResult(videoScore, scores, cropMeta, totalFrames) {
  const pct = (videoScore * 100).toFixed(1);
  scoreValue.textContent = `${pct}%`;
  scoreFill.style.width = `${Math.max(2, videoScore * 100)}%`;

  let verdict, cls, emoji;
  if (videoScore >= 0.75) { verdict = "LIKELY FAKE"; cls = "fake"; emoji = "🚨"; }
  else if (videoScore >= 0.5) { verdict = "SUSPICIOUS"; cls = "suspicious"; emoji = "⚠️"; }
  else if (videoScore >= 0.25) { verdict = "UNCERTAIN"; cls = "uncertain"; emoji = "❓"; }
  else { verdict = "LIKELY REAL"; cls = "real"; emoji = "✅"; }
  verdictEl.textContent = `${emoji} ${verdict}`;
  verdictEl.className = `verdict ${cls}`;

  const hi = scores.filter((s) => s >= 0.5).length;
  detailRows.innerHTML = `
    <div class="row"><span>Frames sampled</span><b>${totalFrames}</b></div>
    <div class="row"><span>Faces scored</span><b>${scores.length}</b></div>
    <div class="row"><span>Frames ≥ 0.5 fake</span><b>${hi} / ${scores.length}</b></div>
    <div class="row"><span>Min / max frame score</span><b>${Math.min(...scores).toFixed(3)} / ${Math.max(...scores).toFixed(3)}</b></div>
    <div class="row"><span>Aggregation</span><b>mean-pool (best in benchmark)</b></div>
    <div class="row"><span>Model</span><b>EfficientNet-B0 · fp16 · on-device</b></div>
  `;

  // per-frame chips
  framesStrip.innerHTML = "";
  cropMeta.forEach((m, i) => {
    const chip = document.createElement("div");
    chip.className = `frame-chip ${scores[i] >= 0.5 ? "hi" : "lo"}`;
    const thumb = document.createElement("canvas");
    thumb.width = 72;
    thumb.height = 72;
    thumb.getContext("2d").drawImage(m.canvas, 0, 0, 72, 72);
    const lab = document.createElement("div");
    lab.className = "fscore";
    lab.textContent = `${m.time.toFixed(1)}s · ${scores[i].toFixed(2)}`;
    chip.appendChild(thumb);
    chip.appendChild(lab);
    framesStrip.appendChild(chip);
  });

  againBtn.hidden = false;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
