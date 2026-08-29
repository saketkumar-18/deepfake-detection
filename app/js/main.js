/**
 * main.js — orchestrates upload → frame sampling → face detection → scoring → verdict.
 * Plus: audio-track decoding + synthetic-speech scoring, fused with the visual branch.
 * Everything runs on-device; no network calls except model/CDN fetches.
 */
import { DeepfakeScreener } from "./inference.js";
import { getFaceDetector, cropLargestFace } from "./facedetect.js";
import { AudioForensics } from "./audio.js";
import { fuse, verdict as verdictOf } from "./fusion.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.bundle.min.mjs";

const NUM_FRAMES = 16;

const screener = new DeepfakeScreener();
const audioForensics = new AudioForensics();
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

// ---------- built-in samples ----------
const SAMPLES = [
  { file: "samples/sample_real_video_natural_voice.mp4", label: "✅ Real face · real voice" },
  { file: "samples/sample_fake_video_natural_voice.mp4", label: "🚨 Fake face · real voice" },
  { file: "samples/sample_real_video_synthetic_voice.mp4", label: "⚠️ Real face · synthetic voice" },
  { file: "samples/sample_fake_video_synthetic_voice.mp4", label: "🚨 Fake face · synthetic voice" },
];
async function loadSample(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`sample not found (${res.status})`);
    const blob = await res.blob();
    const name = url.split("/").pop();
    const file = new File([blob], name, { type: blob.type || "video/mp4" });
    handleFile(file);
  } catch (e) {
    setStatus(`Could not load sample: ${e.message}`, 0);
  }
}
SAMPLES.forEach((s) => {
  const btn = document.createElement("button");
  btn.className = "sample-btn";
  btn.textContent = s.label;
  btn.addEventListener("click", () => loadSample(s.file));
  $("sampleBtns").appendChild(btn);
});

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
      setStatus(`Scoring face crops… ${i + 1}/${crops.length}`, 0.65 + 0.2 * ((i + 1) / crops.length));
    }

    const videoScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    // ---- audio branch: decode the video's own audio track & score it ----
    setStatus("Analyzing audio track…", 0.9);
    let audioResult = null;
    try {
      audioResult = await audioForensics.analyze(ort, file, (msg, f) => setStatus(msg, f));
    } catch (e) {
      audioResult = null; // audio is best-effort; never block the visual verdict
    }

    const fused = fuse(videoScore, audioResult && !audioResult.abstained ? audioResult.prob : null);
    renderResult(fused, videoScore, audioResult, scores, cropMeta, frames.length);
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

function renderResult(fused, videoScore, audioResult, scores, cropMeta, totalFrames) {
  const videoScorePct = (fused.prob * 100).toFixed(1);
  scoreValue.textContent = `${videoScorePct}%`;
  scoreFill.style.width = `${Math.max(2, fused.prob * 100)}%`;

  const v = verdictOf(fused.prob);
  const emoji = { bad: "🚨", warn: "⚠️", good: "✅" }[v.tone];
  const cls = { bad: "fake", warn: "suspicious", good: "real" }[v.tone];
  verdictEl.textContent = `${emoji} ${v.label}`;
  verdictEl.className = `verdict ${cls}`;

  const hi = scores.filter((s) => s >= 0.5).length;
  const audioRow = audioResult
    ? audioResult.abstained
      ? `<div class="row"><span>Audio: synthetic-voice score</span><b>abstained — non-speech content detected</b></div>`
      : `<div class="row"><span>Audio: synthetic-voice score</span><b>${(audioResult.prob * 100).toFixed(1)}% (${audioResult.nSegments} segment${audioResult.nSegments > 1 ? "s" : ""}, ${audioResult.msPerSeg.toFixed(0)} ms/seg)</b></div>`
    : `<div class="row"><span>Audio: synthetic-voice score</span><b>no audio track</b></div>`;
  const visRow = `<div class="row"><span>Visual: face-forgery score</span><b>${(videoScore * 100).toFixed(1)}%</b></div>`;
  detailRows.innerHTML = `
    ${visRow}
    ${audioRow}
    <div class="row"><span>Fusion</span><b>${fused.mode}${fused.note ? " · " + fused.note : ""}</b></div>
    <div class="row"><span>Frames sampled</span><b>${totalFrames}</b></div>
    <div class="row"><span>Faces scored</span><b>${scores.length}</b></div>
    <div class="row"><span>Frames ≥ 0.5 fake</span><b>${hi} / ${scores.length}</b></div>
    <div class="row"><span>Min / max frame score</span><b>${Math.min(...scores).toFixed(3)} / ${Math.max(...scores).toFixed(3)}</b></div>
    <div class="row"><span>Aggregation</span><b>mean-pool (best in benchmark)</b></div>
    <div class="row"><span>Models</span><b>EfficientNet-B0 (visual) · 1D-CNN (audio) · on-device</b></div>
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
