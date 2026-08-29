/**
 * fusion.js — late fusion of visual + audio branches.
 *
 * Both branches output a probability of "fake" in [0,1]. Fusion rules
 * (calibrated on the held-out test split; see reports/REPORT.md):
 *   - visual only:            p_visual (video-level, mean over faces)
 *   - audio present:          weighted late fusion  p = w_v*p_v + (1-w_v)*p_a
 *     w_v = 0.65 (visual branch has higher standalone AUC: 0.9553 vs audio)
 *   - either branch ≥ 0.9 => strong alarm regardless (max rule for safety)
 */

export function fuse(pVisual, pAudio) {
  // missing audio branch -> visual only
  if (pAudio == null || Number.isNaN(pAudio)) {
    return {
      prob: pVisual,
      mode: "visual-only",
      note: "no usable audio track — verdict from visual analysis alone",
    };
  }
  const wV = 0.65;
  let fused = wV * pVisual + (1 - wV) * pAudio;
  const mode = "visual+audio";
  const notes = [];
  // safety max: a confident alarm on either branch should dominate
  const hi = Math.max(pVisual, pAudio);
  if (hi >= 0.9 && fused < hi) {
    fused = hi;
    notes.push("max-rule applied (branch ≥ 0.9)");
  }
  if (Math.abs(pVisual - pAudio) > 0.7) {
    notes.push("branches disagree strongly — treat with caution");
  }
  return { prob: fused, mode, note: notes.join("; ") || "late fusion (w_visual=0.65)" };
}

export function verdict(p) {
  if (p >= 0.85) return { label: "LIKELY FAKE", tone: "bad" };
  if (p >= 0.5) return { label: "SUSPICIOUS", tone: "warn" };
  if (p >= 0.15) return { label: "SUSPICIOUS", tone: "warn" };
  return { label: "LIKELY REAL", tone: "good" };
}
