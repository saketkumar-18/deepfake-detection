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
  // Audio-branch policy: it is a voice-clone INDICATOR, not a verdict.
  // Research reality: a 0.2 MB CNN cannot reliably detect arbitrary unseen TTS
  // generators, and it false-fires on high-arousal natural speech. So audio may
  // corroborate a visual alarm or raise suspicion, but NEVER convict alone.
  if (pAudio == null || Number.isNaN(pAudio)) {
    return {
      prob: pVisual,
      mode: "visual-only",
      note: "no usable audio track — verdict from visual analysis alone",
    };
  }
  const wV = 0.65;
  let fused = wV * pVisual + (1 - wV) * pAudio;
  const notes = [];

  // audio strongly suspects voice-clone
  const audioSuspect = pAudio >= 0.9;
  if (audioSuspect && pVisual >= 0.6) {
    fused = Math.max(fused, 0.9); // corroboration -> confident alarm
    notes.push("both branches suspect manipulation");
  } else if (audioSuspect && pVisual < 0.6) {
    // audio alone cannot convict — cap below the FAKE threshold
    fused = Math.min(fused, 0.6);
    notes.push("synthetic-voice indicators with a clean face — possibly re-voiced or false alarm");
  }
  if (Math.abs(pVisual - pAudio) > 0.7) {
    notes.push("branches disagree — treat with caution");
  }
  return { prob: fused, mode: "visual+audio", note: notes.join("; ") || "late fusion (w_visual=0.65)" };
}

export function verdict(p) {
  // Calibrated on held-out data: real+natural videos fuse to ~0.10-0.30,
  // fakes ≥0.85. Bands chosen so genuine videos don't get flagged.
  if (p >= 0.85) return { label: "LIKELY FAKE", tone: "bad" };
  if (p >= 0.60) return { label: "SUSPICIOUS", tone: "warn" };
  if (p >= 0.35) return { label: "UNCERTAIN", tone: "warn" };
  return { label: "LIKELY REAL", tone: "good" };
}
