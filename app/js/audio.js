/**
 * audio.js — on-device audio forensics branch.
 *
 * Decodes the audio track of the uploaded video entirely in the browser
 * (AudioContext.decodeAudioData), resamples to 16 kHz mono via
 * OfflineAudioContext, tiles into 2 s segments, and runs each through the
 * trained raw-waveform 1D-CNN (ONNX, WASM). A clip-level synthetic-speech
 * probability is the mean over segments. No audio ever leaves the device.
 */

export class AudioForensics {
  constructor() {
    this.session = null;
    this.inputName = null;
    this.outputName = null;
    this.SR = 16000;
    this.SEG = 2.0;
  }

  async init(ort, onProgress) {
    if (this.session) return;
    onProgress && onProgress("loading audio model (~0.9 MB)", 0.55);
    this.session = await ort.InferenceSession.create("assets/model.audio.onnx", {
      executionProviders: ["wasm"],
    });
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
  }

  /** Decode a File/Blob's audio track to 16 kHz mono Float32Array. */
  async decode(blob) {
    const buf = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let decoded;
    try {
      decoded = await ctx.decodeAudioData(buf.slice(0));
    } finally {
      ctx.close();
    }
    if (!decoded) throw new Error("no decodable audio track");
    // resample to 16k mono
    if (decoded.sampleRate === this.SR) {
      const ch = decoded.getChannelData(0);
      return new Float32Array(ch);
    }
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * this.SR), this.SR);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  }

  /** Tile waveform into fixed 2 s segments; final partial segment is zero-padded. */
  segments(x) {
    const n = Math.floor(this.SR * this.SEG);
    const out = [];
    for (let i = 0; i + n <= x.length; i += n) out.push(x.subarray(i, i + n));
    const rem = x.length % n;
    if (rem > n * 0.25 && x.length > n) {
      // keep a padded final segment if it has meaningful content (>25% of a segment)
      const tail = new Float32Array(n);
      tail.set(x.subarray(x.length - rem));
      out.push(tail);
    } else if (out.length === 0 && x.length > n * 0.25) {
      // whole clip shorter than one segment: pad once
      const seg = new Float32Array(n);
      seg.set(x);
      out.push(seg);
    }
    return out;
  }

  /**
   * Full audio analysis: decode -> segment -> score.
   * Includes a speech-domain guard: music/bass-heavy or non-speech content
   * (zero-crossing rate far outside speech range) makes the branch abstain —
   * the model only knows speech, and scoring non-speech would be dishonest.
   * Returns { prob, nSegments, ms } or null (no audio / not speech).
   */
  async analyze(ort, videoBlob, onProgress) {
    await this.init(ort, onProgress);
    let x;
    try {
      x = await this.decode(videoBlob);
    } catch (e) {
      return null; // silent video — audio branch abstains
    }
    // speech-domain guard: ZCR of speech is ~0.02-0.4 at 16 kHz mono;
    // music/bass-dominant content sits < 0.015, pure noise > 0.45.
    let crossings = 0;
    for (let i = 1; i < x.length; i++) if ((x[i] >= 0) !== (x[i - 1] >= 0)) crossings++;
    const zcr = crossings / (x.length - 1);
    if (zcr < 0.015 || zcr > 0.45) {
      return { prob: null, nSegments: 0, msPerSeg: 0, abstained: true, zcr: zcr };
    }
    const segs = this.segments(x);
    if (!segs.length) return null;
    onProgress && onProgress(`scoring ${segs.length} audio segment(s)`, 0.85);

    const t0 = performance.now();
    const probs = [];
    const B = 8;
    for (let i = 0; i < segs.length; i += B) {
      const batch = segs.slice(i, i + B);
      const big = new ort.Tensor("float32", new Float32Array(batch.length * segs[0].length), [batch.length, 1, segs[0].length]);
      const view = big.data;
      for (let b = 0; b < batch.length; b++) view.set(batch[b], b * segs[0].length);
      const res = await this.session.run({ [this.inputName]: big });
      const out = res[this.outputName].data; // (B,1) calibrated fake probabilities
      for (let b = 0; b < batch.length; b++) probs.push(out[b]);
    }
    const ms = (performance.now() - t0) / segs.length;
    return { prob: probs.reduce((a, b) => a + b, 0) / probs.length, nSegments: segs.length, msPerSeg: ms };
  }
}
