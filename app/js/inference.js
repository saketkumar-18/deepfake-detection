/**
 * inference.js — on-device ONNX Runtime inference for deepfake detection.
 *
 * The model never leaves the device: it is fetched once, cached by the browser,
 * and run entirely in WASM. No video or frames are ever transmitted.
 */

export class DeepfakeScreener {
  constructor() {
    this.session = null;
    this.ort = null;
    this.meta = null;
    this.ready = false;
  }

  async init(onProgress) {
    if (this.ready) return;
    const meta = await fetch("assets/meta.json").then((r) => r.json());
    onProgress && onProgress("loading runtime", 0.15);
    // ort.bundle.min.mjs is a self-contained ES module (WASM inlined as base64),
    // so dynamic import() gives real named exports and needs no separate .wasm fetch.
    const ort = await import(
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.bundle.min.mjs"
    );
    if (!ort || !ort.InferenceSession) {
      throw new Error("onnxruntime-web failed to load");
    }
    this.meta = meta;
    this.ort = ort;

    // Prefer WASM; threads only when cross-origin isolation is available.
    const crossIsolated =
      typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
    ort.env.wasm.numThreads = crossIsolated ? 4 : 1;
    ort.env.wasm.simd = true;

    onProgress && onProgress("loading model (fp16, ~8 MB)", 0.45);
    this.session = await ort.InferenceSession.create("assets/model.fp16.onnx", {
      executionProviders: ["wasm"],
    });
    this.ready = true;
    onProgress && onProgress("ready", 1.0);
  }

  /**
   * Score one face crop.
   * @param {HTMLCanvasElement} cropCanvas - canvas containing the face crop
   * @returns {Promise<number>} fake probability in [0, 1]
   */
  async scoreCrop(cropCanvas) {
    if (!this.ready) throw new Error("model not initialized");
    const size = this.meta.img_size;
    const { mean, std } = this.meta.normalization;

    // Draw crop to a size×size canvas, read pixels, normalize to CHW float32.
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(cropCanvas, 0, 0, size, size);
    const { data } = ctx.getImageData(0, 0, size, size);

    const n = size * size;
    const input = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      input[i] = (r - mean[0]) / std[0];
      input[n + i] = (g - mean[1]) / std[1];
      input[2 * n + i] = (b - mean[2]) / std[2];
    }

    const tensor = new this.ort.Tensor("float32", input, [1, 3, size, size]);
    const feeds = { [this.meta.input.name]: tensor };
    const out = await this.session.run(feeds);
    const prob = out[this.meta.output.name].data[0];
    return Number(prob);
  }

  /**
   * Score a batch of crops sequentially (WASM single-thread friendly).
   * @param {HTMLCanvasElement[]} crops
   * @returns {Promise<number[]>}
   */
  async scoreBatch(crops) {
    const scores = [];
    for (const crop of crops) {
      scores.push(await this.scoreCrop(crop));
    }
    return scores;
  }
}
