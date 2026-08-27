/**
 * facedetect.js — on-device face localization via MediaPipe BlazeFace.
 * Also runs fully locally (WASM); used only to locate the crop region.
 */

let detectorPromise = null;

export async function getFaceDetector(onProgress) {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      onProgress && onProgress("loading face detector", 0.3);
      const vision = await import(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
      );
      const { FaceDetector, FilesetResolver } = vision;
      const fileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      const detector = await FaceDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.3,
      });
      return detector;
    })();
  }
  return detectorPromise;
}

/**
 * Find the largest face in a canvas frame and return a crop canvas
 * with margin, or null if no face is found.
 * @param {HTMLCanvasElement} frameCanvas
 * @param {object} detector - MediaPipe FaceDetector
 * @param {number} margin - relative margin around the face box (0.35)
 * @returns {HTMLCanvasElement|null}
 */
export function cropLargestFace(frameCanvas, detector, margin = 0.35) {
  const result = detector.detect(frameCanvas);
  const dets = result && result.detections;
  if (!dets || dets.length === 0) return null;

  // pick largest by bounding-box area
  let best = null;
  let bestArea = 0;
  for (const d of dets) {
    const bb = d.boundingBox;
    const area = bb.width * bb.height;
    if (area > bestArea) {
      bestArea = area;
      best = bb;
    }
  }
  if (!best) return null;

  const W = frameCanvas.width;
  const H = frameCanvas.height;
  const mx = best.width * margin;
  const my = best.height * margin;
  let x0 = Math.max(0, Math.round(best.originX - mx));
  let y0 = Math.max(0, Math.round(best.originY - my));
  let x1 = Math.min(W, Math.round(best.originX + best.width + mx));
  let y1 = Math.min(H, Math.round(best.originY + best.height + my));
  if (x1 - x0 < 24 || y1 - y0 < 24) return null;

  const crop = document.createElement("canvas");
  crop.width = x1 - x0;
  crop.height = y1 - y0;
  crop.getContext("2d").drawImage(frameCanvas, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0);
  return crop;
}
