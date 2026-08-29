# EXACT WORD-FOR-WORD NARRATION — Deepfake Detection Video
(Speak naturally, pause anywhere, don't re-record small stumbles. ~11 minutes.)
Placeholders in [BRACKETS] = fill in your course details before recording.

---

## 1. TITLE SLIDE (screen: IITG title slide, ~20 sec)

"Hello everyone. This is my video submission for [COURSE CODE — COURSE NAME], submitted as part of the BSc Honours Data Science and Artificial Intelligence online degree programme at IIT Guwahati. My name is Saket Kumar, roll number 23035010051, and this is my [INTERNSHIP / TERM PROJECT 9]. My project is Deepfake Video Detection using a Spatial-Temporal CNN and Transformer."

## 2. INTRODUCE YOURSELF (screen: your face, ~30 sec)

"So a little about me first — I'm Saket, a student in the BSc Data Science and AI programme at IIT Guwahati. For this project, I built an end-to-end deepfake detection system — a model that can look at a video of a face and tell you whether it's real or manipulated. I trained it entirely on CPU, studied how well it generalizes to fake videos it has never seen, and deployed it as a privacy-preserving web demo where nothing gets uploaded to any server. In the next ten minutes or so, I'll walk you through the problem, the data, the code, the results, and what I learned from it."

## 3. THE PROBLEM (screen: face + slides, ~90 sec)

"So let me start with the problem. Deepfakes are AI-generated videos where a person's face is swapped or their expressions are re-enacted by a generative model. These are being used for misinformation, financial fraud, and impersonation — and they're getting harder to spot with the human eye.

The task I set out is binary video-level classification: given a video, output — is this genuine footage, or has it been manipulated?

But I didn't want to just build a classifier. There's a deeper research question I wanted to answer, and it's this — which manipulation artifacts actually generalize across different deepfake generators? Because here's the issue: most detectors are trained on one dataset, and they learn dataset-specific quirks — compression, cropping, one generator's fingerprint. When they meet a new generator they've never seen, accuracy collapses. A detector that only works on one type of fake isn't really useful in the real world. So my project answers both: how accurate can a detector be, and which of the cues it uses actually transfer?"

## 4. TOOLS & TECHNOLOGIES (screen: quick list slide or repo README, ~45 sec)

"Quickly, the tools I used. Python and PyTorch for everything — training and evaluation. The backbone model is EfficientNet-B0, pre-trained on ImageNet. OpenCV for face detection and cropping. NumPy and scikit-learn for metrics — AUC, average precision, accuracy. For the deployment side, I exported the model to ONNX and ran it in the browser with onnxruntime-web in WebAssembly, with MediaPipe BlazeFace for face detection client-side. The whole thing is deployed on Vercel. And one thing worth mentioning — everything was trained on CPU only. No GPU at any point."

## 5. THE DATASET (screen: show data folders / dataset-metadata.json, ~90 sec)

"Now the dataset. The primary corpus is FaceForensics++, the C23 compressed version. It has real videos plus fakes from five generators — Deepfakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter. On top of that, I used Celeb-DF v2 as a completely held-out test — a different generator family the model never trains on, which is what makes the generalization test honest. I used a Kaggle mirror with pre-extracted face frames, and I cropped faces with a Haar cascade with a margin around the bounding box. The held-out test set is about twelve hundred ninety videos, and the generalization set is around seventy thousand frames across sources.

Now — two data problems I found, and I want to be upfront about them because they changed everything. First, the Kaggle mirror's official train/validation/test split files leaked videos — 124 validation and 78 test videos also appeared in training. So any model trained on those splits would have inflated metrics. I rebuilt the splits myself, enforcing video-level disjoint — no video or identity appears in two splits. I dropped about a thousand and forty-four leaked rows.

Second, in this mirror, real clips have around 12 frames but fake clips only have 2 to 3. That means a temporal model could cheat — it could get a perfect score just by counting frames, without ever looking at the face. So I added a length-control protocol: every clip gets subsampled to exactly T frames, shorter clips are dropped. All my temporal results use T equals 2, so the shortcut is gone.

These two fixes are why I trust my numbers — and honestly, finding them taught me more than the modeling itself."

## 6. CODE WALKTHROUGH (screen: share screen, open the repo, ~150 sec)

"Let me open the code. This is the GitHub repository — saketkumar-18 slash deepfake-detection. It's a single Python package.

So first, here's the overall pipeline — [open the README architecture diagram] a video comes in, I sample 16 evenly-spaced frames, crop the face from each one, and pass every crop through EfficientNet-B0, which gives me a per-frame fake probability. Then there are two ways to get to a video-level verdict: mean-pool the frame scores, or feed the frame embeddings into a temporal transformer. Let me show you both.

[open src/deepfake_detection/models/spatial.py] This is the spatial detector — EfficientNet-B0 with ImageNet initialization, and I replaced the classification head with a single-logit binary head, sigmoid, binary cross-entropy. I trained it in two phases: first head-only — froze the backbone, which got a validation AUC of 0.79 — then full fine-tuning on a class-balanced subset, which took it to 0.875 and then 0.91 after balanced fine-tuning. The two-phase approach keeps CPU compute tractable and avoids catastrophic forgetting of the ImageNet features early on.

[open src/deepfake_detection/models/temporal.py] This is the temporal transformer. Frame embeddings — 1280 dimensions — get projected to 256, positional embeddings added, then three transformer layers with four heads, a CLS token, attention pooling, and a video-level probability. The key design decision here is the length control I mentioned — with T equals 2, every clip contributes exactly two frames, so the transformer can't learn 'short means fake'.

[open src/deepfake_detection/prepare_data.py briefly] This is where the leak-free splits are built — it's the file that fixed the leakage problem.

And the two analysis scripts — analyze_generalization gives per-generator AUC, and analyze_artifacts produces the FFT spectral fingerprints and Grad-CAM maps — that's the research part, I'll show you those results in a minute."

## 7. OUTPUTS & RESULTS (screen: reports/benchmark.md + figures, ~2 min)

"Alright, results. This is the main benchmark, on the held-out test split, video-level, with length control on.

[Spatial mean-pool row] The best configuration is the spatial detector with mean pooling — video AUC 0.9553, average precision 0.9838, accuracy 91.7 percent.

[Transformer row] The temporal transformer gets 0.9456 — slightly below the simple mean-pool. That surprised me — I expected the transformer to win. The explanation: under length control, each clip only has two frames, so there's very little sequential structure for the transformer to exploit — while averaging frame scores is a robust, variance-reducing estimator. It's a nice example of a simpler method beating a fancier one.

[Cross-source table] Cross-source generalization — this is one model, tested per source. On FaceForensics++ itself: 0.89 AUC. On Celeb-DF — a generator family it never saw: 0.92 AUC. Overall 0.90. So the model does transfer to an unseen generator family, which is the important result.

[Cross-preprocessing table] But here's the honest part. This table is the hard setting — test frames that came through a different face-crop pipeline than training. And performance drops a lot. Deepfakes — the autoencoder-based generator — stays detectable at 0.80. But the swap-based forgeries drop to 0.56 to 0.63. Overall 0.64. So the artifacts the model learned are partly tied to the preprocessing pipeline — that's the main negative finding of the project.

[Artifacts analysis] And the artifact analysis explains why. FF++ fakes deviate from real frames two point three times more in the frequency spectrum than Celeb-DF fakes — driven by the high-frequency band, which is the classic GAN decoder upsampling fingerprint. And Grad-CAM shows that on fakes, the model's attention concentrates inside the face — center of mass 0.32 and 0.22 — while on real frames it scatters, 0.09. So the model is genuinely looking at manipulated facial content, not at borders or compression — that's what a good detector should do.

[Switch to browser demo] And finally — the live demo. This is deployed at deepfake-detector-amber.vercel.app. I can drop a video here, and everything runs on my machine — BlazeFace finds the face, the EfficientNet detector runs as fp16 ONNX in WebAssembly, and no video ever leaves the device. That was a deliberate privacy decision for a forensic tool."

## 8. EVALUATION METRICS (screen: metrics table, ~60 sec)

"A quick word on the metrics, because the choice matters. The primary metric is AUC — area under the ROC curve. It's threshold-independent: it measures how well the model ranks a random fake above a random real, across all thresholds. A 0.9553 AUC means it ranks correctly about 95 percent of the time. Accuracy — 91.7 percent — depends on where you set the threshold, which is why I report both. Average precision, 0.98, is the precision-recall area, which matters when classes are imbalanced. And I report per-generator AUC rather than just overall, because an overall number hides exactly where the model struggles — FaceSwap at 0.56 tells a very different story than the 0.95 headline."

## 9. WRAP-UP (screen: your face, ~60 sec)

"So to wrap up. What did this project achieve? A deepfake detector with 0.955 video-level AUC on a leak-free, length-controlled test split, trained entirely on CPU. A generalization study — 0.90 AUC across sources including an unseen generator family. An artifact analysis identifying which cues transfer — high-frequency spectral fingerprints and face-interior attention. And a fully on-device browser deployment.

What did I take away? Three things. First — data integrity is everything. The two problems I found in the mirror would have made every metric a lie if I hadn't caught them. Second — the simplest aggregation beat the fancier model, and understanding why mattered more than the accuracy itself. And third — the honest limit: my cross-pipeline number, 0.64, says benchmark accuracy on your own pipeline overstates real-world robustness. That's the open problem.

Future work would be training across multiple crop pipelines jointly, preprocessing-invariant features, adversarial robustness, and longer clips for the temporal model.

One ethical note to end on — the model's output is a forensic evidence cue, not legal proof. It should support human judgment, not replace it.

Thank you for watching."

---

## PRE-RECORDING CHECKLIST
- [ ] Fill in [COURSE CODE — COURSE NAME] and project number in section 1
- [ ] IITG title slide filled in (name, ID, course, credits, trimester) = first frame
- [ ] Same slide = YouTube thumbnail
- [ ] Mic test + code editor font enlarged (Ctrl +)
- [ ] Tabs pre-opened: repo, benchmark.md, artifact figures, live demo
- [ ] Face visible the whole time (Zoom screen-share + camera)
- [ ] Don't read it — speak each block once, then move on. Stumbles are fine.
