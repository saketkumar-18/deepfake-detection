# 🎬 Video Script — Deepfake Detection (Term Project Submission)
**Target length: 10–12 minutes** | Face visible throughout | IITG title slide first

---

## SECTION 1 — Title Slide (~15 sec)
**On screen:** IIT Guwahati title slide template (filled in: name, student ID, course code, course name, credits, trimester)

> "Hello, this is my project video submission for [Course Name], submitted as part of the BSc Honours Data Science and AI programme at IIT Guwahati."

---

## SECTION 2 — Introduce Yourself (~30 sec)
**On screen:** Your face (camera view)

> "Hi, my name is Saket Kumar, I'm a student in the BSc Honours Data Science and Artificial Intelligence online degree programme at IIT Guwahati, and today I'll be presenting my Term Project [1/2/3] — a deepfake video detection system."

---

## SECTION 3 — The Problem (~1.5 min)
**On screen:** A couple of example deepfake images/headlines (optional)

Key talking points:
- Deepfakes are AI-generated fake videos that look real — face swaps, lip syncs, expression transfers
- They're being used for misinformation, fraud, and impersonation — this is a real, growing problem
- The core question: **can we automatically tell if a video has been manipulated?**
- My research angle went further: *which manipulation artifacts actually generalize across different deepfake generators?* — because a detector that only works on one type of fake isn't very useful in the real world

> "What I found interesting is that most detectors work well on the type of fake they were trained on, but fail on new generators. So I wanted to study what makes some artifacts more generalizable than others."

---

## SECTION 4 — Tools & Technologies (~45 sec)
**On screen:** Quick list slide or just mention verbally

- **Python + PyTorch** — model training (CPU-only, no GPU needed)
- **EfficientNet-B0** — the frame-level detector (pretrained on ImageNet)
- **OpenCV** — face cropping with Haar cascades
- **NumPy / scikit-learn** — metrics (AUC, AP, accuracy)
- **ONNX Runtime Web + WebAssembly** — to run the model entirely in the browser
- **Vercel** — free deployment of the live demo

> "Everything was trained on CPU only, which kept it accessible — no GPU required."

---

## SECTION 5 — The Dataset (~1.5 min)
**On screen:** Show the dataset folder structure / a few sample frames

- **FaceForensics++ (C23)** — the primary dataset: real videos + fakes from 5 generators (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter)
- **Celeb-DF v2** — used as a held-out, out-of-distribution test set
- Pre-extracted face frames from a Kaggle mirror (~5,400+ test videos)
- **Important preprocessing:** face crops using Haar cascade with margin

**Integrity issues found (great for showing honest reflection):**
> "I found two data integrity problems. First, the Kaggle mirror's provided splits leaked 124 validation and 78 test videos into training — I rebuilt the splits to be video-level disjoint. Second, real clips had 12 frames but fakes only had 2–3, so a model could cheat by just counting frames. I added a length-control protocol that subsamples every clip to exactly T frames. These fixes matter because without them, the metrics would be artificially inflated."

---

## SECTION 6 — Code Walkthrough (~2.5 min)
**On screen:** Share your screen, open the repo. Show these files in order:

### 6a. Architecture (show the diagram from README)
```
video → 16 frames → face crop → EfficientNet-B0 (frame scores)
                                      ↓
                          Temporal Transformer → video verdict
```
> "The pipeline has two stages. First, a spatial detector — EfficientNet-B0 — scores each frame individually. Then a temporal transformer aggregates frame embeddings into a single video-level verdict."

### 6b. `src/deepfake_detection/models/spatial.py`
- EfficientNet-B0 with ImageNet initialization
- Replace the classification head for binary (real/fake)
- Show the forward pass briefly

### 6c. `src/deepfake_detection/train_spatial.py`
- Head-only training first (AUC 0.79), then full fine-tune (0.875 → 0.91)
- Balanced sampling to handle class imbalance

### 6d. `src/deepfake_detection/models/temporal.py`
- 3-layer, 4-head transformer over frame embeddings
- CLS token + attention pooling → video probability

### 6e. `src/deepfake_detection/analyze_artifacts.py` (the research angle)
- FFT radial power spectrum — measures spectral fingerprints
- Grad-CAM — shows where the model is looking

> "I don't need to explain every line — the key decision was using a pretrained backbone and fine-tuning, because training from scratch on CPU with this dataset size would underfit."

---

## SECTION 7 — Outputs & Results (~2 min)
**On screen:** Show benchmark table + charts from `reports/`

### Main results (video-level, held-out test):
| Model | AUC | Accuracy |
|---|---|---|
| Spatial (EfficientNet-B0), mean-pool | **0.9553** | 91.7% |
| Spatial + Temporal Transformer | 0.9456 | 88.1% |

> "The spatial detector with simple mean-pooling actually beat the temporal transformer — 0.955 AUC versus 0.946. That was a surprising finding: for this dataset, averaging frame scores is enough, and the added complexity of the transformer didn't help."

### Cross-generator generalization:
- Celeb-DF (unseen generator family): AUC 0.922
- FaceForensics++: AUC 0.893
- Overall: 0.903

### Artifact analysis (the interesting part):
- **FFT:** FF++ fakes deviate 2.3× more from real in the frequency spectrum than Celeb-DF fakes — driven by high-frequency bands (the GAN upsampling fingerprint)
- **Grad-CAM:** On fakes, attention concentrates on the face interior; on real frames, it scatters — the model is looking at manipulated content, not compression artifacts

### Live demo:
> "I also deployed a fully in-browser demo — the model runs as ONNX in WebAssembly, so no video ever leaves the user's device. It's live at deepfake-detector-amber.vercel.app."

(Show the live demo briefly if time allows — very impressive on screen)

---

## SECTION 8 — Evaluation Metrics (~1 min)
**On screen:** Metrics table

- **AUC (Area Under ROC Curve)** — primary metric; measures ranking quality across all thresholds. 0.955 means the model correctly ranks a random fake above a random real video 95.5% of the time
- **AP (Average Precision)** — 0.984; precision-recall area, important when classes are imbalanced
- **Accuracy** — 91.7% at the chosen threshold
- **Why AUC over accuracy:** accuracy depends on threshold choice; AUC is threshold-independent and more robust for forensic applications

> "I also looked at per-generator AUC to see where the model struggles — FaceSwap was the hardest at 0.56 AUC in the cross-preprocessing setting, because swap-based forgeries leave weaker spectral cues."

---

## SECTION 9 — Wrap-Up (~1 min)
**On screen:** Your face

Key points:
- **What was achieved:** A 0.955 AUC deepfake detector trained entirely on CPU, with a research study on which artifacts generalize across generators
- **Key finding:** Artifacts generalize well within a preprocessing pipeline (AUC ~0.90–0.96) but degrade sharply across pipelines (0.56–0.80) — this is the honest limitation
- **What I learned:** Data integrity matters enormously — the two leakage issues I found would have inflated all metrics; also, simpler aggregation (mean-pool) beat the fancier transformer
- **Future work:** Train on more diverse generators, explore video-level augmentation, and test against adversarial attacks
- **Ethics note:** Detection scores are forensic evidence cues, not legal proof

> "The biggest takeaway for me was that a detector's real-world value isn't its accuracy on its own test set — it's how well it generalizes to fakes it has never seen. And that's still an open problem."

---

## 📝 Pre-Recording Checklist
- [ ] Fill in IITG title slide (name, ID, course code, course name, credits, trimester)
- [ ] Set the title slide as YouTube thumbnail too
- [ ] Test microphone
- [ ] Increase code editor font size (Ctrl +)
- [ ] Have these ready on screen: repo, benchmark table, architecture diagram, live demo tab
- [ ] Record with face visible the entire time (Zoom: screen share + camera)
