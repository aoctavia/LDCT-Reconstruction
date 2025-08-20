# Colab Showcase Structure — LDCT Project

## 0) Title & Setup

**Title:** *Low-Dose CT Reconstruction with Diffusion Models*
**Setup:** import libraries, (optional) mount Drive, set global seed, print environment (Python, PyTorch, CUDA).

---

## 1) Exploratory Data Analysis (EDA)

**Goal:** Demonstrate you understand the data and imaging characteristics—not just “plug & play” ML.

**Dataset context (your setup):**

* Source: **OrganMNIST3D (MedMNIST v2/v3)** — volumes are **28×28×28**, grayscale, intensities scaled to **\[0,1]** (proxy HU).
* We **simulate low-dose (LD)** from the clean slice (GT): mild blur → Poisson noise (photon statistics) → Gaussian readout noise → occasional down–up sampling.
* **No DICOM metadata** (e.g., **KVP, Exposure, ConvolutionKernel, pixel spacing, slice thickness**). Treat any “HU” work as **intensity proxies**; physical-dose analysis is out of scope for this dataset.

**Dataset overview (what you actually check):**

* Number of volumes and the **number of extracted 2D slices** (based on your sampling policy).
* Slice selection strategy (center-biased + random) and augmentation toggle.
* Note: per-patient IDs and clinical attributes are **not provided** in OrganMNIST3D.

**Visual samples (you implemented):**

* Random **LD vs GT** slice with windowing on \[0,1].
* **Error map** (LD−GT) to show spatial noise structure.

**Statistical analysis (you implemented):**

* **Intensity histogram** (proxy HU) for GT (and optionally LD) to see distribution shifts.
* **Noise estimate**: standard deviation in a homogeneous **ROI** (e.g., circular mask)—compare **LD vs GT** (LD should have higher std).
* **Line profile** across a fixed row to visualize edge degradation and noise texture.

**Pairing check (adapted to your pipeline):**

* Because LD is simulated **from** GT at a given `(volume_id, slice_id)`, pairing is guaranteed by construction.
* You validate pairing via index bookkeeping (e.g., asserting `(vi, si)` consistency), rather than DICOM tags like `InstanceNumber` or `ImagePositionPatient`.

**Summary (expected insights with this dataset):**

* **LD is noisier** than GT (↑ROI std, more high-frequency fluctuation in line profiles).
* **No claims** about slice thickness or pixel spacing (not available).
* This is a **proxy study** for LDCT: great for methods benchmarking, but **not** for radiation-dose physics.

> 📌 This EDA shows you understand the imaging problem and the dataset’s limitations (no metadata, tiny 28×28 slices), framing your later results correctly.

---

## 2) Baseline Models (Supervised DL)

**Goal:** Prove you can implement standard restoration baselines and benchmark them.

* **Models:** UNet (2D image-to-image), **DnCNN** (denoising CNN), **SwinIR-tiny** (transformer for restoration).
* **Training:** **L1 + SSIM** loss, **AdamW**, **early stopping**.
* **Metrics:** **PSNR, SSIM** on validation.
* **Visualization:** **LD vs UNet vs GT** (plus optional error maps).

> 📌 Shows solid DL engineering + fair benchmarking before jumping to advanced generative methods.

---

## 3) Advanced Model: Diffusion (Highlight AI/ML)

**Goal:** Showcase advanced generative modeling skills.

* **Conditional Diffusion (DDPM + UNet backbone):**
  **Input:** `LD` + **timestep** `t`.
  **Output:** predicted **noise** `ε`; iterative sampling to reconstruct `x₀`.
* **Training:** **MSE noise loss**, **EMA**, **T = 1000** diffusion steps; inference with **250** sampling steps (default).
* **Ablations:**

  * **Timesteps:** 1000 vs 250 vs 50.
  * **Loss:** **MSE** vs **MSE + SSIM**.
* **Sampling demo:** visual snapshots (e.g., step **10 → 250 → final**).
* **Comparison table:** Baselines vs Diffusion → **PSNR, SSIM, LPIPS**.
  Include **runtime** and **parameter count** (efficiency awareness).

> 📌 Clear highlight that you can implement **SOTA generative** methods and analyze their behavior in medical imaging.

---

## 4) Evaluation & Visualization

* **Side-by-side figure:** **LD | UNet | SwinIR | Diffusion | GT**.
* **Error maps:** `|pred − GT|` for each model.
* **Histogram (proxy HU):** compare output intensity distributions vs GT.
* **Quantitative table (test set):** **PSNR / SSIM / LPIPS**, with params & runtimes.
* **Discussion:**

  * **Diffusion** preserves fine detail/edges better (lower structured residuals; histogram closer to GT).
  * **Baselines** (especially CNNs) often look **smoother/blurrier**—higher bias, lower variance.
  * Recognize **trade-offs**: diffusion wins on fidelity but costs sampling time; CNNs win on speed.

---

## 5) Conclusion & Future Work

* **Conclusion:** Diffusion **outperforms** baselines on **PSNR/SSIM** (and typically LPIPS), reduces noise while **preserving anatomical detail** better than UNet/DnCNN/SwinIR in this proxy setting.
* **Future work:**

  * Extend to **3D** (volumetric UNet / 3D DDPM).
  * **Multimodal conditioning** with **dose/physics metadata** (requires clinical DICOM).
  * **Clinical validation** on real LD/SD datasets; evaluate diagnostic endpoints and reader studies.

---

## 6. (Bonus, kalau sempat) Demo Interaktif

* **Gradio demo**: upload LD slice → reconstruct dengan diffusion.
* Menunjukkan kamu bisa bikin **aplikasi AI medis usable**.

---

# 📊 Colabs Structures


| #  | Section / Cells                    | Purpose (What it does)                                                                                              | Key Outputs / Artifacts                     | Notes & Dependencies                                                    |           |                       |         |                  |                                                          |                                                     |
| -- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------- | --------- | --------------------- | ------- | ---------------- | -------------------------------------------------------- | --------------------------------------------------- |
| 0  | **Title & Setup** (0, 1a, 1b)      | Set notebook title, install deps, print env (Python, Torch, CUDA), set seeds.                                       | –                                           | Run first. Confirms versions & device.                                  |           |                       |         |                  |                                                          |                                                     |
| 1  | **Core Imports & Config** (2)      | Import libs, define device/seed helpers.                                                                            | –                                           | Used by all later cells.                                                |           |                       |         |                  |                                                          |                                                     |
| 2  | **Data & LD Simulator** (3)        | Load **OrganMNIST3D**; create paired `(LD, GT)` slices via blur + Poisson + Gaussian + (optional) down–up sampling. | –                                           | Produces LD–GT pairs; no DICOM metadata.                                |           |                       |         |                  |                                                          |                                                     |
| 3  | **EDA Utilities** (4)              | Visual helpers: LD/GT display, error map (LD–GT), intensity histogram, ROI-noise std, line profile.                 | Inline plots                                | Uses dataset from (3).                                                  |           |                       |         |                  |                                                          |                                                     |
| 4  | **Metrics** (5)                    | Implement **SSIM** & **PSNR** (for training & eval).                                                                | –                                           | Reused across baselines & diffusion.                                    |           |                       |         |                  |                                                          |                                                     |
| 5  | **Models** (6)                     | Define **UNet**, **DnCNN**, **SwinIR-tiny**.                                                                        | –                                           | 28×28 single-channel friendly.                                          |           |                       |         |                  |                                                          |                                                     |
| 6  | **Loss & Early Stop** (7)          | **L1+SSIM** loss; early stopping utility.                                                                           | –                                           | For baselines training.                                                 |           |                       |         |                  |                                                          |                                                     |
| 7  | **Train/Validate Loop** (8)        | Generic training with AdamW (amp optional), validation PSNR/SSIM, curves, sample visualization.                     | `runs/<model-*>/logs.json`, inline curves   | Saves **best\_ckpt.pt** on val SSIM.                                    |           |                       |         |                  |                                                          |                                                     |
| 8  | **EDA Run** (9)                    | Execute EDA: LD vs GT, error map, histogram, ROI noise, line profile.                                               | Inline plots                                | Shows LD noisier than GT (proxy-HU).                                    |           |                       |         |                  |                                                          |                                                     |
| 9  | **Train Baselines** (10)           | Train **UNet/DnCNN/SwinIR** (quick epochs).                                                                         | run dirs per model                          | Needed for comparisons if no ckpt exists.                               |           |                       |         |                  |                                                          |                                                     |
| 10 | **Compare Baselines** (11)         | Summarize last-epoch val PSNR/SSIM; optional viz hooks.                                                             | pandas table                                | Uses histories from (10).                                               |           |                       |         |                  |                                                          |                                                     |
| 11 | **Diffusion Model** (12)           | **Conditional DDPM-UNet** (input: `x_t`, `LD`, `t`; output: **ε**), FiLM + sinusoidal t-emb.                        | –                                           | Core model for diffusion.                                               |           |                       |         |                  |                                                          |                                                     |
| 12 | **DDPM Schedules & EMA** (13)      | Linear/cosine β schedule, q/p sampling helpers, **EMA** weights, sampling recorder.                                 | –                                           | Used by training & demo.                                                |           |                       |         |                  |                                                          |                                                     |
| 13 | **Train Diffusion** (14)           | Train with **MSE noise loss** (optional +SSIM), keep **EMA**, quick val PSNR/SSIM.                                  | `runs_diff/.../ddpm_ema.pt`, logs           | Produces EMA ckpt for inference.                                        |           |                       |         |                  |                                                          |                                                     |
| 14 | **Sampling Demo** (15)             | Sample **250 steps**; snapshot **step 10 → 250 → final**; grid visualization.                                       | `figs/diffusion_grid_demo.png`              | Requires EMA from (14).                                                 |           |                       |         |                  |                                                          |                                                     |
| 15 | **Ablations** (16)                 | Compare **sampling steps** (1000/250/50); optional loss ablation (MSE vs MSE+SSIM).                                 | printed table                               | Quick eval path.                                                        |           |                       |         |                  |                                                          |                                                     |
| 16 | **LPIPS + Comparison** (17)        | (Optional) **LPIPS** with safe resize→224; table: **Baselines vs Diffusion** (PSNR/SSIM/LPIPS + params/runtime).    | `figs/comparison_baseline_vs_diffusion.csv` | Downloads AlexNet weights once; graceful fallback if LPIPS unavailable. |           |                       |         |                  |                                                          |                                                     |
| 17 | **Save & Export** (18)             | Persist demo grid & comparison CSV.                                                                                 | PNG/CSV files                               | For paper/report.                                                       |           |                       |         |                  |                                                          |                                                     |
| 18 | **Side-by-side & Error Maps** (19) | \*\*LD                                                                                                              | UNet                                        | SwinIR                                                                  | Diffusion | GT\*\* montage + \*\* | pred−GT | \*\* error maps. | `figs/side_by_side_test.png`, `figs/error_maps_test.png` | Needs trained baselines (or ckpts) & diffusion EMA. |
| 19 | **Histogram (proxy HU)** (20)      | Compare distributions (UNet/SwinIR/Diffusion vs GT) on test subset.                                                 | `figs/hist_proxy_hu_test.png`               | Proxy-HU since MedMNIST is \[0,1].                                      |           |                       |         |                  |                                                          |                                                     |
| 20 | **Test-set Metrics Table** (21)    | Full test **PSNR/SSIM/LPIPS**, with **param counts**.                                                               | `figs/test_metrics_table.csv`               | Uses compute\_lpips\_batch if defined.                                  |           |                       |         |                  |                                                          |                                                     |
| 21 | **Discussion (Auto-notes)** (22)   | Auto-generate narrative: who’s best, histogram/error-map interpretation, params.                                    | Printed notes                               | Paste into report.                                                      |           |                       |         |                  |                                                          |                                                     |
| 22 | **Bootstrap CI** (23)              | **95% CI** for mean PSNR/SSIM via bootstrap (per-slice).                                                            | `figs/bootstrap_ci_psnr_ssim.csv`           | Strengthens statistical claims.                                         |           |                       |         |                  |                                                          |                                                     |

### Recommended run order (quick)

1. **0–1b** (setup) → **2–8** (data, EDA utils & metrics) → **9** (EDA run)
2. **10–11** (baselines train & compare)
3. **12–15** (diffusion train + demo) → **16–17–18** (ablations & comparison & export)
4. **19–22** (final eval & discussion) → **23** (bootstrap CI)


