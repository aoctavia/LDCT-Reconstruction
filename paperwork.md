# Low-Dose CT Reconstruction with Diffusion Models (mini paper)

## Abstract

We study low-dose CT (LDCT) denoising on **OrganMNIST3D** (MedMNIST) by creating paired 2D slices (GT) and **simulated LD** (blur + Poisson + Gaussian + occasional down/up-sampling). We benchmark **UNet**, **DnCNN**, and **SwinIR-tiny** against a **conditional DDPM (UNet backbone)**. In our setting, baselines achieve strong validation/test PSNR/SSIM, while a short-trained DDPM underperforms; we analyze why and quantify uncertainty with **bootstrap 95% CIs**. Qualitative side-by-sides, error maps, and intensity (proxy-HU) histograms support the quantitative results. (Validation ranking and test metrics; saved figures and tables reported in the Results.) &#x20;

---

## 1 Introduction

Reducing radiation dose increases CT noise and can obscure anatomy. Learning-based reconstruction seeks to restore diagnostic quality from LD inputs. We compare classical supervised baselines (UNet, DnCNN, SwinIR-tiny) to a **conditional diffusion** approach (DDPM) on a small, standardized proxy dataset, discussing **accuracy–efficiency** trade-offs (metrics, runtime, parameter counts).&#x20;

---

## 2 Data & EDA

**Dataset.** OrganMNIST3D volumes (28×28×28) with labels; we extract 2D slices and **simulate LD** from GT per (volume, slice) index, producing **1:1 paired** examples by construction. Pairing is verified via indices and basic statistics (MSE/PSNR). &#x20;

**What is (and isn’t) available.** MedMNIST does **not** contain DICOM metadata (pixel spacing, slice thickness, KVP/mAs, convolution kernel), so all intensity analyses are **proxy-HU**; dose-related correlations are out of scope here.&#x20;

**Key EDA findings.**

* Volumes in train split ≈ **971**; synthetic 1↔1 pairing between LD and GT slices. &#x20;
* **ROI noise std:** LD > GT (LD mean **0.1426** vs GT **0.1291**), consistent with noisier LD.&#x20;
* Visuals included random **LD vs GT** slices, **error maps**, **intensity histograms**, and **line profiles**.&#x20;

---

## 3 Methods

**Baselines.**

* **UNet (2D)**, **DnCNN**, **SwinIR-tiny** trained with **L1 + SSIM** loss, **AdamW**, and early stopping; evaluated by **PSNR/SSIM**. (Validation logs and ranking shown later.)&#x20;

**Conditional diffusion (DDPM).**

* Noise-prediction UNet conditioned on LD and timestep *t*; trained with **MSE ε-loss** and **EMA** weights; **T=1000** diffusion steps; default sampling **250** steps. (Training stats and ablations below.) &#x20;

---

## 4 Experiments

**Training/validation.**
All baselines were trained for a few epochs on CPU; validation **PSNR/SSIM** tracked per epoch. The **validation leaderboard** (best over runs) places **SwinIR > UNet > DnCNN** in our setup.&#x20;

**Diffusion details & ablations.**
DDPM trained briefly (2 epochs here, CPU), with **EMA** and sampling at **1000/250/50** steps; we report validation behavior and timing. &#x20;

---

## 5 Results

**Validation ranking (PSNR/SSIM).**
SwinIR led validation; UNet second; DnCNN third.&#x20;

**Test-set metrics (full):**

* **SwinIR:** PSNR **20.35 dB**, SSIM **0.775**, LPIPS **0.293**
* **UNet:** PSNR **19.98 dB**, SSIM **0.731**, LPIPS **0.312**
* **Diffusion (DDPM-Cond):** PSNR **6.34 dB**, SSIM **0.086**, LPIPS **0.673**
  (Complete table saved; see file/paths in the notebook run.)&#x20;

**Uncertainty (bootstrap 95% CI of mean):**

* **SwinIR:** PSNR **20.76** *(20.60–20.92)*; SSIM **0.775** *(0.768–0.780)*
* **UNet:** PSNR **20.59** *(20.39–20.77)*; SSIM **0.734** *(0.723–0.743)*
* **Diffusion:** PSNR **6.55** *(6.44–6.64)*; SSIM **0.086** *(0.084–0.088)*.&#x20;

**Efficiency.**
One-batch timing and parameter counts (illustrative): **Diffusion** incurs much higher wall-clock per sample than CNN baselines; example quick-timing table and params are reported (e.g., \~2.29 M params for DDPM-UNet in this setup).&#x20;

**Qualitative.**

* **Side-by-side:** **LD | UNet | SwinIR | Diffusion | GT** captured on the test set.
* **Error maps:** |pred−GT| (UNet/SwinIR/Diffusion).
* **Intensity histograms (proxy-HU):** model outputs vs GT.
  Artifacts were saved to figures as reported. &#x20;

---

## 6 Discussion

**Why do the baselines win?** On this tiny-resolution proxy (28×28), **feed-forward CNNs/transformers** learn effective denoising within a few epochs and are efficient at inference. Our DDPM, trained **very briefly** and evaluated at 50–250–1000 sampling steps, underperformed across PSNR/SSIM/LPIPS and was much slower. Short training, CPU-only constraints, small image size, and model/schedule choices likely limited diffusion’s performance here. (Ablation table shows better metrics at 1000 steps but still far below CNNs.)&#x20;

**What do the visuals say?** Error maps suggest diffusion residuals concentrate at edges while CNN baselines exhibit broader low-magnitude smoothing; histogram overlaps indicate SwinIR/UNet are closer to GT than diffusion in this run—matching the metrics.&#x20;

**Takeaway.** In this **proxy** setting, **SwinIR-tiny** slightly outperforms **UNet** (SSIM/PSNR), and both dominate the short-trained diffusion model; however, diffusion provides a flexible generative framework that may surpass baselines with **longer training**, **larger images (e.g., MedMNIST+)**, improved **conditioning**, and **modern objectives** (e.g., **v-prediction**, **cosine schedule**).&#x20;

---

## 7 Limitations

MedMNIST lacks **dose** and **acquisition metadata** (KVP, mAs, kernel, spacing, thickness), so we cannot analyze physics-dose relationships; intensities are **proxy-HU**. Findings should not be considered clinical—this is a methodological benchmark with simulated LD.&#x20;

---

## 8 Conclusion

On simulated LDCT slices from OrganMNIST3D, **UNet** and **SwinIR-tiny** deliver strong reconstruction quality; our brief **DDPM** baseline lags but highlights a path toward **generative** LDCT reconstruction. Future work: 3D models, longer training on larger inputs, better diffusion objectives, and evaluation on **clinical LD/SD** datasets with DICOM metadata for dose-aware analysis. &#x20;

---

## Acknowledgments

We used **MedMNIST** (OrganMNIST3D) for standardized experimentation; please see the dataset’s publications for citation details (Scientific Data 2023; ISBI 2021).

**References**
Yang, J., Shi, R., *et al.* “MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” *Scientific Data*, 10(1):41, 2023.
Yang, J., Shi, R., Ni, B. “MedMNIST Classification Decathlon,” *ISBI*, 2021.

*(Notebook artifacts referenced: validation ranking, test-set table, CI table, and saved figures are listed in the run logs and figure paths.)*&#x20;
