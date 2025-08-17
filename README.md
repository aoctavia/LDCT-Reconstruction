# AI-Powered Low-Dose CT Reconstruction (Diffusion vs UNet/SwinIR)

## 0) Target & Deliverables

* **Goal:** Rekonstruksi **low-dose CT → standard-dose**.
* **Deliverables:**

  1. Repo GitHub (clean, reproducible)
  2. Colab notebook demo
  3. ArXiv preprint (+ poster 1 halaman)
  4. Model weights + eval scripts

---

## 1) Struktur Repo

```
ldct-diffusion/
├─ README.md
├─ env.yml                      # conda env
├─ data/
│  ├─ raw/  ├─ processed/
├─ ldct/
│  ├─ datasets.py               # DICOM/NIfTI loader, pairing LD↔SD
│  ├─ transforms.py             # HU→window, norm, crop, aug
│  ├─ metrics.py                # PSNR, SSIM, LPIPS, (FID/NIQE opsional)
│  ├─ viz.py                    # side-by-side, hist HU
│  ├─ models/
│  │  ├─ unet.py
│  │  ├─ dncnn.py
│  │  ├─ swinir.py
│  │  └─ diffusion/
│  │     ├─ ddpm.py            # UNet backbone + scheduler
│  │     └─ scheduler.py
│  ├─ train_baseline.py         # UNet/DnCNN/SwinIR
│  ├─ train_diffusion.py
│  ├─ eval.py                   # batched eval + tables
│  └─ utils.py                  # seed, ckpt, logging
├─ notebooks/
│  ├─ 00_data_prep.ipynb
│  ├─ 01_train_baselines.ipynb
│  └─ 02_train_diffusion.ipynb
└─ scripts/
   ├─ prepare_mayo.sh
   └─ run_eval.sh
```

---

## 2) Environment

* **PyTorch** ≥ 2.2, **torchvision**, **monai**, **pydicom**, **nibabel**, **einops**, **timm**, **lpips**, **wandb** (atau lightning/fabric opsional).
* `env.yml`:

```yaml
name: ldct
channels: [pytorch, conda-forge, nvidia]
dependencies:
  - python=3.10
  - pytorch pytorch-cuda=12.1 torchvision
  - monai-core pydicom nibabel opencv
  - timm einops lpips scikit-image
  - matplotlib pandas tqdm wandb
```

---

## 3) Data & Preprocessing

**Dataset:** Mayo LDCT (atau AAPM LDCT).
**Langkah:**

1. **Download & Verify** → simpan ke `data/raw/`.
2. **Parse DICOM/NIfTI** → extract pixel array, **RescaleSlope/Intercept**, konversi ke **HU**.
3. **Windowing** (mis. lung: center −600, width 1500) + **normalize** ke `[-1, 1]`.
4. **Pairing** **(LD, SD)** per slice/volume (match SOPInstanceUID/series).
5. **Split** patient-wise (train/val/test).
6. **Augmentasi** (random crop 256×256, flip, slight rotation) — **konsisten LD & SD**.

`datasets.py` (sketsa):

```python
class LDCTPair(torch.utils.data.Dataset):
    def __init__(self, index_csv, crop=256, window=( -600, 1500 ), norm='tanh'):
        # load paths & pairing
    def __getitem__(self, i):
        ld, sd = load_pair_as_hu(self.items[i])
        ld = apply_window_norm(ld); sd = apply_window_norm(sd)
        ld, sd = random_crop_pair(ld, sd, size=self.crop)
        return torch.from_numpy(ld)[None], torch.from_numpy(sd)[None]
```

---

## 4) Baselines (wajib, cepat jadi)

**UNet** & **DnCNN** (image-to-image denoising) + **SwinIR** (Transformer restoration).

* **Loss:** L1 + SSIM (λ=0.84/0.16)
* **Optim:** AdamW (lr 2e-4), Cosine decay, 200–300 epochs
* **Batch:** 16 (256×256), mixed precision
* **Log:** PSNR/SSIM val every epoch, save best

`train_baseline.py` (sketsa):

```python
model = UNet(ch_in=1, ch_base=64, depth=4)  # atau DnCNN/SwinIR
crit_l1 = nn.L1Loss()
crit_ssim = SSIM(window_size=11)
for epoch in range(E):
    for ld, sd in train_loader:
        pred = model(ld)
        loss = 0.84*crit_l1(pred, sd) + 0.16*(1-crit_ssim(pred, sd))
        loss.backward(); opt.step(); opt.zero_grad()
```

---

## 5) Diffusion Model (highlight AI/ML-nya)

**DDPM** dengan **UNet backbone** (2D; opsional 3D patch-wise).

* **Forward (q):** tambahkan noise bertahap `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε`.
* **Model:** prediksi ε (noise) atau x0.
* **Loss:** MSE(ε̂, ε).
* **Sampling:** DDPM/PNDM/DDIM scheduler (mulai 1000 → 250/100 langkah).
* **Conditioning:** pakai **noisy LD** sebagai kondisi → **conditional diffusion** (concat channels atau FiLM).

`ddpm.py` (sketsa inti):

```python
class CondUNet(nn.Module):
    # UNet dengan time-embedding + conditioning LD (concat [LD, x_t])
class DDPM:
    def p_losses(self, x0_sd, ld_cond, t):
        noise = torch.randn_like(x0_sd)
        x_t = q_sample(x0_sd, t, noise)
        x_in = torch.cat([ld_cond, x_t], dim=1)
        noise_pred = self.unet(x_in, t_embed(t))
        return F.mse_loss(noise_pred, noise)
```

**Training Diffusion:**

* Steps: 200–300 epochs; lr 1e-4; EMA weights.
* **Ablasi:**

  * predict-ε vs predict-x0,
  * jumlah steps sampling (1000 vs 250 vs 50),
  * conditioning mode (concat vs cross-attn),
  * 2D slice vs 2.5D (stack tetangga slice).

---

## 6) Evaluasi & Visualisasi

**Metrics:**

* **PSNR**, **SSIM**, **LPIPS** (perceptual), **NIQE**/**FID** (opsional).
* **Clinical sanity checks:** histogram HU, edge-preservation.

`metrics.py` (sketsa):

```python
psnr = calc_psnr(pred, gt)
ssim = calc_ssim(pred, gt)
lpips_v = lpips_model(pred_3c, gt_3c).mean()
```

**Viz:**

* Side-by-side: LD vs **UNet** vs **SwinIR** vs **Diffusion** vs **GT**.
* Error maps |pred−GT|, zoom ROI (lesi/struktur halus).

---

## 7) Experiments Checklist (yang “menjual”)

* ✅ **Baseline**: UNet, DnCNN, SwinIR (bandingkan speed/param/metrics).
* ✅ **Main**: Conditional DDPM (ε-pred, 250 steps).
* ✅ **Ablation**:

  1. jumlah diffusion steps (1000/250/50),
  2. loss L1+SSIM vs pure MSE (diffusion),
  3. windowing scheme (lung vs abdomen),
  4. 2D vs 2.5D,
  5. conditioning method (concat vs FiLM).

Semua hasil → tabel + plot; simpan ke `results/`.

---

## 8) Dokumentasi (supaya terlihat “PhD-ready”)

* **README** (singkat, jelas, install → prepare data → train → eval → demo).
* **ArXiv draft** (Overleaf):

  * Abstract (1 paragraf jelas),
  * Intro (LDCT problem + AI angle),
  * Method (baseline + diffusion),
  * Experiments (dataset, metrics, tables),
  * Results (gambar SxS, ablation),
  * Discussion (artefak, limitasi klinis),
  * Conclusion (future work: 3D, dose-aware).
* **Colab demo** yang load pretrained dan run `eval.py` pada beberapa slice.

---

## 9) Timeline Cepat (≈ 6 minggu)

* **M1:** Data prep + loader + baseline UNet (PSNR/SSIM stabil).
* **M2:** DnCNN/SwinIR + evaluasi komparatif.
* **M3–M4:** Implement **conditional DDPM**, training stabil, sampling 250 steps.
* **M5:** Ablation + LPIPS + visualisasi klinis.
* **M6:** Rapikan repo, tulis arXiv, siapkan poster.

---

## 10) Risiko & Mitigasi

* **Mismatch pairing LD↔SD:** buat validator (cek UID/shape).
* **Noise over-smooth:** tambahkan **perceptual loss** (LPIPS) di baseline; untuk diffusion, kurangi langkah sampling dan pakai guidance.
* **Compute berat:** mulai dari **2D 256×256**, mixed precision, caching patch.
* **Overfit:** patient-wise split + strong aug + early stop.

---

## 11) Command Template

```bash
# 1) Siapkan data
bash scripts/prepare_mayo.sh  # convert ke NIfTI + index.csv

# 2) Train baseline UNet
python -m ldct.train_baseline --model unet --epochs 250 --crop 256 --bs 16

# 3) Train SwinIR
python -m ldct.train_baseline --model swinir --epochs 250 --crop 256 --bs 8

# 4) Train Diffusion (conditional)
python -m ldct.train_diffusion --steps 1000 --sample_steps 250 --ema --bs 8

# 5) Eval
python -m ldct.eval --ckpt runs/diffusion_best.pt --split test --save_fig
```

---

Kalau kamu mau, aku bisa **drop-in contoh file awal** (`datasets.py`, `unet.py`, `ddpm.py`, `train_diffusion.py`, dan `README.md` skeleton) supaya tinggal commit ke repo kamu.
