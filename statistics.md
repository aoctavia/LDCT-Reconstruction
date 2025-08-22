# 📊 Statistical & Physical Formulations in LDCT Project

This document summarizes the **key statistical and physical equations** used in the project.  
All formulations are presented in LaTeX and can be rendered directly in GitHub (via MathJax).

---

## 1. Noise Model in Low-Dose CT (Simulation)

In medical imaging, **low-dose CT (LDCT)** degradation can be simulated as:

1. **Photon statistics (Poisson noise):**

$$
I_{\text{LD}} \sim \text{Poisson}(I_{\text{GT}})
$$

where \( I_{\text{GT}} \) is the ground-truth photon count image.  

2. **Readout noise (Gaussian):**

$$
I_{\text{LD}} \;=\; I_{\text{Poisson}} + \mathcal{N}(0, \sigma^2)
$$

This combination captures stochastic photon arrivals (Poisson) and detector/electronics noise (Gaussian).

---

## 2. Error Map (Difference Image)

To visualize reconstruction fidelity, we use **pixel-wise error maps**:

$$
E(x,y) \;=\; |I_{\text{pred}}(x,y) - I_{\text{GT}}(x,y)|
$$

where \( I_{\text{pred}} \) is the model prediction and \( I_{\text{GT}} \) is the ground truth.

---

## 3. Peak Signal-to-Noise Ratio (PSNR)

A common metric for image reconstruction quality:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left( I_{\text{pred}}^{(i)} - I_{\text{GT}}^{(i)} \right)^2
$$

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{I_{\max}^2}{\text{MSE}} \right)
$$

- \( I_{\max} \): maximum possible intensity value (here = 1.0 since images are normalized).
- Higher PSNR → better reconstruction.

---

## 4. Structural Similarity Index (SSIM)

Captures perceptual similarity:

$$
\text{SSIM}(x,y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

- \( \mu_x, \mu_y \): mean intensities  
- \( \sigma_x^2, \sigma_y^2 \): variances  
- \( \sigma_{xy} \): covariance  
- \( C_1, C_2 \): stability constants  

---

## 5. Learned Perceptual Image Patch Similarity (LPIPS)

A perceptual metric based on deep features:

$$
\text{LPIPS}(x,y) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot ( \phi_l(x)_{h,w} - \phi_l(y)_{h,w} ) \|_2^2
$$

- \( \phi_l(\cdot) \): deep feature maps at layer \( l \)  
- \( w_l \): learned weights  
- Lower LPIPS → perceptually closer to GT  

---

## 6. Line Profile Analysis (Edge Sharpness)

For a fixed row/column in the slice:

$$
L(j) = I(j,y_0), \quad j = 1,2,\dots,W
$$

where \( W \) is the image width.  
Sharpness is measured by gradient:

$$
\text{Edge Contrast} = \max_j \left| \frac{\partial L(j)}{\partial j} \right|
$$

---

## 7. Noise Standard Deviation in ROI

To quantify noise level in a homogeneous region:

$$
\sigma_{\text{ROI}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N \left( I_i - \mu \right)^2}
$$

- \( I_i \): pixel intensities in the ROI  
- \( \mu \): mean intensity  

In LDCT, \( \sigma_{\text{ROI}} \) is expected to be higher.

---

## 8. Diffusion Model Formulation (DDPM)

Forward diffusion process (adding Gaussian noise):

$$
q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I\right)
$$

with variance schedule \( \{\beta_t\} \).  

Closed-form relation:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, 
\quad \epsilon \sim \mathcal{N}(0,I)
$$

where \( \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s) \).  

---

## 9. Reverse Process (Sampling)

The neural network \( \epsilon_\theta \) predicts noise:

$$
p_\theta(x_{t-1} | x_t, I_{\text{LD}}) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t, I_{\text{LD}}), \Sigma_t\right)
$$

with mean:

$$
\mu_\theta(x_t, t, I_{\text{LD}}) = \frac{1}{\sqrt{1-\beta_t}} 
\left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t, I_{\text{LD}}) \right)
$$

---

## 10. Bootstrap Confidence Interval

To provide statistical significance in evaluation metrics:

$$
\hat{\theta}^* = \frac{1}{B} \sum_{b=1}^B \theta^{*(b)}
$$

where \( \theta^{*(b)} \) are bootstrap replicates (resampled metrics).  

95% CI is given by the 2.5% and 97.5% quantiles of \( \{\theta^{*(b)}\}_{b=1}^B \).

---

# ✅ Summary

- **Noise models**: Poisson + Gaussian (low-dose simulation).  
- **Metrics**: PSNR, SSIM, LPIPS (quality & perceptual fidelity).  
- **Analysis tools**: error maps, line profiles, ROI noise statistics.  
- **Generative modeling**: forward/reverse diffusion processes.  
- **Statistical rigor**: bootstrap confidence intervals for metrics.  

These formulations provide the **theoretical backbone** of the LDCT reconstruction experiments in this repository.
