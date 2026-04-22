"""
backend.py — License Plate Deblurring: Core Pipeline
=====================================================

Five stages:
  1. PSF Generation  — motion-blur kernel via bilinear splatting
  2. PSF Estimation  — cepstrum (primary) + Radon transform (fallback)
  3. Deblurring      — TV Split-Bregman → Richardson-Lucy → TV denoise → unsharp mask
  4. Metrics         — PSNR / SSIM
  5. Pipeline        — load → estimate → deblur → measure → save

Usage:
    python code/backend.py --generate
    python code/backend.py --image data/synthetic/plate_001_blurred.png \
                           --ground_truth data/synthetic/plate_001.png
"""

import os
import warnings
import numpy as np
import scipy.signal
from numpy.fft import fft2, ifft2, fftshift
import cv2
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
from skimage.metrics import structural_similarity as _ssim_fn
from skimage.transform import radon


# =============================================================================
# SECTION 1 — PSF GENERATION & SYNTHETIC DATA
# =============================================================================
#
# Degradation model:
#     g = f * h + n
#
# g  = observed blurred image
# f  = sharp (unknown) image
# h  = Point Spread Function (PSF) — the blur kernel
# n  = additive Gaussian noise
# *  = 2D convolution
#
# For linear motion of length L at angle θ, the PSF is a normalised line:
#     (x(t), y(t)) = (cx + t·cosθ,  cy + t·sinθ),   t ∈ [−(L−1)/2, (L−1)/2]
#
# Bilinear splatting distributes each sample into 4 neighbouring pixels:
#     w(i,j) = (1 − |x−i|) · (1 − |y−j|)
# This prevents gaps at oblique angles and guarantees sum(PSF) = 1.
# =============================================================================

def make_motion_psf(length: int, angle_deg: float) -> np.ndarray:
    """
    Generate a 2D motion-blur PSF using bilinear splatting.

    Args:
        length:    Motion trail length in pixels (>= 1).
        angle_deg: Blur direction in degrees (0 = horizontal, CCW positive).

    Returns:
        psf: 2D float64 array, L1-normalised, shape (kernel_size, kernel_size).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if length == 1:
        return np.array([[1.0]], dtype=np.float64)

    size = int(2 * np.ceil(length / 2)) + 3   # odd size with margin
    psf  = np.zeros((size, size), dtype=np.float64)
    cx = cy = size // 2
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))

    for t in np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, max(length * 8, 64)):
        x, y   = cx + t * cos_a, cy + t * sin_a
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        for xi, wx in ((x0, 1.0 - dx), (x0 + 1, dx)):
            for yi, wy in ((y0, 1.0 - dy), (y0 + 1, dy)):
                if 0 <= xi < size and 0 <= yi < size:
                    psf[yi, xi] += wx * wy

    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        psf[cy, cx] = 1.0
    return psf


def apply_motion_blur(image: np.ndarray, psf: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Convolve image with PSF and add Gaussian noise.

    Reflect-pads before convolution to suppress boundary ringing.

    Args:
        image:     Grayscale float64 image in [0, 1].
        psf:       L1-normalised 2D PSF kernel.
        noise_std: Gaussian noise standard deviation.

    Returns:
        blurred: Float64 image in [0, 1].
    """
    ph, pw = psf.shape
    padded  = np.pad(image, ((ph // 2, ph // 2), (pw // 2, pw // 2)), mode="reflect")
    blurred = scipy.signal.convolve2d(padded, psf, mode="valid", boundary="fill")
    return np.clip(blurred + np.random.normal(0.0, noise_std, blurred.shape), 0.0, 1.0)


def _create_synthetic_plate(plate_id: int = 0, size: tuple = (128, 256)) -> np.ndarray:
    """Render a synthetic license plate with OpenCV text (grayscale, float64)."""
    H, W = size
    plates = ["AB-1234", "CD-5678", "EF-9012", "GH-3456",
              "IJ-7890", "KL-2345", "MN-6789", "PQ-0123"]
    text  = plates[plate_id % len(plates)]
    img   = np.full((H, W, 3), (240, 235, 210), dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (W - 3, H - 3), (40, 40, 40), 3)

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = H / 55.0
    thickness  = max(2, int(H / 22))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(img, text, ((W - tw) // 2, (H + th) // 2),
                font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64) / 255.0


def generate_synthetic_dataset(
    output_dir: str,
    n_images: int = 3,
    lengths: list = None,
    angles: list = None,
    noise_std: float = 0.02,
) -> None:
    """
    Generate blurred license plate pairs with ground truth.

    Saves per image:
        plate_{i:03d}.png          — sharp ground truth
        plate_{i:03d}_blurred.png  — motion blurred + noise
        plate_{i:03d}_psf.npy      — PSF used (for offline evaluation)
    """
    os.makedirs(output_dir, exist_ok=True)
    lengths = lengths or [15, 20, 25]
    angles  = angles  or [0, 15, 30]

    for i in range(n_images):
        length  = lengths[i % len(lengths)]
        angle   = angles[i % len(angles)]
        sharp   = _create_synthetic_plate(plate_id=i + 1)
        psf     = make_motion_psf(length=length, angle_deg=angle)
        blurred = apply_motion_blur(sharp, psf, noise_std=noise_std)

        imsave(os.path.join(output_dir, f"plate_{i+1:03d}.png"),
               img_as_ubyte(np.clip(sharp, 0, 1)))
        imsave(os.path.join(output_dir, f"plate_{i+1:03d}_blurred.png"),
               img_as_ubyte(np.clip(blurred, 0, 1)))
        np.save(os.path.join(output_dir, f"plate_{i+1:03d}_psf.npy"), psf)
        print(f"[data] plate_{i+1:03d}  length={length} px  angle={angle}°")


# =============================================================================
# SECTION 2 — PSF ESTIMATION
# =============================================================================
#
# In the frequency domain the degradation model becomes:
#     G(u,v) = F(u,v) · H(u,v) + N(u,v)
#
# A motion PSF of length L at angle θ has an OTF with spectral zeros
# (dark stripes in the FFT magnitude) perpendicular to the blur direction,
# spaced 1/L apart.
#
# Primary method — Cepstrum:
#     c(x,y) = IFFT2( log|FFT2(g)| )
#     The PSF cepstrum has sharp peaks at offsets (±L·cosθ, ±L·sinθ).
#     Image content stays near DC (r < 8 px), so we search a mid-range
#     annulus r ∈ [8, 40] for the dominant peak → get θ.
#     Then spectral null-coincidence finds L.
#
# Fallback — Radon on log-magnitude spectrum:
#     The Radon projection of the spectrum with maximum variance sweeps
#     across OTF dark stripes → angle equals the blur direction.
#     Peak spacing along a 1D profile gives L.
# =============================================================================

def compute_log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the DC-centred log-magnitude Fourier spectrum.

    M(u,v) = log(1 + |FFT2(image)|)
    """
    return np.log1p(np.abs(fftshift(fft2(image))))


def _estimate_length_null_coincidence(image: np.ndarray, angle_deg: float,
                                       L_min: int = 5, L_max: int = 55) -> int:
    """
    Estimate PSF length by matching OTF spectral zeros.

    The OTF of a motion PSF has zeros at:
        (fy, fx) = (cy ± k·H·sinθ/L,  cx ± k·W·cosθ/L),  k = 1, 2, …

    For each candidate L, compute:
        score(L) = energy_at_nulls / energy_at_half-nulls

    The correct L minimises this ratio because its nulls land on genuine
    OTF zeros while the half-null positions land on |sinc| peaks.
    """
    H, W = image.shape
    mag  = np.abs(fftshift(fft2(image)))
    cy, cx = H // 2, W // 2
    cos_a  = np.cos(np.deg2rad(angle_deg))
    sin_a  = np.sin(np.deg2rad(angle_deg))

    def _bilinear(fy, fx):
        fy, fx = float(fy) % H, float(fx) % W
        y0, x0 = int(fy), int(fx)
        y1, x1 = (y0 + 1) % H, (x0 + 1) % W
        dy, dx = fy - y0, fx - x0
        return (mag[y0, x0] * (1 - dy) * (1 - dx) + mag[y0, x1] * (1 - dy) * dx
                + mag[y1, x0] * dy * (1 - dx) + mag[y1, x1] * dy * dx)

    best_score, best_L = np.inf, 15
    for L in range(L_min, L_max + 1):
        null_e = half_e = 0.0
        for k in range(1, 9):
            for sign in (1, -1):
                null_e += _bilinear(cy + sign * k * H * sin_a / L,
                                    cx + sign * k * W * cos_a / L)
                half_e += _bilinear(cy + sign * (k - 0.5) * H * sin_a / L,
                                    cx + sign * (k - 0.5) * W * cos_a / L)
        score = null_e / (half_e + 1e-10)
        if score < best_score:
            best_score, best_L = score, L

    return int(np.clip(best_L, 3, 60))


def _estimate_by_cepstrum(image: np.ndarray):
    """
    Cepstrum-based angle + length estimation.

    Returns (angle_deg, length) or (None, None) if peak confidence is low.
    """
    H, W = image.shape
    log_mag  = np.log(np.abs(fft2(image)) + 1e-8)
    cepstrum = np.abs(fftshift(np.real(ifft2(log_mag))))

    cy, cx = H // 2, W // 2
    Y, X   = np.ogrid[:H, :W]
    r_grid = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    min_r, max_r = 8, min(H // 2 - 4, W // 2 - 4, 40)
    search = cepstrum.copy()
    search[r_grid < min_r] = 0.0
    search[r_grid > max_r] = 0.0

    if search.max() == 0.0:
        return None, None

    py, px    = np.unravel_index(np.argmax(search), search.shape)
    angle_deg = float(np.degrees(np.arctan2(py - cy, px - cx)) % 180.0)

    vals = search[search > 0]
    if vals.size == 0 or search[py, px] < 1.5 * vals.mean():
        return None, None     # low-confidence peak

    length = _estimate_length_null_coincidence(image, angle_deg)
    return angle_deg, length


def estimate_psf_angle_radon(spectrum: np.ndarray) -> float:
    """
    Estimate blur angle from a log-magnitude spectrum using the Radon transform.

    The projection with maximum variance sweeps across OTF dark stripes,
    so the peak Radon angle equals the blur angle.
    """
    H, W  = spectrum.shape
    crop  = min(H, W) // 3
    centre = spectrum[H // 2 - crop: H // 2 + crop,
                      W // 2 - crop: W // 2 + crop].copy()

    dc_r, dc_c = np.array(centre.shape) // 2
    dc_rad     = max(5, min(centre.shape) // 12)
    Y, X = np.ogrid[:centre.shape[0], :centre.shape[1]]
    centre[(Y - dc_r) ** 2 + (X - dc_c) ** 2 < dc_rad ** 2] = 0.0

    theta    = np.arange(0, 180, 1, dtype=float)
    sinogram = radon(centre, theta=theta, circle=False)
    return float(theta[int(np.argmax(sinogram.var(axis=0)))] % 180.0)


def estimate_psf_length(spectrum: np.ndarray, angle_deg: float) -> int:
    """
    Estimate PSF length from the spacing of spectral zeros (Radon fallback path).

    Extracts a 1D profile along the blur direction and measures the minimum
    spacing between dominant dips (OTF zeros).
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    H, W = spectrum.shape
    cx, cy = W // 2, H // 2
    diag   = int(np.sqrt(H ** 2 + W ** 2) / 2)
    cos_a  = np.cos(np.deg2rad(angle_deg))
    sin_a  = np.sin(np.deg2rad(angle_deg))

    ts = np.arange(-diag, diag + 1)
    xs = np.clip((cx + ts * cos_a).astype(int), 0, W - 1)
    ys = np.clip((cy + ts * sin_a).astype(int), 0, H - 1)
    profile = uniform_filter1d(spectrum[ys, xs].astype(np.float64), size=3)

    peaks, _ = find_peaks(-profile, distance=3, prominence=0.05)
    if len(peaks) < 2:
        return 15
    N_eff = max(H * abs(sin_a) + W * abs(cos_a), min(H, W))
    return int(np.clip(round(N_eff / max(float(np.median(np.diff(np.sort(peaks)))), 1.0)), 3, 60))


def estimate_psf(image: np.ndarray, verbose: bool = False) -> dict:
    """
    Automatic PSF parameter estimation.

    Tries cepstrum first; falls back to Radon transform if confidence is low.

    Args:
        image:   Blurred grayscale or RGB image (float or uint8).
        verbose: Print intermediate estimates.

    Returns:
        dict: {'angle', 'length', 'spectrum', 'method'}
    """
    img = img_as_float(image)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)

    angle, length = _estimate_by_cepstrum(img)
    method = "cepstrum"

    if angle is None:
        spectrum = compute_log_magnitude_spectrum(img)
        angle    = estimate_psf_angle_radon(spectrum)
        length   = estimate_psf_length(spectrum, angle)
        method   = "radon"
    else:
        spectrum = compute_log_magnitude_spectrum(img)

    if verbose:
        print(f"[estimate] {method}  angle={angle:.1f}°  length={length} px")

    return {"angle": angle, "length": length, "spectrum": spectrum, "method": method}


# =============================================================================
# SECTION 3 — DEBLURRING
# =============================================================================
#
# All algorithms operate in the frequency domain:
#     G = F · H + N
#
# PSF → OTF conversion:
#     1. Zero-pad PSF to image size (top-left corner).
#     2. Roll by (−cy, −cx) so the PSF centre aligns with frequency-domain
#        origin — eliminates the linear phase ramp that would shift the image.
#     3. FFT2.
#
# Three deconvolution methods used in sequence:
#
#   Wiener filter (L2 prior, closed form):
#       W = H* / (|H|² + K)       K = noise/signal power ratio
#       Fast but produces Gibbs ringing at sharp edges.
#
#   TV Split-Bregman (L1 gradient prior, ADMM):
#       min_u (1/2)||h*u − g||² + λ||∇u||₁
#       Edge-preserving; best for text/plate characters.
#       +2–4 dB PSNR over Wiener on plate images.
#
#   Richardson-Lucy (Poisson MLE, iterative):
#       u^{k+1} = u^k · [h̃ * (g / (h*u^k))]   h̃ = flipped PSF
#       Enforces non-negativity; used as a refinement step after TV.
# =============================================================================

def psf_to_otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert a PSF kernel to an OTF (Optical Transfer Function) of size `shape`.

    Critical: roll by PSF half-dimensions (NOT ifftshift) because the PSF is
    much smaller than the image — ifftshift would shift by the image half-size.
    """
    kH, kW = psf.shape
    cy, cx = kH // 2, kW // 2

    padded = np.zeros(shape, dtype=np.float64)
    h_fit  = min(kH, shape[0])
    w_fit  = min(kW, shape[1])
    padded[:h_fit, :w_fit] = psf[:h_fit, :w_fit]
    s = padded.sum()
    if s > 0:
        padded /= s

    padded = np.roll(padded, -(cy % shape[0]), axis=0)
    padded = np.roll(padded, -(cx % shape[1]), axis=1)
    return fft2(padded)


def wiener_deconvolve(blurred: np.ndarray, psf: np.ndarray, K: float = 1e-3) -> np.ndarray:
    """
    Wiener deconvolution filter.

    Formula:  F̂ = G · H* / (|H|² + K)
    K = σ²_noise / σ²_signal (noise-to-signal power ratio).
    Minimises the mean squared error E[|F − F̂|²].

    Args:
        blurred: Grayscale float64 image in [0, 1].
        psf:     L1-normalised 2D PSF.
        K:       Regularisation constant. Larger → smoother.

    Returns:
        restored: Float64 image in [0, 1].
    """
    H = psf_to_otf(psf, blurred.shape)
    G = fft2(blurred)
    W = np.conj(H) / (np.abs(H) ** 2 + max(float(K), 1e-10))
    return np.clip(np.real(ifft2(W * G)), 0.0, 1.0)


def tv_deconvolve(
    blurred: np.ndarray,
    psf: np.ndarray,
    lam: float = 0.02,
    mu: float = 8.0,
    n_iter: int = 60,
) -> np.ndarray:
    """
    TV-regularised deconvolution via Split Bregman (ADMM).

    Problem:  min_u  (1/2)||h*u − g||²  +  λ||∇u||₁

    Split Bregman introduces splitting variable d = ∇u and Bregman variable b:

        u-step (exact, frequency domain):
            u = IFFT2( [H*·G + μ(Dx*·F(dx−bx) + Dy*·F(dy−by))]
                        / [|H|² + μ(|Dx|² + |Dy|²)] )

        d-step (isotropic soft-thresholding):
            s    = ∇u + b
            norm = ||s||₂ + ε
            d    = max(norm − λ/μ, 0) · s / norm

        b-step (Bregman / dual update):
            b = b + ∇u − d

    Args:
        blurred: Grayscale float64 image in [0, 1].
        psf:     L1-normalised 2D PSF.
        lam:     TV weight λ (larger → smoother). Typical: 0.01–0.05.
        mu:      ADMM penalty μ. Typical: 5–20.
        n_iter:  Number of Split Bregman iterations.

    Returns:
        u: Deconvolved float64 image in [0, 1].
    """
    H_img, W_img = blurred.shape

    H_otf  = psf_to_otf(psf, (H_img, W_img))
    H_conj = np.conj(H_otf)
    H_sq   = np.abs(H_otf) ** 2

    # OTFs of forward-difference gradient operators (periodic BC)
    dx_k         = np.zeros((H_img, W_img)); dx_k[0, 0] =  1.0; dx_k[0, 1] = -1.0
    dy_k         = np.zeros((H_img, W_img)); dy_k[0, 0] =  1.0; dy_k[1, 0] = -1.0
    Dx = fft2(dx_k); Dx_conj = np.conj(Dx); Dx_sq = np.abs(Dx) ** 2
    Dy = fft2(dy_k); Dy_conj = np.conj(Dy); Dy_sq = np.abs(Dy) ** 2

    denom     = np.maximum((H_sq + mu * (Dx_sq + Dy_sq)).real, 1e-10)
    G         = fft2(blurred.astype(np.float64))
    u         = blurred.copy()
    dx = dy = bx = by = np.zeros_like(blurred)
    threshold = lam / mu

    for _ in range(n_iter):
        # u-step
        num = H_conj * G + mu * (Dx_conj * fft2(dx - bx) + Dy_conj * fft2(dy - by))
        u   = np.clip(np.real(ifft2(num / denom)), 0.0, 1.0)

        # Spatial gradients of u
        U  = fft2(u)
        ux = np.real(ifft2(Dx * U))
        uy = np.real(ifft2(Dy * U))

        # d-step (isotropic shrinkage)
        sx, sy = ux + bx, uy + by
        norm   = np.sqrt(sx ** 2 + sy ** 2) + 1e-10
        shrink = np.maximum(norm - threshold, 0.0) / norm
        dx, dy = shrink * sx, shrink * sy

        # b-step
        bx += ux - dx
        by += uy - dy

    return np.clip(u, 0.0, 1.0)


def richardson_lucy_deconvolve(
    blurred: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 15,
    init: np.ndarray = None,
) -> np.ndarray:
    """
    Richardson-Lucy iterative deconvolution.

    EM update step under Poisson noise assumption:
        u^{k+1} = u^k · [ h̃ * (g / (h * u^k)) ]
        h̃ = PSF flipped 180° (adjoint operator)

    Enforces non-negativity at every iteration.
    Warm-started from `init` (typically the TV output) for fast convergence.

    Args:
        blurred: Grayscale float64 image in [0, 1].
        psf:     L1-normalised 2D PSF.
        n_iter:  Number of RL iterations (10–20 is typically optimal).
        init:    Starting estimate u^0 (None → use blurred).

    Returns:
        restored: Float64 image in [0, 1].
    """
    CLIP_MIN = 1e-6
    shape = blurred.shape
    H     = psf_to_otf(psf, shape)
    H_T   = psf_to_otf(psf[::-1, ::-1], shape)     # flipped PSF = adjoint
    u     = np.clip(init.copy() if init is not None else blurred.copy(), CLIP_MIN, 1.0)

    for _ in range(n_iter):
        Hu         = np.clip(np.real(ifft2(H * fft2(u))), CLIP_MIN, None)
        correction = np.real(ifft2(H_T * fft2(blurred / Hu)))
        u          = np.clip(u * np.clip(correction, 0.0, None), 0.0, 1.0)

    return u


def tv_denoise(image: np.ndarray, weight: float = 0.05) -> np.ndarray:
    """
    Chambolle Total Variation denoising (ROF model).

    Solves:  min_u  (1/2)||u − g||²  +  weight · ||∇u||₁

    Removes residual ringing and noise while preserving edges.
    """
    from skimage.restoration import denoise_tv_chambolle
    result = denoise_tv_chambolle(
        np.clip(image, 0, 1),
        weight=weight,
        channel_axis=-1 if image.ndim == 3 else None,
    )
    return np.clip(result.astype(np.float64), 0.0, 1.0)


def unsharp_mask(image: np.ndarray, amount: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    """
    Unsharp mask sharpening.

    Formula:  out = image + amount · (image − GaussianBlur(image, σ))

    Recovers fine-detail crispness lost during denoising.
    """
    ksize  = max(3, 2 * int(np.floor(2.0 * sigma + 0.5)) + 1)
    img32  = image.astype(np.float32)
    blurry = cv2.GaussianBlur(img32, (ksize, ksize), sigma)
    return np.clip((img32 + amount * (img32 - blurry)).astype(np.float64), 0.0, 1.0)


def deblur(
    blurred: np.ndarray,
    psf: np.ndarray,
    K: float = 3e-3,
    rl_iter: int = 10,
    use_tv: bool = True,
    tv_lam: float = 0.02,
    tv_mu: float = 8.0,
    tv_iter: int = 60,
    tv_denoise_weight: float = 0.04,
    sharpen_amount: float = 0.4,
    sharpen_sigma: float = 0.8,
) -> np.ndarray:
    """
    Full deblurring pipeline.

    TV path (default, use_tv=True):
        1. TV Split-Bregman  — edge-preserving deconvolution
        2. Richardson-Lucy   — non-negativity refinement, warm-started from TV
        3. TV Chambolle      — residual noise removal
        4. Unsharp mask      — fine-detail recovery

    Wiener path (use_tv=False):
        1. Wiener filter     — closed-form L2 deconvolution
        2. Richardson-Lucy   — iterative refinement
        3. TV Chambolle      — post-denoising
        4. Unsharp mask

    Handles grayscale (H×W) and colour (H×W×C) by processing each channel.
    """
    img = img_as_float(blurred)

    # Colour: recurse per channel
    if img.ndim == 3:
        return np.stack([
            deblur(img[..., c], psf, K=K, rl_iter=rl_iter, use_tv=use_tv,
                   tv_lam=tv_lam, tv_mu=tv_mu, tv_iter=tv_iter,
                   tv_denoise_weight=tv_denoise_weight,
                   sharpen_amount=sharpen_amount, sharpen_sigma=sharpen_sigma)
            for c in range(img.shape[2])
        ], axis=-1)

    # Grayscale path
    if use_tv:
        deconvolved = tv_deconvolve(img, psf, lam=tv_lam, mu=tv_mu, n_iter=tv_iter)
    else:
        deconvolved = wiener_deconvolve(img, psf, K=K)

    if rl_iter > 0:
        deconvolved = richardson_lucy_deconvolve(img, psf, n_iter=rl_iter, init=deconvolved)

    filtered  = tv_denoise(deconvolved, weight=tv_denoise_weight)
    restored  = unsharp_mask(filtered, amount=sharpen_amount, sigma=sharpen_sigma)
    return restored


# =============================================================================
# SECTION 4 — QUALITY METRICS
# =============================================================================
#
# PSNR (Peak Signal-to-Noise Ratio):
#     PSNR = 10 · log10(MAX² / MSE)   MAX = 1.0 for float images
#     Higher is better. < 20 dB = poor; 30–40 dB = good.
#
# SSIM (Structural Similarity Index):
#     SSIM(f, f̂) = [luminance] · [contrast] · [structure]
#              = (2μ_f μ_f̂ + C1)(2σ_ff̂ + C2)
#                ─────────────────────────────────────
#                (μ_f² + μ_f̂² + C1)(σ_f² + σ_f̂² + C2)
#     SSIM ∈ [−1, 1]; 1 = identical. Computed over local 7×7 Gaussian windows.
# =============================================================================

def compute_psnr(reference: np.ndarray, restored: np.ndarray) -> float:
    """PSNR in dB between reference and restored (both converted to float [0,1])."""
    ref, res = img_as_float(reference), img_as_float(restored)
    if ref.shape != res.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {res.shape}")
    return float(_psnr_fn(ref, res, data_range=1.0))


def compute_ssim(reference: np.ndarray, restored: np.ndarray) -> float:
    """Mean SSIM between reference and restored. Handles grayscale and colour."""
    ref, res = img_as_float(reference), img_as_float(restored)
    if ref.shape != res.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {res.shape}")
    kwargs = dict(data_range=1.0)
    if ref.ndim == 3:
        kwargs["channel_axis"] = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_ssim_fn(ref, res, **kwargs))


def compute_metrics(reference: np.ndarray, blurred: np.ndarray, restored: np.ndarray) -> dict:
    """
    Compute PSNR and SSIM for blurred and restored images vs. reference.

    Returns:
        dict: blurred_psnr, blurred_ssim, restored_psnr, restored_ssim,
              psnr_gain, ssim_gain
    """
    bp = compute_psnr(reference, blurred)
    bs = compute_ssim(reference, blurred)
    rp = compute_psnr(reference, restored)
    rs = compute_ssim(reference, restored)
    return {
        "blurred_psnr":  bp, "blurred_ssim":  bs,
        "restored_psnr": rp, "restored_ssim": rs,
        "psnr_gain":     rp - bp,
        "ssim_gain":     rs - bs,
    }


# =============================================================================
# SECTION 5 — END-TO-END PIPELINE
# =============================================================================

def run_pipeline(
    image_path: str,
    ground_truth_path: str = None,
    psf_npy_path: str = None,
    K: float = 3e-3,
    output_dir: str = "results",
) -> dict:
    """
    Full license plate deblurring pipeline.

    Steps:
        1. Load blurred image → grayscale float.
        2. PSF: load from .npy file OR auto-estimate (cepstrum → Radon fallback).
        3. Deblur: TV Split-Bregman → Richardson-Lucy → TV denoise → unsharp mask.
        4. Metrics: PSNR / SSIM vs ground truth (if provided).
        5. Save restored image.

    Args:
        image_path:        Blurred input image path.
        ground_truth_path: Sharp ground-truth path (optional, for metrics).
        psf_npy_path:      .npy PSF file path (optional, auto-estimated if None).
        K:                 Wiener K (used only in fallback Wiener path).
        output_dir:        Directory for output files.

    Returns:
        dict: {'restored', 'metrics', 'psf_params'}
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Load
    blurred_raw = img_as_float(imread(image_path))
    blurred     = blurred_raw.mean(axis=2) if blurred_raw.ndim == 3 else blurred_raw
    print(f"[pipeline] Loaded   : {image_path}  shape={blurred.shape}")

    # 2. PSF
    if psf_npy_path and os.path.exists(psf_npy_path):
        psf        = np.load(psf_npy_path)
        est        = estimate_psf(blurred, verbose=False)
        psf_params = {"angle": est["angle"], "length": est["length"],
                      "spectrum": est["spectrum"]}
        print(f"[pipeline] PSF      : loaded from {psf_npy_path}")
    else:
        est        = estimate_psf(blurred, verbose=True)
        psf_params = {"angle": est["angle"], "length": est["length"],
                      "spectrum": est["spectrum"]}
        psf        = make_motion_psf(length=est["length"], angle_deg=est["angle"])

    # 3. Deblur
    restored = deblur(blurred, psf, K=K)
    print(f"[pipeline] Deblurred: TV + RL pipeline complete")

    out_path = os.path.join(output_dir, f"{stem}_restored.png")
    imsave(out_path, img_as_ubyte(np.clip(restored, 0, 1)))
    print(f"[pipeline] Saved    : {out_path}")

    # 4. Metrics
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt  = img_as_float(imread(ground_truth_path))
        gt  = gt.mean(axis=2) if gt.ndim == 3 else gt
        metrics = compute_metrics(gt, blurred, restored)
        print(f"[pipeline] PSNR  blurred={metrics['blurred_psnr']:.2f} dB  "
              f"restored={metrics['restored_psnr']:.2f} dB  "
              f"gain=+{metrics['psnr_gain']:.2f} dB")
        print(f"[pipeline] SSIM  blurred={metrics['blurred_ssim']:.4f}     "
              f"restored={metrics['restored_ssim']:.4f}     "
              f"gain=+{metrics['ssim_gain']:.4f}")
    else:
        metrics = {}
        print("[pipeline] No ground truth — skipping metrics.")

    return {"restored": restored, "metrics": metrics, "psf_params": psf_params}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="License Plate Deblurring Pipeline")
    parser.add_argument("--generate",     action="store_true",
                        help="Generate synthetic dataset")
    parser.add_argument("--data_dir",     default="data/synthetic")
    parser.add_argument("--n_images",     type=int, default=3)
    parser.add_argument("--image",        default=None,
                        help="Path to blurred image")
    parser.add_argument("--ground_truth", default=None)
    parser.add_argument("--psf",          default=None,
                        help="PSF .npy file (auto-estimated if omitted)")
    parser.add_argument("--K",            type=float, default=3e-3)
    parser.add_argument("--output_dir",   default="results")
    args = parser.parse_args()

    if args.generate:
        generate_synthetic_dataset(output_dir=args.data_dir, n_images=args.n_images)

    if args.image:
        run_pipeline(
            image_path=args.image,
            ground_truth_path=args.ground_truth,
            psf_npy_path=args.psf,
            K=args.K,
            output_dir=args.output_dir,
        )
    elif not args.generate:
        parser.print_help()
