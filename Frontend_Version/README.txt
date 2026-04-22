================================================================================
  ✦  KINTSUGI  ✦   License Plate Deblurring
  — The Art of Golden Restoration
================================================================================

OVERVIEW
--------
Kintsugi is a computer vision application that restores motion-blurred license
plate images. It automatically estimates the blur kernel (PSF) and applies a
multi-stage deconvolution pipeline to recover sharp, readable plate text.

The web interface is built with Gradio and uses a black-and-gold aesthetic
inspired by the Japanese art of kintsugi (repairing broken pottery with gold).


FILES
-----
  app.py       — Gradio web application (UI layout, theme, and event handling)
  backend.py   — Core image processing pipeline (PSF estimation & deblurring)


REQUIREMENTS
------------
  Python 3.8+

  Install dependencies:
    pip install gradio numpy scipy matplotlib pillow opencv-python scikit-image


USAGE
-----

  1. Run the web app:
       python app.py
       Then open http://127.0.0.1:7860 in your browser.

  2. In the UI:
       - Upload a blurred license plate image.
       - Optionally upload a ground-truth (sharp) image to enable metrics.
       - Click "Generate Demo" to load a built-in synthetic example.
       - Adjust advanced parameters (Wiener K, Richardson-Lucy iterations)
         under the "Advanced Parameters" accordion if needed.
       - Click "✦ RESTORE IMAGE ✦" to run the pipeline.

  3. Run the backend from the command line:

       Generate a synthetic dataset:
         python backend.py --generate --data_dir data/synthetic --n_images 3

       Deblur a specific image:
         python backend.py --image path/to/blurred.png \
                           --ground_truth path/to/sharp.png \
                           --output_dir results/

       Use a known PSF (skips estimation):
         python backend.py --image blurred.png --psf plate_001_psf.npy


PIPELINE STAGES
---------------
  1. PSF Estimation
       - Primary:  Cepstrum analysis (detects blur angle and length)
       - Fallback: Radon transform on the log-magnitude FFT spectrum

  2. Deconvolution
       - TV Split-Bregman  — edge-preserving deconvolution (main step)
       - Richardson-Lucy   — iterative non-negative refinement
       - TV Chambolle      — residual noise removal
       - Unsharp mask      — final sharpening

  3. Quality Metrics (when ground truth is provided)
       - PSNR  (Peak Signal-to-Noise Ratio, in dB)
       - SSIM  (Structural Similarity Index)
       - Both reported for the blurred input and the restored output,
         plus the gain from restoration.


ADVANCED PARAMETERS
-------------------
  Wiener K (default: 0.003)
      Noise-to-signal regularisation constant.
      Lower = sharper but noisier. Higher = smoother.
      Range: 0.0001 – 0.05

  Richardson-Lucy Iterations (default: 15)
      Number of iterative refinement steps after TV deconvolution.
      0 = Wiener/TV only. 15 = balanced. 30 = maximum sharpness.


OUTPUT
------
  Restored Image     — Grayscale PNG, displayed in the UI and available
                       for download.
  Comparison Plot    — Side-by-side: Blurred | Restored | FFT Spectrum
  PSF Info           — Estimated blur length (px), angle (°), and
                       parameters used.
  Metrics Table      — PSNR and SSIM for blurred vs. restored (if
                       ground truth was supplied).

  CLI output files are saved to the directory specified by --output_dir:
      <stem>_restored.png   — restored image
      plate_NNN.png         — sharp ground truth (synthetic data)
      plate_NNN_blurred.png — blurred version   (synthetic data)
      plate_NNN_psf.npy     — PSF kernel array  (synthetic data)


NOTES
-----
  - All processing is done in grayscale internally; colour images are
    handled by processing each channel independently.
  - The cepstrum estimator searches for blur lengths between 5 and 55 px
    and angles across the full 0–180° range.
  - TV Split-Bregman typically yields +2 to +4 dB PSNR over Wiener alone
    on license plate images.
  - The synthetic plate generator covers 8 plate templates rendered with
    OpenCV at 128×256 px.

================================================================================
