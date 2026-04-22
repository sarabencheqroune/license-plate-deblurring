"""
app.py — Kintsugi: License Plate Deblurring Web Application

Kintsugi (金継ぎ) — The art of golden restoration.
Black obsidian background, pure gold accents.
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from backend import (
    make_motion_psf,
    generate_synthetic_dataset,
    estimate_psf,
    deblur,
    compute_metrics,
)

# ─────────────────────────────────────────────────────────────
# Kintsugi Colour Palette — Black & Gold
# ─────────────────────────────────────────────────────────────
BG          = "#080808"   # near-black
SURFACE     = "#111111"   # card surface
SURFACE2    = "#181818"   # slightly lifted surface
GOLD        = "#C9A84C"   # principal gold
GOLD_LIGHT  = "#E8C96E"   # highlight gold
GOLD_DARK   = "#8B6914"   # shadow gold
BORDER      = "#2a1e00"   # subtle warm border
TEXT        = "#E8D9B5"   # warm cream
TEXT_MUTED  = "#9A8866"   # muted warm grey

# ─────────────────────────────────────────────────────────────
# Gradio Theme
# ─────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.Color(
        c50="#FDF8EE", c100="#F5E9C8", c200="#EDD898", c300="#E4C668",
        c400="#DDB848", c500=GOLD, c600=GOLD_DARK,
        c700="#6B4F10", c800="#4A360A", c900="#2E2106", c950="#171003",
    ),
    neutral_hue=gr.themes.colors.Color(
        c50=TEXT, c100="#D4C9A8", c200="#B0A480", c300="#8C8060",
        c400="#685E44", c500="#443E2C", c600="#2E2918",
        c700=SURFACE2, c800=SURFACE, c900=BG, c950="#030303",
    ),
    secondary_hue="yellow",
    spacing_size="lg",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Cormorant Garamond"), "Georgia", "serif"],
    font_mono=[gr.themes.GoogleFont("Courier Prime"), "Courier New", "monospace"],
).set(
    body_background_fill=BG,
    body_background_fill_dark=BG,
    block_background_fill=SURFACE,
    block_background_fill_dark=SURFACE,
    block_border_color=BORDER,
    block_border_color_dark=BORDER,
    block_border_width="1px",
    block_title_text_color=GOLD,
    block_title_text_color_dark=GOLD,
    block_label_text_color=GOLD,
    block_label_text_color_dark=GOLD,
    block_label_background_fill=BG,
    block_label_background_fill_dark=BG,
    panel_background_fill=SURFACE2,
    panel_background_fill_dark=SURFACE2,
    panel_border_color=BORDER,
    panel_border_color_dark=BORDER,

    input_background_fill=SURFACE2,
    input_background_fill_dark=SURFACE2,
    input_border_color=BORDER,
    input_border_color_dark=BORDER,
    input_placeholder_color=TEXT_MUTED,

    button_primary_background_fill=f"linear-gradient(135deg, {GOLD_DARK} 0%, {GOLD} 50%, {GOLD_DARK} 100%)",
    button_primary_background_fill_dark=f"linear-gradient(135deg, {GOLD_DARK} 0%, {GOLD} 50%, {GOLD_DARK} 100%)",
    button_primary_background_fill_hover=f"linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 50%, {GOLD} 100%)",
    button_primary_text_color=BG,
    button_primary_text_color_dark=BG,
    button_primary_border_color=GOLD_DARK,
    button_primary_border_color_dark=GOLD_DARK,

    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover=f"{GOLD}22",
    button_secondary_border_color=GOLD,
    button_secondary_border_color_dark=GOLD,
    button_secondary_border_color_hover=GOLD_LIGHT,
    button_secondary_text_color=GOLD,
    button_secondary_text_color_dark=GOLD,

    slider_color=GOLD,
    slider_color_dark=GOLD,
    checkbox_background_color=SURFACE2,
    checkbox_background_color_dark=SURFACE2,
    checkbox_border_color=GOLD_DARK,
    checkbox_border_color_dark=GOLD_DARK,
    checkbox_label_background_fill=BG,
    checkbox_label_background_fill_dark=BG,

    accordion_text_color=GOLD,
    accordion_text_color_dark=GOLD,
    table_even_background_fill=BG,
    table_even_background_fill_dark=BG,
    table_odd_background_fill=SURFACE,
    table_odd_background_fill_dark=SURFACE,
    loader_color=GOLD,
    loader_color_dark=GOLD,

    shadow_drop=f"0 8px 32px rgba(0,0,0,0.9), 0 0 0 1px {GOLD}22",
    shadow_spread="0px",
    border_color_primary=GOLD_DARK,
    border_color_primary_dark=GOLD_DARK,

    link_text_color=GOLD,
    link_text_color_dark=GOLD,
    link_text_color_visited=GOLD_DARK,
    link_text_color_visited_dark=GOLD_DARK,

    body_text_color=TEXT,
    body_text_color_dark=TEXT,
    body_text_color_subdued=TEXT_MUTED,
    body_text_color_subdued_dark=TEXT_MUTED,

    color_accent=GOLD,
    color_accent_soft=f"{GOLD}22",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&display=swap');

.gradio-container {{
    background: {BG} !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
    box-sizing: border-box !important;
}}

/* ── Header ── */
#kintsugi-header {{
    text-align: center;
    padding: 48px 0 40px;
    margin-bottom: 32px;
    background: {BG};
    border-radius: 0 0 32px 32px;
    border-bottom: 2px solid {GOLD_DARK};
    box-shadow: 0 10px 40px rgba(0,0,0,0.8), inset 0 1px 0 {GOLD}22;
    position: relative;
    overflow: hidden;
}}

#kintsugi-header::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, {GOLD}08 0%, transparent 70%);
    pointer-events: none;
}}

#kintsugi-header .kintsugi-kanji {{
    font-size: 2.2em;
    color: {GOLD};
    letter-spacing: 0.5em;
    margin: 0 0 8px 0.5em;
    opacity: 0.85;
    font-weight: 300;
    text-shadow: 0 0 30px {GOLD}88;
}}

#kintsugi-header h1 {{
    font-size: 3.8em;
    background: linear-gradient(135deg, {GOLD_DARK} 0%, {GOLD} 40%, {GOLD_LIGHT} 60%, {GOLD} 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: 0.35em;
    font-weight: 400;
    margin: 0;
    text-transform: uppercase;
    filter: drop-shadow(0 0 12px {GOLD}55);
}}

#kintsugi-header .kintsugi-sub {{
    color: {TEXT_MUTED};
    font-style: italic;
    font-size: 1.05em;
    letter-spacing: 0.2em;
    margin: 14px 0 0;
    font-weight: 300;
}}

#kintsugi-header .kintsugi-rule {{
    width: 180px;
    height: 1px;
    background: linear-gradient(90deg, transparent, {GOLD}, transparent);
    margin: 22px auto 0;
    border: none;
}}

/* ── Section labels ── */
.section-label {{
    color: {GOLD} !important;
    font-size: 0.65em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.35em !important;
    margin: 0 0 14px 0 !important;
    padding: 0 0 8px 0 !important;
    border-bottom: 1px solid {GOLD_DARK} !important;
    display: inline-block !important;
    font-weight: 600 !important;
}}

/* ── Restore button ── */
#restore-btn {{
    background: linear-gradient(135deg, {GOLD_DARK} 0%, {GOLD} 50%, {GOLD_DARK} 100%) !important;
    font-size: 1.05em !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    padding: 16px 0 !important;
    border: 1px solid {GOLD_DARK} !important;
    border-radius: 50px !important;
    color: {BG} !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px {GOLD}44 !important;
}}

#restore-btn:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px {GOLD}66 !important;
    background: linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 50%, {GOLD} 100%) !important;
}}

/* ── Demo button ── */
#demo-btn {{
    background: transparent !important;
    border: 1px solid {GOLD_DARK} !important;
    border-radius: 50px !important;
    color: {GOLD} !important;
    font-size: 0.9em !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 12px 0 !important;
    transition: all 0.3s ease !important;
}}

#demo-btn:hover {{
    background: {GOLD}18 !important;
    border-color: {GOLD} !important;
    transform: translateY(-1px) !important;
}}

/* ── Metrics box ── */
#metrics-box {{
    background: {SURFACE} !important;
    border: 1px solid {GOLD_DARK} !important;
    border-left: 3px solid {GOLD} !important;
    border-radius: 12px !important;
    padding: 18px !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.9em !important;
}}

#metrics-box th {{
    color: {GOLD} !important;
    border-bottom: 1px solid {GOLD_DARK} !important;
    padding: 8px !important;
    text-transform: uppercase;
    font-size: 0.75em;
    letter-spacing: 0.15em;
}}

#metrics-box td {{
    color: {TEXT} !important;
    padding: 6px !important;
    text-align: center;
}}

#metrics-box strong {{ color: {GOLD_LIGHT} !important; }}
#metrics-box p      {{ color: {TEXT} !important; margin: 6px 0; }}

/* ── PSF info ── */
#psf-info {{
    background: {SURFACE} !important;
    border: 1px solid {GOLD_DARK} !important;
    border-top: 2px solid {GOLD} !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    font-family: monospace !important;
    font-size: 0.85em !important;
    color: {TEXT_MUTED} !important;
    margin-bottom: 18px !important;
}}

/* ── Image panels ── */
.input-image-panel, .output-image-panel {{
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    transition: border-color 0.25s ease !important;
    background: {SURFACE2} !important;
}}

.input-image-panel:hover  {{ border-color: {GOLD_DARK} !important; }}
.output-image-panel       {{ border-color: {GOLD_DARK} !important; box-shadow: 0 0 24px {GOLD}22 !important; }}

#restored-img {{
    border: 1px solid {GOLD} !important;
    border-radius: 14px !important;
    box-shadow: 0 0 32px {GOLD}33 !important;
}}

/* ── Comparison plot ── */
#comparison-plot {{
    border: 1px solid {GOLD_DARK} !important;
    border-top: 2px solid {GOLD} !important;
    border-radius: 16px !important;
    padding: 10px !important;
    background: {SURFACE} !important;
}}

/* ── Results top bar ── */
#results-topbar {{
    background: {SURFACE2} !important;
    border: 1px solid {GOLD_DARK} !important;
    border-bottom: none !important;
    border-radius: 16px 16px 0 0 !important;
    padding: 14px 24px !important;
    font-size: 0.65em !important;
    color: {GOLD} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.35em !important;
    text-align: center !important;
    font-weight: 500 !important;
}}

/* ── Dividers ── */
.kintsugi-divider      {{ border: none; border-top: 1px solid {BORDER}; margin: 18px 0; }}
.kintsugi-divider-gold {{ border: none; border-top: 1px solid {GOLD_DARK}; margin: 14px 0; }}

/* ── Intro text ── */
#intro-text {{
    text-align: center;
    padding: 14px 0 28px;
    color: {TEXT_MUTED};
    font-style: italic;
    font-size: 1em;
    letter-spacing: 0.07em;
    line-height: 1.8;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 22px;
}}

input[type="range"] {{ accent-color: {GOLD} !important; }}
footer, .built-with  {{ display: none !important; }}
"""

HEADER_HTML = f"""
<div id="kintsugi-header">
  <div class="kintsugi-kanji">金継ぎ</div>
  <h1>✦ KINTSUGI ✦</h1>
  <div class="kintsugi-sub">Golden Restoration · License Plate Deblurring</div>
  <hr class="kintsugi-rule">
</div>
"""

# ─────────────────────────────────────────────────────────────
# Processing helpers
# ─────────────────────────────────────────────────────────────
def _pil_to_gray_float(pil_img) -> np.ndarray:
    return np.array(pil_img.convert("L"), dtype=np.float64) / 255.0


def generate_demo():
    tmpdir = tempfile.mkdtemp()
    generate_synthetic_dataset(output_dir=tmpdir, n_images=1,
                                lengths=[20], angles=[15], noise_std=0.03)
    return (Image.open(os.path.join(tmpdir, "plate_001_blurred.png")),
            Image.open(os.path.join(tmpdir, "plate_001.png")))


def build_comparison_figure(blurred, restored, spectrum, length, angle):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04, wspace=0.08)

    panels = [
        (blurred,  "Blurred",  "gray"),
        (restored, "Restored", "gray"),
        (None,     f"FFT Spectrum\n{length} px  ·  {angle:.1f}°", "inferno"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.set_title(title, color=GOLD, fontsize=13, fontfamily="Georgia", pad=12)
        if img is None:
            s = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-9)
            ax.imshow(s, cmap=cmap)
        else:
            ax.imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig


def process_image(blurred_pil, gt_pil, K, rl_iter):
    if blurred_pil is None:
        return None, None, "*Upload a blurred image first.*", ""

    blurred    = _pil_to_gray_float(blurred_pil)
    method_tag = "Classical (cepstrum + Radon)"

    try:
        est = estimate_psf(blurred, verbose=False)
    except Exception as exc:
        return None, None, f"*Estimation failed: {exc}*", ""

    length, angle, spectrum = est["length"], est["angle"], est["spectrum"]

    try:
        psf      = make_motion_psf(length=length, angle_deg=angle)
        restored = deblur(blurred, psf, K=float(K), rl_iter=int(rl_iter))
    except Exception as exc:
        return None, None, f"*Deblur failed: {exc}*", ""

    fig = build_comparison_figure(blurred, restored, spectrum, length, angle)

    psf_md = (
        f"**PSF estimated** via {method_tag}:  "
        f"length = `{length} px`  ·  angle = `{angle:.1f}°`  |  "
        f"Wiener K = `{K:.4f}`  ·  Richardson-Lucy iters = `{int(rl_iter)}`"
    )

    metrics_md = ""
    if gt_pil is not None:
        gt = _pil_to_gray_float(gt_pil)
        if gt.shape == blurred.shape:
            m  = compute_metrics(gt, blurred, restored)
            ps = "+" if m["psnr_gain"] >= 0 else ""
            ss = "+" if m["ssim_gain"] >= 0 else ""
            metrics_md = (
                "| Metric | Blurred | Restored | Gain |\n"
                "|--------|:-------:|:--------:|:----:|\n"
                f"| PSNR (dB) | {m['blurred_psnr']:.2f} | **{m['restored_psnr']:.2f}** | "
                f"**{ps}{m['psnr_gain']:.2f}** |\n"
                f"| SSIM | {m['blurred_ssim']:.4f} | **{m['restored_ssim']:.4f}** | "
                f"**{ss}{m['ssim_gain']:.4f}** |"
            )
        else:
            metrics_md = "*Ground truth size does not match blurred image.*"
    else:
        metrics_md = "*Upload a ground-truth image to see PSNR / SSIM metrics.*"

    restored_pil = Image.fromarray(
        (np.clip(restored, 0, 1) * 255).astype(np.uint8), mode="L"
    )
    return restored_pil, fig, psf_md, metrics_md


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Kintsugi — License Plate Deblurring") as demo:

    gr.HTML(HEADER_HTML)


    with gr.Row(equal_height=False):

        with gr.Column(scale=1):
            gr.HTML("<div class='section-label'>Input Images</div>")
            blurred_input = gr.Image(label="Blurred License Plate", type="pil",
                                     height=180, elem_classes=["input-image-panel"])
            gt_input = gr.Image(label="Ground Truth  (optional — enables metrics)",
                                type="pil", height=140, elem_classes=["input-image-panel"])

            gr.HTML("<hr class='kintsugi-divider'>")

            gr.HTML("<hr class='kintsugi-divider'>")
            with gr.Accordion("Advanced Parameters", open=False):
                K_slider = gr.Slider(minimum=0.0001, maximum=0.05, value=0.003, step=0.0001,
                                     label="Wiener K  (regularisation)",
                                     info="Lower = sharper but noisier  ·  Higher = smoother")
                rl_slider = gr.Slider(minimum=0, maximum=30, value=15, step=1,
                                      label="Richardson-Lucy Iterations",
                                      info="0 = Wiener only  ·  15 = balanced  ·  30 = maximum")

            gr.HTML("<hr class='kintsugi-divider'>")
            demo_btn    = gr.Button("Generate Demo", variant="secondary", elem_id="demo-btn")
            gr.HTML("<div style='height:8px'></div>")
            restore_btn = gr.Button("✦  RESTORE IMAGE  ✦", variant="primary", elem_id="restore-btn")

        with gr.Column(scale=2):
            gr.HTML("<div id='results-topbar'>✦ &nbsp; RESTORATION RESULTS &nbsp; ✦</div>")

            psf_info_out = gr.Markdown(
                value="*Results will appear here after processing.*",
                elem_id="psf-info",
            )

            gr.HTML("<div class='section-label'>Restored Image</div>")
            restored_out = gr.Image(label="Restored", type="pil", interactive=False,
                                    height=200, elem_id="restored-img",
                                    elem_classes=["output-image-panel"])

            gr.HTML("<hr class='kintsugi-divider-gold'>")
            gr.HTML("<div class='section-label'>Quality Metrics</div>")
            metrics_out = gr.Markdown(
                value="*Upload a ground-truth image to see PSNR / SSIM metrics.*",
                elem_id="metrics-box",
            )

            gr.HTML("<hr class='kintsugi-divider-gold'>")
            comparison_plot = gr.Plot(
                label="Blurred  ·  Restored  ·  FFT Spectrum",
                show_label=True, elem_id="comparison-plot",
            )

    gr.HTML(f"""
    <div style="text-align:center; padding:26px 0 14px;
                border-top:1px solid {GOLD_DARK}; margin-top:32px;">
        <span style="color:{TEXT_MUTED}; font-size:0.65em; letter-spacing:0.25em;">
            ✦ KINTSUGI · Computer Vision · ESIN UIR Spring 2026 ✦
        </span>
    </div>
    """)

    demo_btn.click(fn=generate_demo, inputs=[], outputs=[blurred_input, gt_input])
    restore_btn.click(fn=process_image,
                      inputs=[blurred_input, gt_input, K_slider, rl_slider],
                      outputs=[restored_out, comparison_plot, psf_info_out, metrics_out])


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "─" * 60)
    print("  ✦  KINTSUGI  ✦   License Plate Deblurring  ✦")
    print("─" * 60)
    print("  Opening at  http://127.0.0.1:7860")
    print("─" * 60 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                show_error=True, quiet=False, theme=theme, css=CSS)
