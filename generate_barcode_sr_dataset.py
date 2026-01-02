#!/usr/bin/env python3
"""
Synthetic barcode dataset generator (SR-focused, but also outputs barcode polygon labels).

Outputs:
  out_dir/
    hr/00000001.png       # clean high-res (HR)
    lr/00000001.png       # degraded low-res (native LR resolution)
    lr_up/00000001.png    # LR resized back to HR size (typical SR network input)
    meta.jsonl            # JSON lines with symbology, payload, polygon, and transform params

Supported symbologies (auto-skips if optional deps missing):
  - code128, ean13, upca (via python-barcode)
  - qrcode (via qrcode)
  - datamatrix (via pylibdmtx)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Pillow resampling compatibility
if hasattr(Image, "Resampling"):
    R_NEAREST = Image.Resampling.NEAREST
    R_BILINEAR = Image.Resampling.BILINEAR
    R_BICUBIC = Image.Resampling.BICUBIC
    R_LANCZOS = Image.Resampling.LANCZOS
else:
    R_NEAREST = Image.NEAREST
    R_BILINEAR = Image.BILINEAR
    R_BICUBIC = Image.BICUBIC
    R_LANCZOS = Image.LANCZOS

# Optional deps
try:
    import barcode  # python-barcode
    from barcode.writer import ImageWriter
except Exception:
    barcode = None
    ImageWriter = None

try:
    import qrcode
except Exception:
    qrcode = None

try:
    from pylibdmtx.pylibdmtx import encode as dmtx_encode
except Exception:
    dmtx_encode = None

try:
    import cv2  # opencv-python
except Exception:
    cv2 = None


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def clamp_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def rand_alnum(rng: random.Random, n: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(alphabet) for _ in range(n))


def random_payload(sym: str, rng: random.Random) -> str:
    # python-barcode auto-calculates check digits for EAN/UPC family (you pass the body).
    if sym == "ean13":
        return rand_digits(rng, 12)
    if sym == "upca":
        return rand_digits(rng, 11)
    if sym == "code128":
        # Code128 can carry broader ASCII, but keep it simple and scanner-friendly
        return rand_alnum(rng, rng.randint(8, 24))
    if sym == "qrcode":
        # Mix URLs and simple tokens
        if rng.random() < 0.5:
            return f"https://example.com/{rand_alnum(rng, rng.randint(6, 12))}"
        return f"ID:{rand_alnum(rng, rng.randint(10, 24))}"
    if sym == "datamatrix":
        return f"DM:{rand_alnum(rng, rng.randint(10, 40))}"
    raise ValueError(f"Unknown symbology: {sym}")


def render_1d_python_barcode(sym: str, data: str, rng: random.Random) -> Image.Image:
    if barcode is None or ImageWriter is None:
        raise RuntimeError(
            "python-barcode not installed (pip install 'python-barcode[images]')"
        )

    bc_class = barcode.get_barcode_class(sym)
    writer = ImageWriter()

    # Random but controlled styling
    # Keep high contrast most of the time; occasionally invert for "light-on-dark"
    invert = rng.random() < 0.15
    if invert:
        fg = (245, 245, 245)
        bg = (20, 20, 20)
    else:
        fg = (rng.randint(0, 20), rng.randint(0, 20), rng.randint(0, 20))
        base = rng.randint(235, 255)
        bg = (base, base, base)

    # Choose module_width so mm->px conversion is less "fractional" at 300 dpi
    # 0.254 mm == 0.01 inch; at 300 dpi that's ~3 px per module
    module_width_mm = rng.choice([0.254, 0.3, 0.35])
    module_height_mm = rng.uniform(8.0, 22.0)
    quiet_zone_mm = rng.uniform(2.0, 8.0)

    show_text = rng.random() < 0.25
    font_size = rng.randint(8, 12) if show_text else 0

    writer_options = {
        "module_width": float(module_width_mm),
        "module_height": float(module_height_mm),
        "quiet_zone": float(quiet_zone_mm),
        "font_size": int(font_size),
        "text_distance": float(rng.uniform(1.0, 5.0)),
        "background": rgb_to_hex(bg),
        "foreground": rgb_to_hex(fg),
        "format": "PNG",
        "dpi": int(rng.choice([200, 300, 400])),
    }

    bc = bc_class(data, writer=writer)
    rendered = bc.render(writer_options=writer_options)

    # python-barcode typically returns a PIL Image for ImageWriter,
    # but keep a fallback in case it returns bytes-like.
    if isinstance(rendered, Image.Image):
        img = rendered
    else:
        bio = BytesIO(rendered)
        img = Image.open(bio)

    return img.convert("RGB")


def render_qr(data: str, rng: random.Random) -> Image.Image:
    if qrcode is None:
        raise RuntimeError("qrcode not installed (pip install 'qrcode[pil]')")

    # Error correction levels: L/M/Q/H
    ec = rng.choice(
        [
            qrcode.constants.ERROR_CORRECT_L,
            qrcode.constants.ERROR_CORRECT_M,
            qrcode.constants.ERROR_CORRECT_Q,
            qrcode.constants.ERROR_CORRECT_H,
        ]
    )
    box_size = rng.randint(4, 14)
    border = rng.choice([4, 4, 4, 2, 6])  # mostly 4

    # Mostly black-on-light; sometimes invert
    invert = rng.random() < 0.12
    if invert:
        fill = (250, 250, 250)
        back = (15, 15, 15)
    else:
        fill = (0, 0, 0)
        back = (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255))

    qr = qrcode.QRCode(
        version=None,
        error_correction=ec,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill, back_color=back)
    return img.convert("RGB")


def render_datamatrix(data: str, rng: random.Random) -> Image.Image:
    if dmtx_encode is None:
        raise RuntimeError("pylibdmtx not installed or libdmtx missing")

    encoded = dmtx_encode(data.encode("utf-8"))
    img = Image.frombytes("RGB", (encoded.width, encoded.height), encoded.pixels)

    # Enlarge modules with nearest neighbor to keep edges crisp
    scale = rng.randint(2, 8)
    img = img.resize((img.width * scale, img.height * scale), resample=R_NEAREST)

    # Occasionally invert
    if rng.random() < 0.12:
        img = ImageOps.invert(img)

    return img.convert("RGB")


def render_barcode(sym: str, data: str, rng: random.Random) -> Image.Image:
    if sym in {"code128", "ean13", "upca"}:
        return render_1d_python_barcode(sym, data, rng)
    if sym == "qrcode":
        return render_qr(data, rng)
    if sym == "datamatrix":
        return render_datamatrix(data, rng)
    raise ValueError(sym)


def make_background(
    w: int, h: int, rng: random.Random, np_rng: np.random.Generator
) -> Image.Image:
    """
    Simple synthetic backgrounds: solid, gradient, paper-ish noise.
    """
    mode = rng.choices(
        ["solid", "gradient", "paper_noise", "dark_solid"],
        weights=[0.45, 0.25, 0.25, 0.05],
        k=1,
    )[0]

    if mode == "solid":
        base = rng.randint(230, 255)
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        return np_rgb_to_pil(arr)

    if mode == "dark_solid":
        base = rng.randint(10, 40)
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        return np_rgb_to_pil(arr)

    if mode == "gradient":
        start = rng.randint(210, 255)
        end = rng.randint(210, 255)
        if rng.random() < 0.5:
            # vertical
            g = np.linspace(start, end, h, dtype=np.float32)[:, None]
            arr = np.repeat(g, w, axis=1)
        else:
            # horizontal
            g = np.linspace(start, end, w, dtype=np.float32)[None, :]
            arr = np.repeat(g, h, axis=0)
        rgb = np.stack([arr, arr, arr], axis=-1)
        rgb = clamp_uint8(rgb)
        return np_rgb_to_pil(rgb)

    # "paper_noise"
    base = rng.randint(220, 255)
    noise = np_rng.normal(0, rng.uniform(2.0, 10.0), size=(h, w, 1)).astype(np.float32)
    arr = np.full((h, w, 3), base, dtype=np.float32)
    arr += noise
    arr = clamp_uint8(arr)

    img = np_rgb_to_pil(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    return img


def maybe_perspective_warp(
    patch: Image.Image,
    rng: random.Random,
    max_warp_frac: float = 0.18,
) -> Tuple[Image.Image, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      warped_patch: PIL RGB
      poly: (4,2) float32 corners in patch coordinate system
      H: 3x3 homography if cv2 available, else None
    """
    w, h = patch.size
    poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    if cv2 is None or rng.random() > 0.70:
        # No warp (or OpenCV missing)
        return patch, poly, None

    # Jitter corners
    jx = max_warp_frac * w
    jy = max_warp_frac * h
    jitter = np.array(
        [[rng.uniform(-jx, jx), rng.uniform(-jy, jy)] for _ in range(4)],
        dtype=np.float32,
    )

    dst = poly + jitter

    # Shift into positive region with padding
    pad = max(4.0, 0.02 * max(w, h))
    min_xy = dst.min(axis=0)
    dst_shift = dst - min_xy + pad

    out_w = int(math.ceil(dst_shift[:, 0].max() + pad))
    out_h = int(math.ceil(dst_shift[:, 1].max() + pad))

    bg = patch.getpixel((0, 0))
    border_val = tuple(int(c) for c in bg)

    src = poly.copy()
    H = cv2.getPerspectiveTransform(src, dst_shift)

    arr = pil_to_np_rgb(patch)
    warped = cv2.warpPerspective(
        arr,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    warped_img = np_rgb_to_pil(warped)
    return warped_img, dst_shift, H


def fit_patch_to_canvas(
    patch: Image.Image,
    poly: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    rng: random.Random,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Resize patch to be small/medium within canvas and scale polygon accordingly.
    """
    w, h = patch.size

    # Barcodes can be small in the image; bias to smaller sizes
    # target area fraction relative to canvas:
    area_frac = rng.choice([0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
    target_area = area_frac * canvas_w * canvas_h
    target_scale = math.sqrt(max(1.0, target_area / max(1.0, (w * h))))

    # Clamp scale so it doesn't get ridiculous
    target_scale = float(np.clip(target_scale, 0.15, 1.2))

    new_w = max(8, int(round(w * target_scale)))
    new_h = max(8, int(round(h * target_scale)))

    # Ensure the patch fits in the canvas (with a small margin)
    max_w = int(canvas_w * 0.98)
    max_h = int(canvas_h * 0.98)

    if new_w > max_w:
        ratio = max_w / new_w
        new_w = max_w
        new_h = max(8, int(new_h * ratio))

    if new_h > max_h:
        ratio = max_h / new_h
        new_h = max_h
        new_w = max(8, int(new_w * ratio))

    resample = R_LANCZOS if (new_w < w or new_h < h) else R_NEAREST
    patch2 = patch.resize((new_w, new_h), resample=resample)

    scale_xy = np.array([new_w / w, new_h / h], dtype=np.float32)
    poly2 = poly * scale_xy[None, :]

    return patch2, poly2


def paste_on_canvas(
    canvas: Image.Image,
    patch: Image.Image,
    patch_poly: np.ndarray,
    rng: random.Random,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Paste patch onto canvas at random location (fully inside). Return updated polygon in canvas coords.
    """
    cw, ch = canvas.size
    pw, ph = patch.size

    x0 = rng.randint(0, max(0, cw - pw))
    y0 = rng.randint(0, max(0, ch - ph))

    canvas2 = canvas.copy()
    canvas2.paste(patch, (x0, y0))

    poly_canvas = patch_poly + np.array([x0, y0], dtype=np.float32)
    return canvas2, poly_canvas


def apply_motion_blur(
    img: Image.Image, rng: random.Random
) -> Tuple[Image.Image, Dict[str, Any]]:
    if cv2 is None:
        # Skip if OpenCV isn't available
        return img, {
            "name": "motion_blur",
            "skipped": True,
            "reason": "cv2_not_installed",
        }

    k = rng.choice([3, 5, 7, 9, 11, 13, 15])
    angle = rng.uniform(0, 180)

    # Line kernel then rotate it
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s

    arr = pil_to_np_rgb(img)
    blurred = cv2.filter2D(arr, -1, kernel)
    return np_rgb_to_pil(blurred), {
        "name": "motion_blur",
        "ksize": k,
        "angle_deg": angle,
    }


def degrade_for_sr(
    hr_img: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]], int]:
    """
    Create LR + LR_up from HR and record transform params.
    """
    ops: List[Dict[str, Any]] = []
    img = hr_img

    # Mild photometric jitter on HR (optional): keep this light so HR stays "clean"
    if rng.random() < 0.20:
        b = rng.uniform(0.92, 1.08)
        c = rng.uniform(0.92, 1.10)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        ops.append({"name": "hr_brightness_contrast", "brightness": b, "contrast": c})

    # Blur before downscale (camera lens / motion)
    if rng.random() < 0.65:
        if rng.random() < 0.6:
            r = rng.uniform(0.3, 1.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
            ops.append({"name": "gaussian_blur", "radius": r})
        else:
            img, meta = apply_motion_blur(img, rng)
            ops.append(meta)

    # Downscale
    w, h = img.size
    scale = rng.choice([2, 3, 4, 6, 8])
    lw = max(8, w // scale)
    lh = max(8, h // scale)

    down_method = rng.choice(["lanczos", "bilinear", "bicubic"])
    resample = {"lanczos": R_LANCZOS, "bilinear": R_BILINEAR, "bicubic": R_BICUBIC}[
        down_method
    ]
    lr = img.resize((lw, lh), resample=resample)
    ops.append(
        {
            "name": "downscale",
            "scale": scale,
            "lr_size": [lw, lh],
            "method": down_method,
        }
    )

    # Noise on LR
    if rng.random() < 0.75:
        sigma = rng.uniform(0.0, 12.0)  # in pixel space (0..255)
        arr = pil_to_np_rgb(lr).astype(np.float32)
        noise = np_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr2 = clamp_uint8(arr + noise)
        lr = np_rgb_to_pil(arr2)
        ops.append({"name": "gaussian_noise", "sigma_px": sigma})

    # Random occlusion (scratches / dirt)
    if rng.random() < 0.20:
        arr = pil_to_np_rgb(lr)
        hh, ww = arr.shape[:2]
        x1 = rng.randint(0, ww - 1)
        y1 = rng.randint(0, hh - 1)
        x2 = rng.randint(0, ww - 1)
        y2 = rng.randint(0, hh - 1)
        thickness = rng.choice([1, 1, 2, 3])
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if cv2 is not None:
            cv2.line(arr, (x1, y1), (x2, y2), color, thickness=thickness)
        else:
            # crude fallback: draw a thick line by stamping points
            for t in np.linspace(0, 1, num=200):
                xx = int(round(x1 + t * (x2 - x1)))
                yy = int(round(y1 + t * (y2 - y1)))
                x0 = max(0, xx - thickness)
                y0 = max(0, yy - thickness)
                x3 = min(ww, xx + thickness + 1)
                y3 = min(hh, yy + thickness + 1)
                arr[y0:y3, x0:x3] = color
        lr = np_rgb_to_pil(arr)
        ops.append({"name": "occlusion_line", "thickness": thickness})

    # JPEG compression artifact simulation on LR
    if rng.random() < 0.80:
        q = rng.randint(25, 95)
        buf = BytesIO()
        lr.save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
        buf.seek(0)
        lr = Image.open(buf).convert("RGB")
        ops.append({"name": "jpeg", "quality": q})

    # Upsample back to HR size (network input usually matches GT size)
    up_method = rng.choice(["bicubic", "bilinear"])
    up_resample = R_BICUBIC if up_method == "bicubic" else R_BILINEAR
    lr_up = lr.resize((w, h), resample=up_resample)
    ops.append({"name": "upsample", "to": [w, h], "method": up_method})

    return lr, lr_up, ops, scale


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument(
        "--hr_size", type=int, default=512, help="HR canvas is hr_size x hr_size"
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--barcode_type",
        type=str,
        default="1d",
        choices=["1d", "2d", "all"],
        help="Type of barcodes to generate (default: 1d)",
    )
    ap.add_argument(
        "--symbologies",
        type=str,
        default=None,
        help="Comma-separated. Auto-skips missing deps. Overrides --barcode_type.",
    )
    ap.add_argument(
        "--generate_config", action="store_true", help="Generate a training config file"
    )
    ap.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Path to pretrained model weights",
    )
    ap.add_argument(
        "--config_out",
        type=str,
        default="train_config.yml",
        help="Output path for config file",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    hr_dir = out_dir / "hr"
    lr_dir = out_dir / "lr"
    lr_up_dir = out_dir / "lr_up"
    ensure_dir(hr_dir)
    ensure_dir(lr_dir)
    ensure_dir(lr_up_dir)

    meta_path = out_dir / "meta.jsonl"

    rng = random.Random(args.seed)

    if args.symbologies is not None:
        requested = [
            s.strip().lower() for s in args.symbologies.split(",") if s.strip()
        ]
    else:
        # Defaults based on type
        SYM_1D = ["code128", "ean13", "upca"]
        SYM_2D = ["qrcode", "datamatrix"]

        if args.barcode_type == "1d":
            requested = SYM_1D
        elif args.barcode_type == "2d":
            requested = SYM_2D
        else:  # all
            requested = SYM_1D + SYM_2D

    available: List[str] = []

    for s in requested:
        if s in {"code128", "ean13", "upca"} and barcode is None:
            continue
        if s == "qrcode" and qrcode is None:
            continue
        if s == "datamatrix" and dmtx_encode is None:
            continue
        available.append(s)

    if not available:
        raise SystemExit(
            f"No symbologies available for {args.barcode_type} (requested: {requested}). Install deps: "
            "pip install numpy pillow 'python-barcode[images]' 'qrcode[pil]' opencv-python pylibdmtx"
        )

    with meta_path.open("w", encoding="utf-8") as f:
        for i in range(1, args.num_samples + 1):
            sample_seed = rng.randint(0, 2**31 - 1)
            sample_rng = random.Random(sample_seed)
            sample_np_rng = np.random.default_rng(sample_seed)

            sym = sample_rng.choice(available)
            payload = random_payload(sym, sample_rng)

            # Render barcode patch
            patch = render_barcode(sym, payload, sample_rng)

            # Optional small border pad around patch (quiet zone / label margin)
            pad_px = sample_rng.randint(6, 24)
            bg = patch.getpixel((0, 0))
            patch = ImageOps.expand(patch, border=pad_px, fill=bg)

            # Perspective warp (optional)
            patch_warp, poly_patch, H = maybe_perspective_warp(patch, sample_rng)

            # HR canvas
            S = int(args.hr_size)
            bg_img = make_background(S, S, sample_rng, sample_np_rng)

            # Resize patch to be small/medium within canvas
            patch_fit, poly_fit = fit_patch_to_canvas(
                patch_warp, poly_patch, S, S, sample_rng
            )

            # Paste onto canvas and get polygon
            hr_img, poly_hr = paste_on_canvas(bg_img, patch_fit, poly_fit, sample_rng)

            # Degrade to LR + LR_up
            lr_img, lr_up_img, ops, scale = degrade_for_sr(
                hr_img, sample_rng, sample_np_rng
            )

            # Save images
            stem = f"{i:08d}"
            hr_path = hr_dir / f"{stem}.png"
            lr_path = lr_dir / f"{stem}.png"
            lr_up_path = lr_up_dir / f"{stem}.png"

            hr_img.save(hr_path)
            lr_img.save(lr_path)
            lr_up_img.save(lr_up_path)

            # Labels / metadata
            rec: Dict[str, Any] = {
                "id": stem,
                "seed": sample_seed,
                "symbology": sym,
                "payload": payload,
                "hr_path": str(hr_path.relative_to(out_dir)),
                "lr_path": str(lr_path.relative_to(out_dir)),
                "lr_up_path": str(lr_up_path.relative_to(out_dir)),
                "hr_size": [S, S],
                "lr_scale": scale,
                "barcode_polygon_hr": poly_hr.round(3).tolist(),  # 4 points (x,y)
                "transforms": ops,
                "opencv_used_for_warp": (cv2 is not None) and (H is not None),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {args.num_samples} samples to: {out_dir}")
    print(f"Manifest: {meta_path}")

    if args.generate_config:
        generate_yaml_config(
            out_dir=out_dir,
            config_out=Path(args.config_out),
            pretrained_path=args.pretrained_weights,
            scale=2,  # Assuming scale 2 for now based on default degrader
            hr_size=args.hr_size,
        )


def generate_yaml_config(
    out_dir: Path,
    config_out: Path,
    pretrained_path: Optional[str],
    scale: int = 2,
    hr_size: int = 512,
) -> None:
    """
    Generates a BasicSR/SPAN compatible YAML config for fine-tuning.
    """
    abs_out_dir = out_dir.resolve()
    hr_path = abs_out_dir / "hr"
    lr_path = abs_out_dir / "lr"

    # Default template based on EDSR/SPAN structure
    yaml_content = f"""# Auto-generated fine-tuning config
name: FineTune_SPAN_S{scale}_{out_dir.name}
model_type: SRModel
scale: {scale}
num_gpu: 1
manual_seed: 10

# Dataset settings
datasets:
  train:
    name: CustomBarcode
    type: PairedImageDataset
    dataroot_gt: {hr_path}
    dataroot_lq: {lr_path}
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

    gt_size: 96
    use_hflip: true
    use_rot: true

    # DataLoader
    num_worker_per_gpu: 4
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: CustomBarcodeVal
    type: PairedImageDataset
    dataroot_gt: {hr_path}
    dataroot_lq: {lr_path}
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

# Network settings
network_g:
  type: SPAN
  num_in_ch: 3
  num_out_ch: 3
  upscale: {scale}
  # SPAN specific defaults (adjust if needed)
  num_feat: 48
  num_block: 6  # Check your pretrained model structure!

path:
  pretrain_network_g: {pretrained_path if pretrained_path else "~"}
  strict_load_g: true
  resume_state: ~

# Training settings
train:
  optim_g:
    type: Adam
    lr: !!float 5e-5
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [25000, 50000, 75000]
    gamma: 0.5

  total_iter: 100000
  warmup_iter: -1

  # Losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# Validation settings
val:
  val_freq: !!float 5e3
  save_img: false

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: {scale}
      test_y_channel: false

# Logging
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# Distributed settings
dist_params:
  backend: nccl
  port: 29500
"""

    ensure_dir(config_out.parent)
    with open(config_out, "w") as f:
        f.write(yaml_content)

    print(f"Generated training config at: {config_out}")
    print(
        f"  - Pretrained weights: {pretrained_path if pretrained_path else 'None (Training from scratch)'}"
    )
    print(f"  - Data root: {abs_out_dir}")


if __name__ == "__main__":
    main()
