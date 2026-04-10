#!/usr/bin/env python3
"""
create_dataset.py  –  AI Vision LLM YOLO Dataset Creator
=========================================================
Automatically generates YOLO-format annotations for images in
dataset/images/train/ by querying a local Vision LLM (LM Studio).

Pipeline per image:
  1. Encode image as base64 and send to LM Studio Vision LLM
  2. Parse the JSON response (detections with normalized bbox corners)
  3. Save raw JSON to  debug/<name>.json            (overwritten on re-run)
  4. Draw bounding boxes on the image, save as debug/<name>_lmm_vision.<ext>
  5. Convert JSON → YOLO format, save to  labels/train/<name>.txt
  6. Write classes.txt + dataset.yaml to dataset/

Author : AI Vision LLM YOLO Dataset
"""

import os
import re
import sys
import json
import time
import base64
import logging
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# ============================================================
# CONFIGURATION  –  edit these values as needed
# ============================================================

# --- LM Studio endpoint ---
LM_STUDIO_HOST        = "192.168.2.111"
LM_STUDIO_PORT        = 1234
LM_STUDIO_MODEL       = "zai-org/glm-4.6v-flash"
LM_STUDIO_TEMPERATURE = 0.1
LM_STUDIO_TIMEOUT     = 120          # seconds per request

# --- System prompt ---
SYSTEM_PROMPT_FILE    = "llm_prompt.txt"

# --- Dataset paths ---
DATASET_ROOT          = Path("dataset")
IMAGES_TRAIN_DIR      = DATASET_ROOT / "images" / "train"
IMAGES_VAL_DIR        = DATASET_ROOT / "images" / "val"
LABELS_TRAIN_DIR      = DATASET_ROOT / "labels" / "train"
LABELS_VAL_DIR        = DATASET_ROOT / "labels" / "val"
DEBUG_DIR             = DATASET_ROOT / "debug"   # JSON + annotated previews (overwritten on re-run)

# --- Log file ---
LOG_FILE              = "dataset.log"            # written next to the script

# --- Image formats to process (all formats supported by YOLOv8) ---
SUPPORTED_EXTENSIONS  = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# --- Bounding box drawing settings ---
BOX_COLOR             = (255, 255, 255)  # RGB: bounding box outline colour (white)
BOX_THICKNESS         = 3               # outline thickness in pixels
FONT_SIZE             = 16              # label font size in points
LABEL_PADDING         = 3              # pixels of padding inside the label chip
LABEL_BG_COLOR        = (0, 0, 0)      # RGB: label background colour (black)
LABEL_BG_ALPHA        = 200            # 0–255: label background transparency
LABEL_TEXT_COLOR      = (0, 255, 0)    # RGB: label text colour (green)

# --- NMS (Non-Maximum Suppression) settings ---
IOU_THRESHOLD         = 0.80         # overlap threshold; boxes above this are merged
NMS_MODE              = "per_label"  # "per_label" | "global" | "off"


# ============================================================
# FOLDER STRUCTURE
# ============================================================

def create_folder_structure() -> None:
    """Create the standard YOLO dataset folder structure if not yet present."""
    for directory in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR,
                      LABELS_TRAIN_DIR, LABELS_VAL_DIR,
                      DEBUG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR] {directory}")


# ============================================================
# LOGGING
# ============================================================

def setup_logger() -> logging.Logger:
    """
    Configure a logger that writes to both the console and LOG_FILE.
    The log file is appended to on every run so history is preserved.
    Format:  2024-01-15 10:23:45 | INFO    | <message>
    """
    logger = logging.getLogger("dataset")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler – append mode keeps history across multiple runs
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler – INFO and above only (keeps terminal readable)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_image_info(image_path: Path) -> dict:
    """
    Read basic metadata from an image file without fully decoding it.
    Returns a dict with keys: file_size_kb, pil_format, resolution.
    On failure returns safe fallback values.
    """
    info = {"file_size_kb": 0.0, "pil_format": "unknown", "resolution": "unknown"}
    try:
        info["file_size_kb"] = round(image_path.stat().st_size / 1024, 1)
        with Image.open(image_path) as img:
            info["pil_format"]  = img.format or image_path.suffix.upper().lstrip(".")
            info["resolution"]  = f"{img.width}x{img.height}"
    except Exception:
        pass
    return info


# ============================================================
# LLM  –  QUERY & PARSE
# ============================================================

def load_system_prompt(filepath: str) -> str:
    """Load system prompt text from an external file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode image to base64 for the LM Studio API.
    Returns (base64_string, mime_type).

    JPEG and PNG are sent as-is; all other formats (WEBP, BMP, TIFF, …)
    are converted in-memory to JPEG first.  The LM Studio API rejects
    formats like WEBP/BMP/TIFF with a 400 error even though the GUI
    handles them fine — this conversion fixes that without touching the
    original file on disk.
    """
    import io

    # Formats the API accepts natively
    PASSTHROUGH = {".jpg", ".jpeg", ".png"}

    ext = image_path.suffix.lower()

    if ext in PASSTHROUGH:
        # Send original bytes directly
        with open(image_path, "rb") as fh:
            raw = fh.read()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    else:
        # Convert to JPEG in-memory so the API always receives a known format
        buf = io.BytesIO()
        with Image.open(image_path) as img:
            img.convert("RGB").save(buf, format="JPEG", quality=95)
        raw  = buf.getvalue()
        mime = "image/jpeg"

    return base64.b64encode(raw).decode("utf-8"), mime


def query_llm(image_path: Path, system_prompt: str) -> str:
    """
    Send image + system prompt to the LM Studio Vision LLM.
    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    Returns the raw text content of the model response.
    """
    api_url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"
    b64_image, mime_type = encode_image_base64(image_path)

    payload = {
        "model": LM_STUDIO_MODEL,
        "temperature": LM_STUDIO_TEMPERATURE,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and return bounding box "
                            "detections as instructed in the system prompt."
                        )
                    }
                ]
            }
        ]
    }

    response = requests.post(api_url, json=payload, timeout=LM_STUDIO_TIMEOUT)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def parse_llm_response(raw: str) -> dict:
    """
    Extract valid JSON from the LLM response.
    The model may wrap output in a markdown ```json ... ``` code block.
    Raises ValueError when no usable JSON is found.
    """
    # Primary: extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Fallback: find the outermost { ... } in the response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No valid JSON object found in LLM response:\n{raw[:500]}")


# ============================================================
# BOUNDING BOX  –  HELPERS
# ============================================================

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the range [lo, hi]."""
    return max(lo, min(hi, value))


def iou(box_a: tuple, box_b: tuple) -> float:
    """
    Compute Intersection over Union for two (x1, y1, x2, y2) pixel boxes.
    Returns a float in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def apply_nms(detections: list, iou_thr: float = 0.8,
              mode: str = "per_label") -> list:
    """
    Apply Non-Maximum Suppression to remove highly-overlapping boxes.

    mode:
      "per_label"  –  only suppress boxes with the same label
      "global"     –  suppress regardless of label
      "off"        –  no suppression
    """
    if mode == "off":
        return detections

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in dets:
        suppress = False
        for k in kept:
            if mode == "per_label" and k["label"] != det["label"]:
                continue
            if iou(det["_box"], k["_box"]) > iou_thr:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


# ============================================================
# BOUNDING BOX  –  DRAWING
# ============================================================

def draw_bounding_boxes(image_path: Path, detections: list,
                        output_path: Path) -> None:
    """
    Draw bounding boxes + semi-transparent label chips on the image.
    Saves the result to output_path.

    detections entries must contain:
      label       : str
      confidence  : float
      bbox_norm   : {x1, y1, x2, y2}  (normalized 0–1 corner format)
    """
    img_rgb = Image.open(image_path).convert("RGB")
    W, H = img_rgb.size

    # Load font (falls back to PIL default when Arial is not installed)
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Convert normalized coords → pixel coords and apply NMS
    pixel_dets = []
    for det in detections:
        bn = det.get("bbox_norm") or {}
        try:
            x1 = int(float(bn["x1"]) * W)
            y1 = int(float(bn["y1"]) * H)
            x2 = int(float(bn["x2"]) * W)
            y2 = int(float(bn["y2"]) * H)
        except (KeyError, TypeError, ValueError):
            continue

        # Clamp to image bounds
        x1 = clamp(x1, 0, W - 1)
        y1 = clamp(y1, 0, H - 1)
        x2 = clamp(x2, 0, W - 1)
        y2 = clamp(y2, 0, H - 1)

        # Fix inverted coordinates
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Skip degenerate boxes
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue

        pixel_dets.append({
            "label":      str(det.get("label", "unknown")),
            "confidence": float(det.get("confidence", 0.0)),
            "_box":       (x1, y1, x2, y2)
        })

    # NMS: remove highly-overlapping duplicate boxes
    pixel_dets = apply_nms(pixel_dets, iou_thr=IOU_THRESHOLD, mode=NMS_MODE)

    # --- Drawing phase ---
    # Use RGBA so we can composite a semi-transparent label background
    img_rgba = img_rgb.convert("RGBA")
    overlay  = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw_ov  = ImageDraw.Draw(overlay)
    draw     = ImageDraw.Draw(img_rgba)

    for det in pixel_dets:
        label = det["label"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = det["_box"]
        txt = f"{label} {conf:.2f}"

        # Draw bounding box outline (multi-pass for thickness)
        box_rgba = (*BOX_COLOR, 255)
        for t in range(BOX_THICKNESS):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=box_rgba
            )

        # Compute label chip size
        tb  = draw.textbbox((0, 0), txt, font=font)
        tw  = tb[2] - tb[0]
        th  = tb[3] - tb[1]
        pad = LABEL_PADDING

        bx1 = x1
        by1 = max(0, y1 - (th + pad * 2))
        bx2 = clamp(x1 + tw + pad * 2, 0, W)
        by2 = clamp(by1 + th + pad * 2, 0, H)

        # Semi-transparent label background chip
        draw_ov.rectangle(
            [bx1, by1, bx2, by2],
            fill=(*LABEL_BG_COLOR, LABEL_BG_ALPHA)
        )
        # Label text
        draw_ov.text(
            (bx1 + pad, by1 + pad),
            txt, font=font, fill=(*LABEL_TEXT_COLOR, 255)
        )

    # Composite overlay onto image
    result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
    result.save(output_path, quality=95)
    print(f"  [IMG]  Annotated image saved  → {output_path}")


# ============================================================
# YOLO LABEL CONVERSION
# ============================================================

def update_class_map(detections: list, class_map: dict) -> None:
    """
    Register any new labels from detections into class_map.
    class_map maps label (str) → class_id (int).
    Modifies class_map in place.
    """
    for det in detections:
        label = det.get("label", "unknown")
        if label not in class_map:
            class_map[label] = len(class_map)


def detections_to_yolo(detections: list, class_map: dict) -> list[str]:
    """
    Convert a list of detections to YOLO annotation lines.

    Input bbox_norm format  : x1, y1, x2, y2  (normalized corner coords)
    Output YOLO format      : class_id cx cy w h  (normalized center coords)

    The LLM returns corner-based coords; YOLO expects center-based coords.
    Conversion:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
    """
    lines = []
    for det in detections:
        label    = det.get("label", "unknown")
        bbox     = det.get("bbox_norm") or {}
        class_id = class_map.get(label, 0)

        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        # Convert corners → YOLO center format
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = x2 - x1
        h  = y2 - y1

        # Clamp all values to [0.0, 1.0]
        cx = clamp(cx, 0.0, 1.0)
        cy = clamp(cy, 0.0, 1.0)
        w  = clamp(w,  0.0, 1.0)
        h  = clamp(h,  0.0, 1.0)

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def save_dataset_metadata(class_map: dict, output_dir: Path) -> None:
    """
    Write classes.txt and dataset.yaml to output_dir.
    These files are required for YOLO training.
    """
    sorted_classes = sorted(class_map.items(), key=lambda kv: kv[1])
    names_list     = [label for label, _ in sorted_classes]

    # classes.txt  –  one label per line, ordered by class_id
    classes_txt = output_dir / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(names_list) + "\n")
    print(f"  [CLS]  classes.txt saved       → {classes_txt}")

    # dataset.yaml  –  standard YOLO training config
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("# YOLO dataset configuration\n")
        fh.write(f"path:  {DATASET_ROOT.resolve()}\n")
        fh.write(f"train: images/train\n")
        fh.write(f"val:   images/val\n\n")
        fh.write(f"nc:    {len(names_list)}\n")
        fh.write(f"names: {names_list}\n")
    print(f"  [YAML] dataset.yaml saved      → {yaml_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_images() -> None:
    """
    Full pipeline:
      1. Create YOLO folder structure
      2. Load system prompt
      3. Find all images in images/train/
      4. For each image: query LLM → save JSON → draw boxes → save YOLO .txt
      5. Write classes.txt + dataset.yaml
    """
    log = setup_logger()
    log.info("=" * 60)
    log.info(f"Session started  |  model: {LM_STUDIO_MODEL}")
    log.info("=" * 60)

    # ----- Step 1: Folder structure -----
    log.info("Creating YOLO folder structure")
    create_folder_structure()

    # ----- Step 2: System prompt -----
    log.info(f"Loading system prompt from '{SYSTEM_PROMPT_FILE}'")
    if not os.path.isfile(SYSTEM_PROMPT_FILE):
        log.error(f"'{SYSTEM_PROMPT_FILE}' not found — create the file and add your LLM prompt.")
        sys.exit(1)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    log.info(f"System prompt loaded ({len(system_prompt)} chars)")

    # ----- Step 3: Discover images -----
    image_files = sorted([
        f for f in IMAGES_TRAIN_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        log.warning(f"No images found in '{IMAGES_TRAIN_DIR}' — place source images there and re-run.")
        return

    log.info(f"Found {len(image_files)} image(s) to process")

    class_map: dict[str, int] = {}   # label → class_id (accumulated over run)
    stats = {"ok": 0, "skipped": 0, "errors": 0}

    # ----- Per-image loop -----
    for idx, image_path in enumerate(image_files, start=1):
        stem      = image_path.stem    # filename without extension
        ext       = image_path.suffix
        img_info  = get_image_info(image_path)
        t_start   = time.time()

        log.info(f"[{idx}/{len(image_files)}] {image_path.name} | "
                 f"{img_info['file_size_kb']} KB | "
                 f"{img_info['pil_format']} | "
                 f"{img_info['resolution']}")

        # --- Verify the image can actually be opened by Pillow ---
        try:
            with Image.open(image_path) as _probe:
                _probe.verify()   # checks file integrity without full decode
        except UnidentifiedImageError:
            log.warning(f"  SKIP – unrecognised image format: {image_path.name} "
                        "(Pillow cannot open it; your build may lack WEBP/TIFF support)")
            stats["skipped"] += 1
            continue
        except Exception as exc:
            log.warning(f"  SKIP – could not open image: {image_path.name} | {exc}")
            stats["skipped"] += 1
            continue

        # --- Query the Vision LLM ---
        try:
            raw_response = query_llm(image_path, system_prompt)
        except requests.RequestException as exc:
            elapsed = time.time() - t_start
            log.error(f"  LLM request failed | {image_path.name} | {elapsed:.1f}s | {exc}")
            stats["errors"] += 1
            continue

        # --- Parse JSON from response ---
        try:
            parsed_json = parse_llm_response(raw_response)
        except (ValueError, json.JSONDecodeError) as exc:
            elapsed = time.time() - t_start
            log.error(f"  JSON parse failed | {image_path.name} | {elapsed:.1f}s | {exc}")
            log.debug(f"  Raw response snippet: {raw_response[:300]}")
            stats["errors"] += 1
            continue

        detections = parsed_json.get("detections") or []
        log.info(f"  LLM returned {len(detections)} detection(s)")

        # --- Save raw JSON to debug folder (overwritten on re-run) ---
        json_path = DEBUG_DIR / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(parsed_json, fh, indent=2, ensure_ascii=False)
        log.info(f"  Debug JSON saved → {json_path}")

        # --- Draw bounding boxes and save preview to debug folder ---
        annotated_path = DEBUG_DIR / f"{stem}_lmm_vision{ext}"
        try:
            draw_bounding_boxes(image_path, detections, annotated_path)
            log.info(f"  Annotated preview → {annotated_path}")
        except Exception as exc:
            log.warning(f"  Could not draw bounding boxes: {exc}")

        # --- Convert to YOLO format and save .txt ---
        update_class_map(detections, class_map)
        yolo_lines = detections_to_yolo(detections, class_map)

        yolo_path = LABELS_TRAIN_DIR / f"{stem}.txt"
        with open(yolo_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(yolo_lines))
            if yolo_lines:
                fh.write("\n")

        elapsed = time.time() - t_start
        log.info(f"  YOLO label saved → {yolo_path}")
        log.info(f"  Completed in {elapsed:.1f}s | "
                 f"model: {LM_STUDIO_MODEL} | "
                 f"detections: {len(detections)} | SUCCESS")
        stats["ok"] += 1

    # ----- Save dataset metadata -----
    log.info("Saving dataset metadata")
    save_dataset_metadata(class_map, DATASET_ROOT)

    # ----- Summary -----
    log.info("=" * 60)
    log.info(f"Session finished  |  "
             f"ok: {stats['ok']}  skipped: {stats['skipped']}  errors: {stats['errors']}")
    log.info(f"Classes : {list(class_map.keys())}")
    log.info(f"Dataset : {DATASET_ROOT.resolve()}")
    log.info(f"Log     : {Path(LOG_FILE).resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    process_images()
