#!/usr/bin/env python3
"""
create_dataset.py  –  AI Vision LLM YOLO Dataset Creator
=========================================================
Automatically generates YOLO-format annotations for images in
dataset/images/train/ by querying a local Vision LLM (LM Studio).

Pipeline per image:
  1. Encode image as base64 and send to LM Studio Vision LLM
  2. Parse the JSON response (detections with normalized bbox corners)
  3. Save raw JSON to  labels/train/<name>.json   (debug)
  4. Draw bounding boxes on the image, save as <name>_lmm_vision.<ext>
  5. Convert JSON → YOLO format, save to  labels/train/<name>.txt
  6. Write classes.txt + dataset.yaml to dataset/

Author : AI Vision LLM YOLO Dataset
"""

import os
import re
import sys
import json
import base64
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

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

# --- Image formats to process ---
SUPPORTED_EXTENSIONS  = (".jpg", ".jpeg", ".png")

# --- Bounding box drawing settings ---
BOX_THICKNESS         = 3            # outline thickness in pixels
FONT_SIZE             = 16
LABEL_PADDING         = 3
LABEL_ALPHA           = 200          # 0–255: label background transparency

# --- NMS (Non-Maximum Suppression) settings ---
IOU_THRESHOLD         = 0.80         # overlap threshold; boxes above this are merged
NMS_MODE              = "per_label"  # "per_label" | "global" | "off"


# ============================================================
# FOLDER STRUCTURE
# ============================================================

def create_folder_structure() -> None:
    """Create the standard YOLO dataset folder structure if not yet present."""
    for directory in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR,
                      LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR] {directory}")


# ============================================================
# LLM  –  QUERY & PARSE
# ============================================================

def load_system_prompt(filepath: str) -> str:
    """Load system prompt text from an external file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode image file to base64 string.
    Returns (base64_string, mime_type).
    """
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")

    ext = image_path.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    else:
        mime = "image/jpeg"   # safe fallback

    return b64, mime


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

        # Draw white bounding box outline (multi-pass for thickness)
        for t in range(BOX_THICKNESS):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=(255, 255, 255, 255)
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

        # Semi-transparent black background chip
        draw_ov.rectangle(
            [bx1, by1, bx2, by2],
            fill=(0, 0, 0, LABEL_ALPHA)
        )
        # Green label text
        draw_ov.text(
            (bx1 + pad, by1 + pad),
            txt, font=font, fill=(0, 255, 0, 255)
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

    # ----- Step 1: Folder structure -----
    print("\n=== Creating YOLO folder structure ===")
    create_folder_structure()

    # ----- Step 2: System prompt -----
    print(f"\n=== Loading system prompt from '{SYSTEM_PROMPT_FILE}' ===")
    if not os.path.isfile(SYSTEM_PROMPT_FILE):
        print(f"[ERROR] '{SYSTEM_PROMPT_FILE}' not found. "
              "Create the file and add your LLM prompt.")
        sys.exit(1)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    print(f"  [OK]   {len(system_prompt)} characters loaded.")

    # ----- Step 3: Discover images -----
    # Exclude already-annotated _lmm_vision images
    image_files = sorted([
        f for f in IMAGES_TRAIN_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
        and "_lmm_vision" not in f.stem
    ])

    if not image_files:
        print(f"\n[WARNING] No images found in '{IMAGES_TRAIN_DIR}'.")
        print("Place your source images there and re-run this script.")
        return

    print(f"\n=== Found {len(image_files)} image(s) to process ===")

    class_map: dict[str, int] = {}   # label → class_id (accumulated over run)

    # ----- Per-image loop -----
    for idx, image_path in enumerate(image_files, start=1):
        stem = image_path.stem   # filename without extension
        ext  = image_path.suffix

        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")

        # --- Query the Vision LLM ---
        try:
            raw_response = query_llm(image_path, system_prompt)
        except requests.RequestException as exc:
            print(f"  [ERROR] LLM request failed: {exc}")
            continue

        # --- Parse JSON from response ---
        try:
            parsed_json = parse_llm_response(raw_response)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  [ERROR] JSON parse failed: {exc}")
            print(f"  [RAW]  {raw_response[:300]}")
            continue

        detections = parsed_json.get("detections") or []
        print(f"  [LLM]  Received {len(detections)} detection(s)")

        # --- Save raw JSON (debug) ---
        json_path = LABELS_TRAIN_DIR / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(parsed_json, fh, indent=2, ensure_ascii=False)
        print(f"  [JSON] Debug JSON saved        → {json_path}")

        # --- Draw bounding boxes on image ---
        annotated_path = IMAGES_TRAIN_DIR / f"{stem}_lmm_vision{ext}"
        try:
            draw_bounding_boxes(image_path, detections, annotated_path)
        except Exception as exc:
            print(f"  [ERROR] Could not draw bounding boxes: {exc}")

        # --- Convert to YOLO format and save .txt ---
        update_class_map(detections, class_map)
        yolo_lines = detections_to_yolo(detections, class_map)

        yolo_path = LABELS_TRAIN_DIR / f"{stem}.txt"
        with open(yolo_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(yolo_lines))
            if yolo_lines:
                fh.write("\n")
        print(f"  [YOLO] Label file saved        → {yolo_path}")

    # ----- Save dataset metadata -----
    print("\n=== Saving dataset metadata ===")
    save_dataset_metadata(class_map, DATASET_ROOT)

    # ----- Summary -----
    print(f"\n=== Done! ===")
    print(f"  Processed : {len(image_files)} image(s)")
    print(f"  Classes   : {list(class_map.keys())}")
    print(f"  Dataset   : {DATASET_ROOT.resolve()}")


if __name__ == "__main__":
    process_images()
