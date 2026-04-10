# AI Vision LLM YOLO Dataset

Automatically generate YOLO-format training datasets from images using a **local Vision LLM** (LM Studio). No manual annotation tools like Roboflow or labelImg are required.

---

## How it works

```
images/train/<image>.jpg
        │
        ▼
  Vision LLM (LM Studio)
        │
        ├── dataset/debug/<image>.json            ← raw LLM output (overwritten on re-run)
        ├── dataset/debug/<image>_lmm_vision.jpg  ← annotated preview (overwritten on re-run)
        └── labels/train/<image>.txt              ← YOLO annotation
```

1. Each image in `dataset/images/train/` is base64-encoded and sent to the Vision LLM together with a system prompt loaded from `llm_prompt.txt`.
2. The LLM returns a JSON object with detected objects and normalised bounding boxes (corner format: x1, y1, x2, y2).
3. The raw JSON and annotated preview are saved to `dataset/debug/` for inspection. Re-running the script simply overwrites these files — the training folder stays clean.
4. The corner coordinates are converted to YOLO center format (`cx cy w h`) and saved as `.txt` label files.
5. `dataset/classes.txt` and `dataset/dataset.yaml` are written for YOLO training.

---

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.10 or newer (uses `tuple[...]` type hints) |
| LM Studio | Running locally with a Vision-capable model |
| Python packages | `Pillow`, `requests` |

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## LM Studio setup

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Load a Vision LLM, for example **`zai-org/glm-4.6v-flash`**.
3. Start the **Local Server** (default port `1234`).
4. Make sure the server is reachable from the machine running this script.

---

## Configuration

Open `create_dataset.py` and adjust the variables at the top of the file:

### LM Studio endpoint

```python
LM_STUDIO_HOST        = "192.168.2.111"      # IP address of the LM Studio server
LM_STUDIO_PORT        = 1234                 # Port (default 1234)
LM_STUDIO_MODEL       = "zai-org/glm-4.6v-flash"
LM_STUDIO_TEMPERATURE = 0.1                  # Lower = more deterministic output
LM_STUDIO_TIMEOUT     = 120                  # Seconds to wait per image
```

### Bounding box drawing style

All visual properties of the annotated debug previews are configurable:

```python
BOX_COLOR        = (255, 255, 255)  # RGB: bounding box outline colour
BOX_THICKNESS    = 3                # outline thickness in pixels
FONT_SIZE        = 16               # label text size in points
LABEL_PADDING    = 3                # padding inside the label chip (pixels)
LABEL_BG_COLOR   = (0, 0, 0)       # RGB: label background colour
LABEL_BG_ALPHA   = 200             # 0–255: label background transparency (0=invisible, 255=solid)
LABEL_TEXT_COLOR = (0, 255, 0)     # RGB: label text colour
```

Some example colour schemes:

| Style | BOX_COLOR | LABEL_BG_COLOR | LABEL_TEXT_COLOR |
|---|---|---|---|
| Default (white/green) | `(255, 255, 255)` | `(0, 0, 0)` | `(0, 255, 0)` |
| Red boxes, yellow text | `(255, 50, 50)` | `(80, 0, 0)` | `(255, 220, 0)` |
| Cyan boxes, black text | `(0, 220, 220)` | `(0, 80, 80)` | `(0, 0, 0)` |

### NMS (Non-Maximum Suppression)

```python
IOU_THRESHOLD = 0.80        # boxes with overlap above this are removed
NMS_MODE      = "per_label" # "per_label" | "global" | "off"
```

---

## System prompt

The LLM system prompt is loaded from **`llm_prompt.txt`** — edit this file to change what the model detects, how labels are named, or how many objects are returned.

The prompt instructs the model to return bounding boxes in normalised **corner format** (`x1 y1 x2 y2`). The script automatically converts these to the YOLO **center format** (`cx cy w h`) before writing the `.txt` files — see [Coordinate conversion](#coordinate-conversion-corner--yolo-center) below.

---

## Usage

### 1. Prepare your images

Place your source JPG (or PNG) images in:

```
dataset/images/train/
```

### 2. Run the script

```bash
python create_dataset.py
```

The script will:
- Create the full folder structure if it does not exist yet.
- Process every image in `images/train/`.
- Print progress for each image.
- Overwrite existing debug files on re-run (the training labels are always regenerated too).

### 3. Check the output

```
dataset/
├── classes.txt                          ← all detected class names (one per line)
├── dataset.yaml                         ← YOLO training config
├── debug/
│   ├── photo.json                       ← raw LLM JSON response
│   └── photo_lmm_vision.jpg             ← annotated preview with bounding boxes
├── images/
│   ├── train/
│   │   └── photo.jpg                    ← original image (never modified)
│   └── val/                             ← put validation images here manually
└── labels/
    ├── train/
    │   └── photo.txt                    ← YOLO annotation
    └── val/
```

---

## YOLO label format

Each `.txt` file contains one line per detected object:

```
<class_id> <cx> <cy> <w> <h>
```

All values are normalised floats in `[0.0, 1.0]`.  
`class_id` corresponds to the line index in `classes.txt` (zero-based).

---

## Coordinate conversion — corner → YOLO center

The LLM returns bounding boxes as two corner points (top-left and bottom-right).
YOLO expects a center point plus width and height. The script converts automatically:

```
Corner format (LLM):              YOLO format (calculated):

(x1,y1)──────────┐                ·         ·         ·
  │               │                ·    (cx,cy)  ←  center point
  │               │                ·         │         ·
  └────────(x2,y2)                 ·        w×h        ·
```

```
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w  =  x2 - x1
h  =  y2 - y1
```

All four values stay normalised (0.0 – 1.0) — no pixel coordinates involved.

### Worked example

Given this LLM response for a ruler image:

```json
{
  "detections": [
    {
      "label": "ruler",
      "confidence": 0.99,
      "bbox_norm": { "x1": 0.05, "y1": 0.32, "x2": 0.95, "y2": 0.66 }
    }
  ]
}
```

Conversion step by step:

```
cx = (0.05 + 0.95) / 2 = 0.500   → horizontally centered (ruler spans full width)
cy = (0.32 + 0.66) / 2 = 0.490   → slightly above vertical center
w  =  0.95 − 0.05      = 0.900   → 90 % of image width
h  =  0.66 − 0.32      = 0.340   → 34 % of image height
```

Resulting YOLO label (`labels/train/ruler.txt`):

```
0 0.500000 0.490000 0.900000 0.340000
```

`0` = class_id for "ruler" (first entry in `classes.txt`)

The annotated debug preview confirms the box fits tightly around the object:

![ruler example](dataset/debug/ruler_lmm_vision.jpg)

---

## Adding validation data

After generating train labels, inspect the annotated previews in `dataset/debug/`.
Move correctly-annotated source images and their `.txt` files to the `val/` folders:

```
dataset/images/val/<image>.jpg
dataset/labels/val/<image>.txt
```

A common split is **80 % train / 20 % val**.

---

## Tips

- **Temperature**: keep `LM_STUDIO_TEMPERATURE` low (0.1–0.3) for consistent, structured JSON output.
- **NMS**: if the same object appears multiple times in detections, lower `IOU_THRESHOLD` or switch to `NMS_MODE = "global"`.
- **Prompt tuning**: edit `llm_prompt.txt` to restrict or expand the set of detected classes.
- **Re-running**: debug files are overwritten automatically; the original images in `images/train/` are never touched.
- **Validation split**: a common split is 80 % train / 20 % val.

---

## File overview

| File | Purpose |
|---|---|
| `create_dataset.py` | Main script |
| `llm_prompt.txt` | System prompt sent to the Vision LLM |
| `requirements.txt` | Python dependencies |
| `dataset/` | Generated dataset (created on first run) |
| `dataset/debug/` | LLM JSON responses and annotated preview images |
