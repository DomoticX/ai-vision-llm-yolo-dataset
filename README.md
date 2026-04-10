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
        ├── labels/train/<image>.json   ← raw LLM output (debug)
        ├── images/train/<image>_lmm_vision.jpg  ← annotated preview
        └── labels/train/<image>.txt   ← YOLO annotation
```

1. Each image in `dataset/images/train/` is base64-encoded and sent to the Vision LLM together with a system prompt loaded from `llm_prompt.txt`.
2. The LLM returns a JSON object with detected objects and normalised bounding boxes (corner format: x1, y1, x2, y2).
3. The raw JSON is saved for debugging purposes.
4. Bounding boxes are drawn on a copy of the image (with semi-transparent label chips) and saved as `<name>_lmm_vision.<ext>`.
5. The corner coordinates are converted to YOLO center format (`cx cy w h`) and saved as `.txt` label files.
6. `dataset/classes.txt` and `dataset/dataset.yaml` are written for YOLO training.

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

```python
LM_STUDIO_HOST        = "192.168.2.111"   # IP address of the LM Studio server
LM_STUDIO_PORT        = 1234              # Port (default 1234)
LM_STUDIO_MODEL       = "zai-org/glm-4.6v-flash"
LM_STUDIO_TEMPERATURE = 0.1              # Lower = more deterministic output
LM_STUDIO_TIMEOUT     = 120             # Seconds to wait per image

IOU_THRESHOLD         = 0.80            # NMS overlap threshold
NMS_MODE              = "per_label"     # "per_label" | "global" | "off"
```

---

## System prompt

The LLM system prompt is loaded from **`llm_prompt.txt`** — edit this file to change what the model detects, how labels are named, or how many objects are returned.

The prompt instructs the model to return bounding boxes in normalised **corner format** (`x1 y1 x2 y2`). The script automatically converts these to the YOLO **center format** (`cx cy w h`) before writing the `.txt` files.

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
- Process every image that does not already have a `_lmm_vision` suffix.
- Print progress for each image.

### 3. Check the output

```
dataset/
├── classes.txt                          ← all detected class names
├── dataset.yaml                         ← YOLO training config
├── images/
│   ├── train/
│   │   ├── photo.jpg                    ← original image (unchanged)
│   │   └── photo_lmm_vision.jpg         ← annotated preview
│   └── val/                             ← put validation images here manually
└── labels/
    ├── train/
    │   ├── photo.json                   ← raw LLM JSON (debug)
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
`class_id` corresponds to the index in `classes.txt`.

---

## Adding validation data

After generating train labels, manually review the annotated `_lmm_vision` images. Move correctly-annotated images and their `.txt` files to the `val/` folders when you are satisfied with the quality:

```
dataset/images/val/<image>.jpg
dataset/labels/val/<image>.txt
```

---

## Tips

- **Temperature**: keep `LM_STUDIO_TEMPERATURE` low (0.1–0.3) for consistent, structured JSON output.
- **NMS**: if the same object is detected multiple times, lower `IOU_THRESHOLD` or switch to `NMS_MODE = "global"`.
- **Prompt tuning**: edit `llm_prompt.txt` to restrict or expand the set of object classes.
- **Re-running**: already-annotated `_lmm_vision` images are automatically skipped.
- **Validation split**: a common split is 80 % train / 20 % val.

---

## File overview

| File | Purpose |
|---|---|
| `create_dataset.py` | Main script |
| `llm_prompt.txt` | System prompt sent to the Vision LLM |
| `requirements.txt` | Python dependencies |
| `dataset/` | Generated dataset (created on first run) |
