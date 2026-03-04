# Cat Detector — Custom YOLOv8 for Raspberry Pi 5 + Hailo 10H

Identify your individual cats by name using a custom-trained object detection model
running on Raspberry Pi 5 with the Hailo 10H AI accelerator (26 TOPS).

## Overview

This project provides a complete pipeline for training and deploying a custom cat
detector that recognizes your specific cats by name. The workflow has three phases:

1. **Collect** — Capture and label images of each cat using a web dashboard on the Pi
2. **Train** — Auto-annotate, train YOLOv8, and convert to Hailo HEF on a GPU machine
3. **Deploy** — Run real-time inference on the Pi with the Hailo 10H accelerator

A single unified dashboard (`dashboard.py`) handles both data collection and inference
control. Once a trained model exists, the dashboard can start/stop inference directly,
and auto-captured detection frames flow into the same review pipeline.

## Hardware Requirements

**Raspberry Pi (collection + deployment):**
- Raspberry Pi 5
- Hailo 10H AI HAT+ (26 TOPS) with hailort and hailo-apps 25.12.0
- USB camera (e.g., Arducam)

**Training machine (WSL2/Linux x86_64):**
- NVIDIA GPU with CUDA support (6+ GB VRAM recommended)
- Hailo Dataflow Compiler (DFC) for HEF conversion

## Quick Start

```bash
# === On the Raspberry Pi ===

# 1. Clone and setup
git clone <your-repo-url> ~/cat-detector
cd ~/cat-detector
bash scripts/setup_pi.sh

# 2. Start the dashboard
source venv_collect/bin/activate
python dashboard.py
# Open http://<pi-ip>:5003 in your browser
# Capture tab: select a cat, capture 300+ images per cat
# Review tab: label unlabeled images, delete bad captures

# 3. Transfer images to your training machine
tar -czf dataset_$(date +%Y%m%d).tar.gz -C data/raw .
scp dataset_*.tar.gz user@training-machine:~/cat-detector/


# === On the Training Machine (WSL2/Linux) ===

# 4. Clone and setup
git clone <your-repo-url> ~/cat-detector
cd ~/cat-detector
bash scripts/setup_training.sh

# 5. Extract the dataset
tar -xzf dataset_*.tar.gz -C data/raw/

# 6. Prepare YOLO dataset (auto-annotates images)
source venv_train/bin/activate
python train/prepare_dataset.py

# 7. Train the model
python train/train.py

# 8. Export to ONNX
python train/export_onnx.py

# 9. Convert to HEF (requires Hailo DFC)
./train/convert_hef.sh runs/cat_detect/train/weights/best.onnx 3 data/dataset/calibration

# 10. Transfer model to Pi
scp models/cat_detector.hef <user>@<pi-hostname>:~/cat-detector/models/
scp deploy/labels.json <user>@<pi-hostname>:~/cat-detector/deploy/


# === Back on the Raspberry Pi ===

# 11. Switch to fine-tuned model via the dashboard
#     The dashboard auto-detects models in models/*.hef
#     Click "Fine-tuned (individual cats)" in the Capture tab to use your trained model

# Or run inference standalone:
source ~/hailo-apps/venv_hailo_apps/bin/activate
cd ~/cat-detector
python deploy/detect.py
```

## Phase 1: Collecting Data

### Setup

```bash
cd ~/cat-detector
bash scripts/setup_pi.sh
source venv_collect/bin/activate
```

### Running the Dashboard

```bash
python dashboard.py
```

Open `http://<pi-ip>:5003` in your browser. The dashboard has two tabs:

**Capture tab:**
- **Live camera preview** via MJPEG streaming
- **Cat selection buttons** (keyboard shortcuts 1-9) to capture and label images
- **Auto-capture mode** that takes images at a regular interval
- **Detection mode switcher** — toggle between "COCO (cat detection)" for general cat
  detection and "Fine-tuned (individual cats)" when a trained `.hef` model is in `models/`

**Review tab:**
- **Image viewer** to iterate through captured images
- **Label buttons** to assign/reassign cat labels (or mark as unlabeled)
- **Delete button** to move bad captures to trash
- **Navigation** — next/prev/skip with keyboard shortcuts

Command-line options:

```bash
python dashboard.py --no-camera    # Review-only mode (no camera needed)
python dashboard.py --port 8080    # Custom port
python dashboard.py --config alt.yaml  # Alternate config file
```

### Tips for Good Training Data

- Aim for **300-500 images per cat** minimum
- Vary **lighting**: day, evening, lamp-lit
- Vary **distance and angle**: close-up, mid-range, across the room
- Include different **poses**: sitting, walking, lying down, grooming
- Include images with **multiple cats** in frame
- Include **partially occluded** cats (behind furniture, in a box)
- The Arducam's IR mode produces grayscale images at night — include these
  if you want night detection
- Try to keep class counts **roughly balanced** across cats

### Reviewing and Labeling Data

The Review tab in the dashboard replaces the old `collect/review.py` tool. It shows
images one at a time and lets you assign labels, delete bad captures, or skip to the
next image. Unlabeled images (from auto-capture or inference) are shown first.

### Transferring Data to the Training Machine

```bash
tar -czf dataset_$(date +%Y%m%d).tar.gz -C data/raw .
scp dataset_*.tar.gz user@training-machine:~/cat-detector/
```

Alternatively, `python collect/export.py` creates an archive with an additional
manifest (per-cat image counts, session metadata) if you want to inspect the
dataset summary before transferring.

## Phase 2: Training

### Setup

```bash
cd ~/cat-detector
bash scripts/setup_training.sh
source venv_train/bin/activate
```

Verify CUDA is available — the setup script will report your GPU.

### Preparing the Dataset

```bash
# Extract the archive from the Pi
tar -xzf dataset_*.tar.gz -C data/raw/

# Auto-annotate and split into train/val/test
python train/prepare_dataset.py
```

This uses a pretrained COCO YOLOv8 model to detect cats in your images, then assigns
your custom class labels based on the folder structure. Output goes to `data/dataset/`
in YOLO format.

If auto-annotation doesn't work well (e.g., unusual angles or backgrounds), you can
manually annotate using [Roboflow](https://roboflow.com/) or
[CVAT](https://github.com/cvat-ai/cvat), then place the YOLO-format labels in
`data/dataset/labels/`.

### Training the Model

```bash
python train/train.py
```

Override defaults from the command line:

```bash
python train/train.py --epochs 200 --batch-size 8 --model-size yolov8m
```

Watch for:
- **mAP50** > 0.8 indicates good detection accuracy
- **Confusion matrix** in `runs/cat_detect/train/` shows per-cat accuracy
- Training stops early if validation loss plateaus (patience: 30 epochs)

### Exporting to ONNX

```bash
python train/export_onnx.py
```

Exports with `opset=11`, `nms=False`, and `simplify=False` as required by the Hailo
pipeline. The ONNX file is saved alongside the trained weights.

### Converting to HEF

Requires the [Hailo Dataflow Compiler](https://github.com/hailo-ai/hailo_model_zoo):

```bash
./train/convert_hef.sh runs/cat_detect/train/weights/best.onnx 3 data/dataset/calibration
```

Arguments: `<onnx_path> <num_classes> <calibration_dir> [hw_arch]`

This runs INT8 quantization using calibration images and compiles to a HEF binary.
The process can take 30+ minutes depending on model size.

## Phase 3: Deployment

### Transferring the Model

```bash
scp models/cat_detector.hef <user>@<pi-hostname>:~/cat-detector/models/
scp deploy/labels.json <user>@<pi-hostname>:~/cat-detector/deploy/
```

Replace `<user>@<pi-hostname>` with your Pi's username and hostname (e.g., `zack@catpi`).

### Running Inference

**Via the dashboard** (recommended): The dashboard always runs `detect.py` as a managed
subprocess in the background. By default it uses a COCO YOLOv8 model for general cat
detection. Once you have a trained `.hef` model in `models/`, click
"Fine-tuned (individual cats)" in the Capture tab to switch to your custom model, which
identifies cats by name.

**Standalone** (requires the hailo-apps venv):

```bash
source ~/hailo-apps/venv_hailo_apps/bin/activate
cd ~/cat-detector
python deploy/detect.py
```

All settings are read from `configs/project.yaml` — no CLI flags needed. The config
controls the HEF model path, camera device, class filtering, tracking, feeding
session logging, and detection-triggered capture. You can override Hailo-specific
arguments by passing them after `--config`:

```bash
python deploy/detect.py --input /dev/video2
```

### Detection-Triggered Capture

When `detection_capture.enabled` is `true` in `project.yaml`, inference automatically
saves frames when cats are detected. COCO class names listed in `unlabeled_classes`
(e.g., `["cat"]`) are saved to `data/raw/unlabeled/` for later labeling in the
Review tab. Named cats are saved to their own directories.

### Feeding Session Tracking

When `feeding.enabled` is `true`, the system tracks when cats are eating and logs
sessions to a SQLite database. Generate reports with:

```bash
python scripts/feeding_report.py
```

### Verifying the Model

- Check that labels match your cat names (not generic "cat")
- Confidence scores should be > 0.5 for clear detections
- Tracking IDs should stay consistent as a cat moves across frames
- Test with each cat individually to verify per-class accuracy

## Configuration Reference

All settings are in `configs/project.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `cats` | Luna, Mochi, Bean | List of cat names (class labels) |
| **Camera** | | |
| `camera.device` | /dev/video0 | Camera device path or index |
| `camera.width` / `height` | 1920 x 1080 | Camera capture resolution |
| `camera.fps` | 30 | Camera frame rate |
| `camera.format` | MJPG | V4L2 pixel format |
| **Collection** | | |
| `collection.output_dir` | data/raw | Where captured images are stored |
| `collection.min_images_per_cat` | 300 | Minimum images per cat for training |
| `collection.capture_interval_ms` | 500 | Min ms between burst captures |
| `collection.auto_capture_interval_sec` | 3 | Seconds between auto-captures |
| `collection.jpeg_quality` | 95 | JPEG save quality |
| **Web UI** | | |
| `web_ui.host` | 0.0.0.0 | Dashboard bind address |
| `web_ui.port` | 5003 | Dashboard port |
| `web_ui.preview_width` / `height` | 960 x 540 | MJPEG preview resolution |
| **Training** | | |
| `training.model_size` | yolov8s | YOLOv8 variant (n/s/m) |
| `training.image_size` | 640 | Training input resolution |
| `training.epochs` | 150 | Maximum training epochs |
| `training.batch_size` | 16 | Training batch size |
| `training.patience` | 30 | Early stopping patience |
| `training.val_split` | 0.15 | Validation set fraction |
| `training.test_split` | 0.05 | Test set fraction |
| `training.seed` | 42 | Random seed for splits |
| **Deployment** | | |
| `deployment.hef_path` | models/cat_detector.hef | Deployed HEF model path |
| `deployment.labels_json` | deploy/labels.json | Labels file path |
| `deployment.detection_threshold` | 0.5 | Minimum confidence score |
| `deployment.hw_arch` | hailo10h | Hailo hardware target |
| `deployment.track_all` | true | Track all detected classes |
| `deployment.filter_classes` | [] | Empty = all; list = filter to specific cats |
| **Detection Capture** | | |
| `detection_capture.enabled` | true | Auto-capture frames on detection |
| `detection_capture.output_dir` | data/raw | Where detected frames are saved |
| `detection_capture.unlabeled_classes` | ["cat"] | COCO classes saved as "unlabeled" |
| `detection_capture.min_interval_seconds` | 15 | Min seconds between captures per cat |
| `detection_capture.max_captures_per_session` | 0 | 0 = unlimited |
| `detection_capture.jpeg_quality` | 95 | JPEG save quality |
| `detection_capture.csv_logging` | true | Append to capture_log.csv |
| **Feeding Tracker** | | |
| `feeding.enabled` | false | Enable feeding session tracking |
| `feeding.db_path` | data/feeding_log.db | SQLite database path |
| `feeding.cooldown_seconds` | 30 | Seconds of no detection before ending session |
| `feeding.min_detection_frames` | 3 | Consecutive frames to confirm eating |
| `feeding.log_interval_seconds` | 60 | Console status print interval |
| **Label Review** | | |
| `label_review.trash_dir` | data/trash | Where deleted images are moved |

## Project Structure

```
cat_detection/
├── dashboard.py              # Unified web dashboard (capture + review + inference)
├── templates/
│   └── dashboard.html        # Dashboard HTML template
├── configs/
│   └── project.yaml          # Central configuration
├── collect/                   # Data collection modules
│   ├── capture.py            # Camera thread, save worker, capture app
│   ├── label_review.py       # Image review and labeling logic
│   └── export.py             # Dataset archive export
├── train/                     # Model training (GPU machine)
│   ├── prepare_dataset.py    # Auto-annotate and split dataset
│   ├── train.py              # YOLOv8 training
│   ├── export_onnx.py        # Export to ONNX format
│   └── convert_hef.sh        # ONNX → HEF conversion
├── deploy/                    # Inference (Raspberry Pi + Hailo)
│   ├── detect.py             # Main detection pipeline (config-driven)
│   ├── detection_capturer.py # Auto-capture on detection
│   ├── feeding_tracker.py    # Feeding session tracking
│   └── labels.json           # Class ID → cat name mapping
├── scripts/                   # Setup and utility scripts
│   ├── setup_pi.sh           # Pi environment setup
│   ├── setup_training.sh     # Training environment setup
│   ├── sync_to_pi.sh         # Sync code to Pi via SSH
│   ├── transfer_model.sh     # Transfer model to Pi
│   ├── pull_dataset.sh       # Pull dataset from Pi
│   └── feeding_report.py     # Feeding session reports
└── utils/                     # Shared utilities
    ├── __init__.py
    ├── config.py             # YAML config loading, path resolution
    └── display.py            # Colored console output
```

## Troubleshooting

**Camera not detected:**
- Check available devices: `ls /dev/video*`
- Set the device in `configs/project.yaml` under `camera.device`
- Verify USB connection: `lsusb`

**Low FPS in capture UI:**
- Ensure USB 3.0 connection (not 2.0)
- Reduce resolution in `configs/project.yaml`

**No cats detected during auto-annotation:**
- Lower the confidence threshold: `python train/prepare_dataset.py --min-confidence 0.15`
- Check that images clearly show cats (not too dark/blurry)

**Bad training accuracy:**
- Check class balance — collect more images for underrepresented cats
- Try a larger model: `--model-size yolov8m`
- Increase training epochs: `--epochs 300`
- Review and remove bad images in the dashboard's Review tab

**HEF version mismatch on Pi:**
- Ensure the Hailo DFC version matches the hailort version installed on the Pi
- Check with: `hailortcli fw-control identify` on the Pi

**Web UI not accessible from browser:**
- Ensure the Pi's firewall allows port 5003
- The dashboard listens on all interfaces (0.0.0.0) by default
- Try accessing via the Pi's IP address directly

**"Hailo libraries not available" error:**
- Activate the hailo-apps venv: `source ~/hailo-apps/venv_hailo_apps/bin/activate`
- The deploy script requires the Hailo SDK which is only available on the Pi

## Improving Accuracy

- **More data** — especially for underrepresented cats
- **Balance classes** — aim for equal image counts per cat
- **Larger model** — `yolov8m` is slower but more accurate than `yolov8s`
- **More augmentation** — adjust for your specific lighting and camera setup
- **Edge cases** — cats in unusual positions, multiple cats together, partial views
- **Night images** — include IR/grayscale captures if you need nighttime detection
- **Iterate** — use auto-capture mode to collect more labeled data, review it in the
  Review tab, then retrain with the expanded dataset
