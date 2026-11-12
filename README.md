# Cat Detector - Raspberry Pi 5 + Hailo 10H

Identify your individual cats by name from a live camera feed using a two-stage
detection and classification pipeline on Raspberry Pi 5 with the Hailo 10H AI
accelerator.

- **Stage 1 (Detection):** Stock COCO YOLOv8n runs on the Hailo accelerator (26 TOPS)
  to detect cats in the camera stream. No fine-tuning required.
- **Stage 2 (Classification):** A MobileNetV2 INT8 classifier runs on the CPU to
  identify which cat is in each detection crop.

While built for cats, the same architecture works for any pet or animal. Just point
the camera, collect images, and train the classifier on your own classes.

> **License notice:** source-available for noncommercial use only. No redistribution,
> no commercial use, and no use as AI/ML training data. Models you train with this
> software are for your own noncommercial use only. See [License](#license).

## Overview

The workflow has three phases:

1. **Collect**: Capture and label images of each cat using a web dashboard on the Pi
2. **Train**: Crop detections, train the MobileNetV2 classifier on a GPU machine
3. **Deploy**: Run two-stage inference on the Pi (Hailo detection + CPU classification)

Two web dashboards manage the pipeline:
- `pi_dashboard.py`: Data collection, image review, and live inference control (runs on Pi)
- `training_dashboard.py`: Classifier training pipeline with version management, live
  metrics, and image labeling (runs on GPU machine)

## Hardware Requirements

**Raspberry Pi (collection + deployment):**
- Raspberry Pi 5
- Hailo 10H AI HAT+ (26 TOPS) with hailort and hailo-apps 25.12.0
- USB camera (e.g., Arducam)

**Training machine (WSL2/Linux x86_64):**
- NVIDIA GPU with CUDA support (6+ GB VRAM recommended)

## Quick Start

```bash
# === On the Dev/Training Machine (WSL2/Linux) ===

# 1. Clone and setup
git clone <your-repo-url> ~/cat-detector
cd ~/cat-detector
bash scripts/setup_training.sh

# 2. Create your config
cp configs/project.yaml.example configs/project.yaml
# Edit configs/project.yaml: set your cat names under "cats:"

# 3. Set up passwordless SSH to the Pi
ssh-keygen -t ed25519       # skip if you already have a key
ssh-copy-id <user>@<pi-hostname>

# 4. Create .env with your Pi connection details
cat > .env << 'EOF'
PI_USER=<user>
PI_HOST=<pi-hostname>
PI_REMOTE_DIR=~/Repos/cat_detection
EOF

# 5. Push code + config to the Pi
./scripts/sync_to_pi.sh


# === On the Raspberry Pi (via SSH) ===

# 6. One-time Pi setup
ssh <user>@<pi-hostname>
cd ~/Repos/cat_detection
bash scripts/setup_pi.sh

# 7. Start the dashboard
source hailo-apps/setup_env.sh
python pi_dashboard.py
# Open http://<pi-ip>:5003 in your browser
# Capture tab: select a cat, capture 300+ images per cat
# Review tab: label unlabeled images, delete bad captures


# === Back on the Dev/Training Machine ===

# 8. Train the classifier via the training dashboard
source venv_train/bin/activate
python training_dashboard.py
# Open http://localhost:5004, run steps: pull > label > crop > prepare > train
# The "Pull" step imports captures from the Pi automatically

# 9. Deploy: use the training dashboard's "Deploy" step to push the model to the Pi
# Restart the dashboard on the Pi to load the new model
# Toggle the classifier on/off from the Pi dashboard's detection mode switcher
```

## Phase 1: Collecting Data

### Setup

Push the project to the Pi from your dev machine, then set up on the Pi:

```bash
# On the dev machine
./scripts/sync_to_pi.sh

# On the Pi (via SSH)
cd ~/cat-detector
bash scripts/setup_pi.sh    # one-time
source hailo-apps/setup_env.sh
```

### Running the Dashboard

```bash
python pi_dashboard.py
```

Open `http://<pi-ip>:5003` in your browser. The dashboard has two tabs:

**Capture tab:**
- **Live camera preview** via MJPEG streaming
- **Cat selection buttons** (keyboard shortcuts 1-9) to capture and label images
- **Auto-capture mode** that takes images at a regular interval
- **Detection mode switcher**: toggle between COCO (generic cat detection) and
  classifier mode (individual cat identification) when a trained model exists

**Review tab:**
- **Image viewer** to iterate through captured images
- **Label buttons** to assign/reassign cat labels (or mark as unlabeled)
- **Delete button** to move bad captures to trash
- **Navigation**: next/prev/skip with keyboard shortcuts

Command-line options:

```bash
python pi_dashboard.py --no-camera    # Review-only mode (no camera needed)
python pi_dashboard.py --port 8080    # Custom port
python pi_dashboard.py --config alt.yaml  # Alternate config file
```

### Tips for Good Training Data

- Aim for **300-500 images per cat** minimum
- Vary **lighting**: day, evening, lamp-lit
- The Arducam's IR mode produces grayscale images at night; include these
  if you want night detection
- Try to keep class counts **roughly balanced** across cats

### Reviewing and Labeling Data

The Review tab in the dashboard shows images one at a time and lets you assign labels,
delete bad captures, or skip to the next image. Unlabeled images (from auto-capture or
detection capture) are shown first.

Detection capture saves a bbox sidecar JSON alongside each image, recording where cats
were detected in the frame. These sidecars are used later by `train/crop_images.py` to
extract classifier training crops.

### Transferring Data to the Training Machine

Use the training dashboard's **Pull** step (see Phase 2), which runs
`train/import_captures.py` to import captures from the Pi into a versioned
directory on the training machine.

## Phase 2: Training the Classifier

### Setup

```bash
cd ~/cat-detector
bash scripts/setup_training.sh
source venv_train/bin/activate
```

Verify CUDA is available. The setup script will report your GPU.

### Training Dashboard (recommended)

```bash
python training_dashboard.py
```

Open `http://localhost:5004` in your browser. The dashboard guides you through the
full training pipeline:

1. **Pull**: Import captures from `data/raw/` into a versioned directory
2. **Label**: Review and assign cat labels to unlabeled images
3. **Crop**: Extract 224x224 crops from raw images using bbox sidecars
4. **Prepare**: Build train/val/test splits with session-aware deduplication
5. **Train**: Run 4-phase MobileNetV2 training with live progress
6. **Deploy**: Copy the trained model to the deployment location

The dashboard also provides:
- **Version management**: create, rename, duplicate, or delete training versions
- **Metrics tab**: training curves (loss, accuracy) and per-fold cross-validation results
- **Settings tab**: edit classifier training hyperparameters
- **Dataset stats**: image counts per cat, per split

### Manual CLI Training

If you prefer the command line, run the steps individually:

```bash
# 1. Import captures into version directory
python train/import_captures.py

# 2. Extract 224x224 crops from raw images using bbox sidecars
python train/crop_images.py

# 3. Build train/val/test splits (session-aware deduplication)
python train/prepare_classifier_dataset.py

# 4. (Optional) Run k-fold cross-validation to evaluate generalization
python train/cross_validate.py

# 5. Train the classifier (4-phase MobileNetV2)
python train/train_classifier.py
```

### How Training Works

The classifier is built on MobileNetV2, a lightweight image recognition model
pre-trained on ImageNet (millions of general images). It has two main parts:

- **Backbone**: The bulk of the network. It extracts visual features from the input
  image (edges, textures, shapes, patterns). Since it was pre-trained on ImageNet,
  it already knows how to "see" - we mostly keep these weights and only fine-tune
  the last few layers.
- **Classification head**: A small layer at the end that takes the backbone's features
  and outputs a probability for each cat. This is replaced with a new layer sized to
  your number of cats and trained from scratch.

Training runs in four phases:

1. **Phase 1 - Head warmup (FP32):** Only train the classification head while the
   backbone is frozen. This teaches the new head to map visual features to your
   specific cats without disrupting the backbone's learned representations.
   (default: 20 epochs, lr=0.001). This phase converges quickly since only a small
   layer is training. 10-20 epochs is usually enough; more won't help much.
2. **Phase 2 - Partial fine-tune (FP32):** Unfreeze the last 3 backbone blocks and
   train them alongside the head. This adapts the deeper feature extraction to
   cat-specific details (fur patterns, face shapes) while keeping earlier layers
   (basic edges, textures) intact. Uses a smaller learning rate for the backbone
   than the head to avoid overwriting useful pre-trained features. (default: 40
   epochs). **This is the most impactful phase to increase** - if accuracy is low,
   try 60-100 epochs here first. Early stopping (patience: 15) will cut it short
   if the model stops improving.
3. **Phase 3 - Quantization-aware training (QAT):** The Pi's CPU runs faster with
   INT8 math (8-bit integers) than FP32 (32-bit floats), but converting directly
   loses accuracy. QAT simulates the rounding errors of INT8 during training so
   the model learns to be robust to them. (default: 10 epochs). If you notice a
   significant accuracy drop between your FP32 validation results and the final
   INT8 model, increase this to 15-25 epochs.
4. **Phase 4 - INT8 export:** Export the final model as a TorchScript file with INT8
   weights, ready for fast inference on the Pi's CPU. This is not a training phase,
   just a conversion step.

**Output:** `models/cat_classifier.pt`, a self-contained TorchScript model with the
class name list embedded, ready for deployment.

**Data augmentation:** During training, images are randomly transformed (flipped,
brightened, rotated, blurred, etc.) so the model sees more variation and generalizes
better to real-world conditions.

Training stops early if validation loss plateaus (patience: 15 epochs).

## Phase 3: Deployment

### Deploying the Classifier

The training dashboard's **Deploy** step handles copying the trained model to the Pi.
You can also deploy manually with `./scripts/transfer_model.sh`.

### Running Inference

**Via the dashboard** (recommended): The dashboard runs `detect.py` as a managed
subprocess. Stage 1 (Hailo YOLOv8) always runs for cat detection. When the classifier
is enabled, Stage 2 identifies which cat is in the bowl ROI.

**Standalone** (requires the hailo-apps venv):

```bash
source ~/hailo-apps/venv_hailo_apps/bin/activate
cd ~/cat-detector
python deploy/detect.py
```

All settings are read from `configs/project.yaml` (copy from `project.yaml.example` to
get started). The config controls camera device, classifier settings, detection capture,
and feeding session tracking. You can override Hailo-specific arguments:

```bash
python deploy/detect.py --input /dev/video2
```

### Two-Stage Pipeline

1. **Stage 1 (Hailo):** GStreamer captures frames, YOLOv8n detects objects, filters
   for "cat" class detections
2. **Classifier gate:** Checks if a cat detection overlaps the configured bowl ROI and
   has been stable for `stability_frames` consecutive frames
3. **Stage 2 (CPU):** Crops the detection to 224x224, sends to the MobileNetV2
   classifier running in a separate process (pinned to a single CPU core)
4. **Vote window:** Accumulates predictions over `vote_window_size` frames; confirms
   identity when one class has majority votes AND average confidence exceeds
   `confidence_threshold`
5. **Logging:** Confirmed identifications are logged to SQLite (`classifier_log.db`)
   with timing, confidence, and frame count

### Detection-Triggered Capture

When `detection_capture.enabled` is `true` in `project.yaml`, inference automatically
saves frames when cats are detected. COCO class names listed in `unlabeled_classes`
(e.g., `["cat"]`) are saved to `data/raw/unlabeled/` with bbox sidecar JSONs for later
labeling and cropping. Named cats are saved to their own directories.

### Feeding Session Tracking

When `feeding.enabled` is `true`, the system tracks when cats are at the bowl and logs
sessions to a SQLite database on the Pi. The classifier provides per-cat identification,
so feeding logs include which specific cat was eating.

## Configuration Reference

All settings are in `configs/project.yaml`. This file is gitignored; copy from
`configs/project.yaml.example` to create your local config.

| Key | Default | Description |
|-----|---------|-------------|
| `cats` | (your cat names) | List of cat names (class labels for the classifier) |
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
| **Deployment** | | |
| `deployment.hailo_apps_dir` | ./hailo-apps | Path to hailo-apps installation |
| `deployment.hef_path` | "" | Custom HEF model path (empty = COCO default) |
| `deployment.labels_json` | "" | Custom labels JSON (empty = COCO default) |
| `deployment.detection_threshold` | 0.5 | Minimum confidence score |
| `deployment.hw_arch` | hailo10h | Hailo hardware target |
| `deployment.track_all` | true | Track all detected classes |
| `deployment.filter_classes` | [] | Empty = all; list = filter to specific classes |
| `deployment.preview_path` | /tmp/cat_detection_preview.jpg | Preview frame path |
| `deployment.preview_fps` | 15 | Preview frame rate |
| **Classifier (Stage 2)** | | |
| `classifier.enabled` | false | Enable Stage 2 identity classification |
| `classifier.model_path` | models/cat_classifier.pt | TorchScript classifier model |
| `classifier.bowl_roi` | [760, 390, 1160, 790] | Bowl region of interest [x_min, y_min, x_max, y_max] |
| `classifier.stability_frames` | 5 | Consecutive detection frames before classifying |
| `classifier.classify_interval_seconds` | 1.0 | Throttle classification rate |
| `classifier.confidence_threshold` | 0.85 | Vote confirmation confidence threshold |
| `classifier.vote_window_size` | 10 | Rolling window size for majority voting |
| `classifier.db_path` | data/classifier_log.db | SQLite feeding event log path |
| `classifier.crop_padding` | 0.2 | Padding around bbox crop (fraction of bbox size) |
| `classifier.roi_overlap_threshold` | 0.01 | Min overlap with bowl ROI to trigger classification |
| `classifier.cpu_core` | 3 | Pin classifier process to this CPU core |
| **Detection Capture** | | |
| `detection_capture.enabled` | false | Auto-capture frames on detection |
| `detection_capture.output_dir` | data/raw | Where detected frames are saved |
| `detection_capture.unlabeled_classes` | ["cat"] | COCO classes saved as "unlabeled" |
| `detection_capture.filter_labels` | [] | Filter to specific labels |
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
| `label_review.host` | 0.0.0.0 | Label review bind address |
| `label_review.port` | 5001 | Label review port |
| `label_review.trash_dir` | data/trash | Where deleted images are moved |
| **Classifier Training** | | |
| `classifier_training.phase1_epochs` | 20 | Head warmup epochs |
| `classifier_training.phase2_epochs` | 40 | Fine-tune epochs |
| `classifier_training.phase3_epochs` | 10 | QAT epochs |
| `classifier_training.phase1_lr` | 0.001 | Phase 1 learning rate |
| `classifier_training.phase2_backbone_lr` | 1e-5 | Phase 2 backbone learning rate |
| `classifier_training.phase2_head_lr` | 1e-4 | Phase 2 head learning rate |
| `classifier_training.weight_decay` | 0.005 | Weight decay |
| `classifier_training.batch_size` | 32 | Training batch size |
| `classifier_training.patience` | 15 | Early stopping patience |
| `classifier_training.optimizer` | AdamW | Optimizer |
| `classifier_training.session_gap_minutes` | 5 | Session grouping gap for splits |
| `classifier_training.mixup_alpha` | 0.2 | Mixup augmentation alpha |
| `classifier_training.balance_classes` | true | Oversample minority classes |
| `classifier_training.cv_folds` | 5 | Cross-validation folds |

## Project Structure

```
cat_detection/
├── pi_dashboard.py                     # Web dashboard (capture + review + inference)
├── training_dashboard.py               # Training pipeline UI
├── templates/
│   ├── dashboard.html                  # Dashboard UI template
│   └── training_dashboard.html         # Training dashboard UI template
├── configs/
│   └── project.yaml.example            # Configuration template (copy to project.yaml)
├── collect/                            # Data collection modules
│   ├── capture.py                      # Camera thread, save worker
│   ├── label_review.py                 # Image review and labeling
│   └── export.py                       # Dataset archive export
├── train/                              # Classifier training (GPU machine)
│   ├── import_captures.py              # Import captures into version directory
│   ├── crop_images.py                  # Extract 224x224 crops from bbox sidecars
│   ├── prepare_classifier_dataset.py   # Session-aware train/val/test split
│   ├── train_classifier.py            # 4-phase MobileNetV2 training
│   └── cross_validate.py              # K-fold cross-validation
├── deploy/                             # Inference (Raspberry Pi + Hailo)
│   ├── detect.py                       # Stage 1: GStreamer + Hailo detection pipeline
│   ├── cat_classifier.py              # Stage 2: MobileNetV2 identity classifier
│   ├── coco_labels.json               # COCO 80-class labels for Stage 1
│   ├── detection_capturer.py          # Auto-capture on detection
│   └── feeding_tracker.py             # Feeding session tracking
├── scripts/                            # Setup and utility scripts
│   ├── _common.sh                      # Shared shell helpers (venv activation, .env)
│   ├── setup_pi.sh                     # Pi environment setup
│   ├── setup_training.sh              # Training machine setup
│   ├── sync_to_pi.sh                  # Sync code to Pi via SSH
│   ├── transfer_model.sh             # Transfer model to Pi
│   └── pull_dataset.sh               # Pull training data from Pi
├── utils/                              # Shared utilities
│   ├── __init__.py
│   ├── config.py                       # YAML config loading, path resolution
│   └── display.py                      # Colored console output
├── requirements-pi.txt                 # Pi dependencies (Flask, OpenCV, PyTorch, PyYAML)
├── requirements-training.txt           # Training machine dependencies (PyTorch, CUDA)
└── requirements-scripts.txt            # Script-only dependencies (PyYAML)
```

## Troubleshooting

**Camera not detected:**
- Check available devices: `ls /dev/video*`
- Set the device in `configs/project.yaml` under `camera.device`
- Verify USB connection: `lsusb`

**Low FPS in capture UI:**
- Ensure USB 3.0 connection (not 2.0)
- Reduce resolution in `configs/project.yaml`

**Classifier not running:**
- Ensure `classifier.enabled` is `true` in `configs/project.yaml`
- Check that `models/cat_classifier.pt` exists on the Pi
- Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`

**Wrong cat identification:**
- Increase `classifier.vote_window_size` for more stable predictions
- Check training data balance and collect more images for underrepresented cats
- Review per-cat accuracy in the training dashboard metrics tab
- Re-run cross-validation (`python train/cross_validate.py`) to check generalization

**Classifier responding slowly:**
- Check `classifier.cpu_core` is set to an available core
- Increase `classifier.classify_interval_seconds` to reduce CPU load

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

- **More data**: especially for underrepresented cats; aim for balanced counts
- **Varied conditions**: different lighting, angles, distances, and poses
- **Night images**: include IR/grayscale captures if you need nighttime detection
- **Cross-validation**: run `python train/cross_validate.py` to check per-fold accuracy
  and identify cats that the model struggles with
- **QAT tuning**: if INT8 accuracy drops significantly, increase `phase3_epochs`
- **Training dashboard**: use the metrics tab to compare training runs and identify
  weak classes
- **Iterate**: use detection capture to collect more labeled data automatically,
  review in the dashboard, then retrain with the expanded dataset

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit a pull request and the terms that apply to contributions.

## License

This project is source-available under the [Purrceptron Source-Available License 1.0](LICENSE.md):

- **Allowed:** using the software for noncommercial purposes, and changing it privately for your own setup.
- **Allowed:** forking the repository solely to contribute changes back through pull requests (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- **Not allowed:** redistributing the software, publishing modified versions, or maintaining a fork as a separate project.
- **Not allowed:** using the software, or any part of it, as training, evaluation, or retrieval data for any AI or machine learning model, giving it to an AI service whose provider is allowed to train on it, or text and data mining. All such rights are reserved. AI coding assistants are fine if their terms don't let the provider train on your input.
- **Not allowed:** commercial use without a separate license from the author.
- **Trained models:** any model you train with the software may be used only by you, for noncommercial purposes. You may not share, sell, or commercially use it.

Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
