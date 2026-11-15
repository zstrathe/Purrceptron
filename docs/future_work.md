# Future work: telling look-alike cats apart

The deployed classifier has only been tested on cats that are easy to tell apart.
The most recent dataset (three cats, 1,743 training and 533 validation crops)
reaches 98.5% validation accuracy, with the smallest class at 92% on 39 images.
Nothing in those numbers says how the system would cope with two cats of the same
breed and coat, because it has never been asked to. This page records what would
break first, what the data says about it, and the changes most likely to help,
roughly in order of payoff per unit of work. None of it is implemented.

## How the system fails today when two cats look alike

The classifier is closed-set: it must pick one of the trained cats for every crop.
When two cats are hard to distinguish the model splits its belief between them, and
the two safeguards downstream behave like this:

- **Energy check.** Passes. Energy is `-logsumexp(logits)`, so two strong logits
  give an even lower energy than one. The check correctly says "this is one of my
  cats"; it was never meant to say which one. This is the right behaviour, and it
  is worth stating because the energy check is sometimes mistaken for a confidence
  check.
- **Vote window.** Stalls. Confirmation needs a strict majority of the window
  (6 of 10 by default) for one cat and an average confidence at or above
  `confidence_threshold`. Alternating votes never get there, so the visit is never
  confirmed, never logged, and every image is saved to `unlabeled/`.

So the failure mode is safe (nothing wrong is recorded) but useless (nothing is
recorded). Any of the changes below should be judged on whether it turns stalled
visits into confirmed ones without raising the error rate.

## Where fine detail is lost

Identity cues for look-alikes are small: face markings, eye colour, a notch in an
ear, the exact shape of a chest patch, a collar. The current pipeline throws away
more of that detail than it needs to.

Measured on the latest raw image set (2,318 bounding boxes from the bowl camera at
1920x1080):

| Measure | Value |
| --- | --- |
| Bounding box width, median (p10 to p90) | 836 px (665 to 1100) |
| Bounding box height, median | 637 px |
| Aspect ratio (width / height), median | 1.34 |
| Classifier input | 224 x 224 |

The crop is resized straight to 224x224 with `cv2.resize`, so a typical bounding
box is downscaled about 3.7x horizontally and 2.8x vertically and squashed by
1.34x. Training ([train/crop_images.py](../train/crop_images.py)) and inference
([deploy/detect.py](../deploy/detect.py), `ClassifierGate.put_crop`) do the same
thing, so the model is at least consistent, but a 10 px marking in the frame is
about 3 px by the time the model sees it.

224 was chosen because MobileNetV2 is pretrained at that size and the classifier
runs on one Pi CPU core alongside the GStreamer pipeline. For cats with different
coats that is plenty. For look-alikes it is the first thing to revisit.

## Candidate enhancements

### 1. Collect data where the confusion is

Fine-grained identification is limited by data before it is limited by
architecture. The most effective fix for a confused pair is more images of that
pair, from the bowl camera, across lighting conditions and times of day.

The pipeline nearly supports this already. Cross-validation
([train/cross_validate.py](../train/cross_validate.py)) reports per-class
accuracy but not a confusion matrix, so it can say "Cat2 is at 85%" but not "Cat2
is being mistaken for Cat3". Adding a confusion matrix to `cv_summary.json` and the
Metrics tab would point collection effort at the right pair. Small change, no
retraining cost.

### 2. Keep more of the pixels: letterbox and a configurable input size

Two independent changes, both cheap:

- **Pad to square before resizing** instead of stretching. Preserves the shape of
  markings and the proportions of the face. Needs the same change in
  `crop_images.py` and `put_crop` so training and inference match.
- **Make the input size a setting** (`classifier.input_size`, default 224) and try
  288 or 320. MobileNetV2 is fully convolutional with global pooling, so the
  pretrained weights work unchanged at other sizes. Sites to update:
  `RandomResizedCrop` and the export `example_input` in `train_classifier.py`, the
  `--size` argument in `crop_images.py` (passed from `remote_training/bundle.py`
  when the Pi packages a job), the resize in `put_crop`, and the model's embedded constants so the Pi can
  check that the deployed model and the configured crop size agree.

Cost: compute grows with the square of the side, so 320 is about twice the work of
224 per crop. At one classification per second on a Pi 5 core that is affordable,
but the latency should be measured against `classify_interval_seconds` before
settling on a size. Benefit should be measured with cross-validation on a dataset
that actually contains a confusable pair; on distinct cats it will show nothing.

### 3. Remove augmentations that erase identity

The augmentation set was tuned for generalization on broad classes. Two of the
defaults work against individual identification:

- **Horizontal flip** (`horizontal_flip: true`). A patch on the left cheek becomes a
  patch on the right cheek, which teaches the model that left/right asymmetry
  carries no information. For look-alikes it may be the only information.
- **Hue/saturation shift** (`hue_saturation: true` in the shipped template, up to
  20 hue steps). Coat colour is an identity cue; shifting it teaches the model to
  ignore it.

**Mixup** (`mixup_alpha: 0.2`) is a milder case: it blends two training images and
asks the model to predict a 50/50 label, which is good for calibration on broad
classes and arguably the opposite of what fine-grained discrimination wants.

All three are already configurable, so this is an A/B in cross-validation with no
code change. It should be run before anything larger, because if it helps it is
free.

### 4. Metric learning instead of a softmax head

Individual re-identification (faces, pets, livestock) is normally done with an
embedding trained by ArcFace, triplet or supervised-contrastive loss, then
classified by nearest prototype. The loss explicitly pushes similar individuals
apart, which softmax cross-entropy does only as a side effect.

Secondary benefits that fit this project well:

- Unknown-cat rejection becomes a distance threshold to the nearest prototype,
  which is easier to reason about than an energy threshold and can be calibrated
  per cat.
- A new cat can be enrolled by averaging the embeddings of a few dozen crops, with
  no retraining.

Cost: a different training loop (the four-phase schedule, QAT and INT8 export all
carry over, but the head and the loss change), a prototype file alongside the
model, and a small change in the classifier worker to compute cosine distances.
This is the largest change on the list and the one most likely to matter if two
cats are genuinely hard to separate.

### 5. Run Stage 2 on the NPU

The CPU budget is what keeps the classifier small. Compiling it to a HEF with
Hailo's Dataflow Compiler would let Stage 2 share the Hailo-10H with the detector,
at which point a larger backbone (EfficientNet-B0, ConvNeXt-Tiny) at 320 px costs
nothing on the CPU. The pipeline would also lose the queue-and-drop behaviour,
since inference would keep up with the frame rate.

Cost: Hailo's toolchain, a second HEF to deploy, and changes to the GStreamer
pipeline. Worth doing only after items 2 and 4 have shown that the model, not the
data, is the limit.

Measured on the Pi 5 on 2026-11-15 (HailoRT 5.1.1, `hailortcli run2`, stock Model Zoo
HEFs), the NPU has room for it:

| Test | Result |
| --- | --- |
| yolov8m alone, as fast as possible | 78 fps |
| Stock `mobilenet_v2_1.0` HEF, one crop | 2.9 ms (ExecuTorch on one CPU core: about 10 ms) |
| yolov8m at 30 fps + MobileNetV2 at 5 crops/s | detector stays at 29.8 fps |
| yolov8m at 30 fps + MobileNetV2 as fast as possible | detector 29.8 fps, classifier 1,313 crops/s |
| CPU used by HailoRT during these runs | about 3% of one core |

So the 2026-03 concern that two networks on the Hailo would choke the CPU does not
hold for the inference itself. The toolchain is the real cost:

- The Dataflow Compiler is proprietary, needs a Hailo Developer Zone account, and
  must match the Pi's HailoRT version (5.1.1 from the Raspberry Pi apt repo).
- Hailo does not support its GPU optimization under WSL2, where the training
  worker runs; without a GPU it falls back to its lowest optimization level.
- Its license allows internal use only, so putting it in the Hugging Face job image
  is a grey area.
- Hailo always quantizes from the float model, so the PT2E quantization-aware
  training is not reused.
- The energy check needs 16-bit logit outputs, and a threshold re-fit on HEF
  outputs.

At one crop per second the CPU path costs about 1% of a core, so this only pays off
for a much larger model or for classifying every frame.

### 6. Smaller inference-side changes

- Weight each vote by how far its energy is below the threshold, so confident
  crops count for more.
- Grow the vote window when the top two cats are close, instead of stalling at a
  fixed 10.
- Test-time augmentation: classify the crop and its horizontal mirror (if flip
  augmentation is kept) and average.

Each is a few lines in `deploy/cat_classifier.py`. None addresses the root cause,
but they would make the vote window degrade more gracefully.

## Measuring it

Whatever is tried, two measurements are missing today and should come first:

- **A confusion matrix in cross-validation**, so improvements can be checked per
  pair rather than as an average.
- **A held-out-cat test for unknown rejection.** The energy threshold is set so that
  99% of known-cat validation images pass; nothing measures how many unfamiliar
  cats are rejected. Training with one cat excluded and reporting what fraction of
  its crops are rejected would give that number. It also matters for look-alikes:
  the more cats the model knows, the more of the appearance space is "known", and
  the fewer visitors are caught.

## Suggested order

1. Confusion matrix in cross-validation (measurement, small).
2. Augmentation A/B: flip off, hue/saturation off, mixup off (no code change).
3. Letterbox + `input_size` 288 or 320 (small code change, measure Pi latency).
4. Metric-learning head (larger change, only if 1 to 3 are not enough).
5. NPU offload (largest change, only if the model size is the limit).

# Future work: CLIP on the Hailo

hailo-apps ships a precompiled CLIP ViT-B/32 image encoder for the Hailo-10H
(`/usr/local/hailo/resources/models/hailo10h/clip_vit_b_32_image_encoder.hef`). It
turns a 224x224 crop into a 512-number embedding, and needs no Hailo compiler on our
side. It was considered on 2026-11-15 and left out for now, to keep the processing
load down. None of it is implemented.

Measured on the Pi 5 (HailoRT 5.1.1, `hailortcli run2`):

| Test | Result |
| --- | --- |
| CLIP image encoder, one crop | 14.7 ms |
| yolov8m at 30 fps + CLIP at 5 crops/s | detector stays at 29.8 fps, CLIP 4.9 crops/s |
| Model size on the chip | 15 contexts (swapped in and out per inference) |

A forum user on HailoRT 5.3.0 saw a YOLO detector's latency rise about 1.6x while
sharing the chip with CLIP running continuously, so CLIP on every frame is not
affordable. A few crops per visit is.

## Uses, roughly in order of value

1. **Reject false detections.** After the detector reports a cat, compare the crop's
   embedding with text prompts ("a photo of a cat" against "a dog", "a blanket",
   "an empty bowl") and drop poor matches. This is hailo-apps' `clip` pipeline
   (detection, crop, CLIP). Worth it only if false detections get past the bowl ROI
   and stability checks often enough to matter; count them in the trashed captures
   first.
2. **Labeling help** (no live-pipeline cost). Embed all captures to group
   near-duplicates in the Annotate tab, pre-suggest which cat a crop is, and search
   captures by text ("Cat1 at night").
3. **Bowl state and behaviour** without bounding boxes: classify the bowl ROI as "a cat
   eating from a bowl" against "a cat sitting next to a bowl", or "an empty bowl"
   against "a bowl of food". Untested on this camera.
4. **Animals COCO lacks** (raccoon, opossum, fox), flagged from text prompts alone,
   if the bowl is somewhere they can reach.
5. **Stage 2 replacement.** Keep CLIP frozen and train only a small classifier on the
   embeddings: seconds to train, and a new cat needs photos but no retraining job.
   CLIP was never trained to tell individual cats apart, so it may lose to the
   fine-tuned MobileNetV2 (99.3% in 5-fold cross-validation on 2026-11-15, Cat1
   99.0%). Test it on the PC first: float CLIP ViT-B/32 embeddings plus logistic
   regression on the same folds. Only if it is close, repeat with embeddings from the
   Hailo, since its 8-bit encoder gives slightly different numbers.
