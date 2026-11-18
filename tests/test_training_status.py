# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""What the Training tab is told about jobs: progress, new images, duration."""
import json

from remote_training.manager import TrainingJobManager, parse_progress

# Lines as they appear in real job logs.
TRAIN_LOG = [
    "Training job 20261114-211947 on Local PC",
    "  [P1   1/20] loss 0.413/0.077  acc 84.1%/98.6%",
    "  [P1  20/20] loss 0.300/0.050  acc 90.0%/99.0%",
    "  [P2  12/40] loss 0.481/0.411  acc 70.2%/99.1%",
    "Early stop at epoch 12 (patience 15)",
]
CV_LOG = [
    "Fold 1/5",
    "[CV Fold 1/5] Train: 4426 imgs, Val: 955 imgs",
    "  [CV Fold 1/5] [P3  10/10] loss 0.363/0.414  acc 99.4%/99.1%",
    "Fold 2/5",
    "[CV Fold 2/5] Train: 4400 imgs, Val: 981 imgs",
    "  [CV Fold 2/5] [P1   3/20] loss 0.281/0.049  acc 89.3%/99.3%",
]


def test_progress_training():
    assert parse_progress(TRAIN_LOG) == {"phase": 2, "phases": 3, "epoch": 12, "epochs": 40}


def test_progress_crossval():
    assert parse_progress(CV_LOG) == {"phase": 1, "phases": 3, "epoch": 3, "epochs": 20,
                                      "fold": 2, "folds": 5}


def test_progress_before_training():
    assert parse_progress(["Packaging 5381 images", "Uploading 120 MB"]) is None


def make_manager(tmp_path, counts_now):
    m = TrainingJobManager.__new__(TrainingJobManager)
    m.project_root = tmp_path
    m.jobs_dir = tmp_path / "data" / "training_jobs"
    m.candidates_dir = tmp_path / "models" / "candidates"
    m._read_config = lambda: {"classifier": {"model_path": "models/cat_classifier.pte"}}
    m.image_counts = lambda: counts_now
    return m


def add_job(m, job_id, **state):
    d = m.jobs_dir / job_id
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"job_id": job_id, **state}))


def deploy(m, job_id):
    models = m.project_root / "models"
    models.mkdir(exist_ok=True)
    (models / "cat_classifier.pte").write_bytes(b"x")
    (models / "cat_classifier.meta.json").write_text(json.dumps({"job_id": job_id}))


def test_new_images_since_the_model_in_use(tmp_path):
    m = make_manager(tmp_path, {"Cat1": {"with_bbox": 140}, "Cat2": {"with_bbox": 300}})
    add_job(m, "20261114-211947", kind="train", stage="ready",
            counts={"Cat1": {"with_bbox": 100}, "Cat2": {"with_bbox": 300}})
    assert m.new_images() is None                    # no model in use
    deploy(m, "20261114-211947")
    assert m.new_images() == {"Cat1": 40, "Cat2": 0}


def test_new_images_unknown_for_a_model_installed_by_hand(tmp_path):
    m = make_manager(tmp_path, {"Cat1": {"with_bbox": 140}})
    deploy(m, "")
    assert m.new_images() is None


def test_last_duration_of_a_successful_training(tmp_path):
    m = make_manager(tmp_path, {})
    add_job(m, "20261114-211947", kind="train", stage="ready",
            stage_times={"packaging": "2026-11-14T21:19:47", "ready": "2026-11-14T21:28:00"})
    add_job(m, "20261115-004913", kind="crossval", stage="ready",
            stage_times={"packaging": "2026-11-15T00:49:00", "ready": "2026-11-15T01:30:00"})
    add_job(m, "20261116-090000", kind="train", stage="failed", stage_times={})
    assert m.last_duration() == 493
