# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Pi-driven classifier training on a pluggable backend (local PC, cloud).

- bundle.py:          package labeled images into a job bundle (runs on the Pi)
- backends/:          one module per place a job can run
- manager.py:         job lifecycle on the Pi (package, submit, poll, fetch,
                      validate, deploy, rollback)
- validate_model.py:  load-test a returned model with the Pi's own torch
- worker.py:          HTTP job API the training PC exposes for the local_pc backend
"""
