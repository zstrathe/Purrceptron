# Training worker for the Pi dashboard's "Local PC" backend, with the GPU.
#
# The image holds only the Python environment. The project is mounted at /app
# when the container runs, so code updates need no rebuild; a change to
# requirements-training.txt does (docker compose ... up -d --build).
#
# The base is the same pytorch/pytorch image Hugging Face jobs use, so both
# backends train with the same torch build.
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

# The image's Python is the system one (PEP 668), and pip runs as root.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/cache/torch

COPY requirements-training.txt /tmp/requirements-training.txt
RUN python -m pip install -r /tmp/requirements-training.txt \
 && python -c "import torch, executorch, torchao, cv2, albumentations; from executorch.runtime import Runtime; print('torch', torch.__version__, '| executorch ok')"

WORKDIR /app
EXPOSE 5004
CMD ["python", "train_worker.py", "--host", "0.0.0.0", "--port", "5004", "--jobs-dir", "/jobs"]
