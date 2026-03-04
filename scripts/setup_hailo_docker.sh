#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# One-time setup: Load Hailo AI Software Suite Docker image
# ═══════════════════════════════════════════════════════════════
#
# Download the Docker tar.gz from https://hailo.ai/developer-zone/
# then run this script to load and tag it.
#
# Usage:
#   ./scripts/setup_hailo_docker.sh <hailo-ai-sw-suite-docker.tar.gz>
#
# After setup, the Convert HEF step in the training dashboard
# will automatically use Docker for model compilation.
# ═══════════════════════════════════════════════════════════════

TARBALL="${1:?Usage: ./scripts/setup_hailo_docker.sh <hailo-ai-sw-suite-docker.tar.gz>}"

if [ ! -f "$TARBALL" ]; then
    echo "[ERROR] File not found: $TARBALL"
    exit 1
fi

echo "[INFO] Loading Hailo Docker image from $TARBALL ..."
echo "       (this may take a few minutes for large images)"
echo ""

LOAD_OUTPUT=$(docker load < "$TARBALL")
echo "$LOAD_OUTPUT"

# Extract the loaded image name
LOADED_IMAGE=$(echo "$LOAD_OUTPUT" | grep -oP 'Loaded image: \K.*' | tail -1)

if [ -z "$LOADED_IMAGE" ]; then
    echo "[ERROR] Could not determine loaded image name."
    echo "        Try manually: docker load < $TARBALL"
    echo "        Then: docker tag <image_name> hailo-dfc:latest"
    exit 1
fi

echo ""
echo "[INFO] Loaded: $LOADED_IMAGE"
echo "[INFO] Tagging as hailo-dfc:latest ..."
docker tag "$LOADED_IMAGE" hailo-dfc:latest

echo "[INFO] Verifying hailomz is accessible ..."
if docker run --rm hailo-dfc:latest hailomz --version 2>/dev/null; then
    echo ""
    echo "[OK] Hailo DFC Docker setup complete!"
    echo "     The Convert HEF step in the training dashboard will now use Docker."
else
    echo ""
    echo "[WARN] Could not verify hailomz inside container."
    echo "       The image is tagged as hailo-dfc:latest but hailomz may need"
    echo "       a different entrypoint. Check with: docker run --rm -it hailo-dfc:latest bash"
fi
