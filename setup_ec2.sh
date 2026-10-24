#!/usr/bin/env bash
# Run once on a fresh g4dn.xlarge (Deep Learning Base GPU AMI, Ubuntu 24.04 or 22.04): ./setup_ec2.sh
# Installs git-lfs, Docker and the NVIDIA container toolkit only if they are missing, downloads the
# adapter weights stored with Git LFS, then checks that a container can see the T4.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 0/4 Git LFS: adapter weights"
if ! git lfs version >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git-lfs
fi
git lfs install
# A clone made before git-lfs was installed only has small pointer files; this replaces them.
git -C "$PROJECT_DIR" lfs pull
ls -lh "$PROJECT_DIR/adapter/adapter_model.safetensors"

echo "== 1/4 NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Launch the instance from a Deep Learning Base GPU AMI (Ubuntu 24.04 or 22.04)." >&2
    exit 1
fi
nvidia-smi

echo "== 2/4 Docker"
if command -v docker >/dev/null 2>&1; then
    echo "Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sudo sh
fi
if ! sudo docker compose version >/dev/null 2>&1; then
    # Docker from Ubuntu's own packages comes without the compose plugin.
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin || {
        sudo mkdir -p /usr/local/lib/docker/cli-plugins
        sudo curl -fsSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
            -o /usr/local/lib/docker/cli-plugins/docker-compose
        sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    }
fi
sudo docker compose version
# Lets this user run docker without sudo after logging out and back in.
sudo usermod -aG docker "$USER"

echo "== 3/4 NVIDIA container toolkit"
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "NVIDIA container toolkit already installed: $(nvidia-ctk --version | head -n 1)"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "== 4/4 GPU inside a container"
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo
echo "Setup complete. Log out and back in (or run 'newgrp docker') so 'docker' works without sudo."
