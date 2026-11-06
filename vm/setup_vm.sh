#!/usr/bin/env bash
# One-time (idempotent) VM setup. Run with: sudo bash setup_vm.sh
set -euo pipefail
umask 000
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating /workspace (shared, setgid, world-writable)"
mkdir -p /workspace/hf_cache/hub /workspace/hf_cache/datasets /workspace/results /workspace/repo
chmod -R a+rwX /workspace
find /workspace -maxdepth 2 -type d -exec chmod g+s {} +

echo "==> Writing /etc/profile.d/team20.sh"
cat > /etc/profile.d/team20.sh <<'PROFILE'
# shared model/dataset cache; each user's HF login token stays in their own home dir
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
umask 000  # teammates are not in a shared unix group, so new files must be world-writable
case $- in *i*) echo "Team 20: code in /workspace/repo, results in /workspace/results. Run: source /workspace/venv/bin/activate && cd /workspace/repo";; esac
PROFILE
chmod 644 /etc/profile.d/team20.sh

echo "==> Creating venv at /workspace/venv (reuses preinstalled CUDA torch)"
if ! python3 -c "import ensurepip" 2>/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-venv
fi
if [ ! -x /workspace/venv/bin/python ]; then
    python3 -m venv --system-site-packages /workspace/venv
else
    echo "    venv already exists, reusing"
fi

echo "==> Installing requirements"
/workspace/venv/bin/pip install --upgrade pip
/workspace/venv/bin/pip install -r "$HERE/requirements.txt"
# the image ships torchaudio 2.11 against torch 2.9.1, which breaks `import transformers`
/workspace/venv/bin/pip install --no-deps torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
chmod -R a+rwX /workspace/venv

echo "==> GPU status"
nvidia-smi
/workspace/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "==> Done. Everyone: source /workspace/venv/bin/activate"
