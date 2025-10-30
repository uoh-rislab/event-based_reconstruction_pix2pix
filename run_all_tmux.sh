#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pix2pix_unet_faces"
DEVICE="dgx-1"
GPU="0"
VARIANTS=( "X1_t_t" "X1_t_t0" "X1_t0_t" "X1_t0_t0" )

# Si necesitas activar conda dentro de tmux, descomenta y ajusta estas líneas:
# CONDA_INIT='source ~/.bashrc && conda activate pix2pix'

# --- Checks ---
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux no está instalado."
  exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"

# --- Launch one tmux per run ---
for X1 in "${VARIANTS[@]}"; do
  SESS="pix2pix_${X1}"

  # Si existe, la reemplaza (evita choques al relanzar)
  if tmux has-session -t "${SESS}" 2>/dev/null; then
    echo "[INFO] Sesión ${SESS} ya existe. Matando la sesión anterior..."
    tmux kill-session -t "${SESS}"
  fi

  CMD="
    cd \"${PROJECT_DIR}\" && \
    echo \"[INFO] Iniciando ${X1} en GPU ${GPU}\" && \
    ${CONDA_INIT:-:} && \
    python train_pytorch.py --device ${DEVICE} --x1 ${X1} --gpu ${GPU} \
    2>&1 | tee \"logs/${SESS}.log\"
  "

  tmux new-session -d -s "${SESS}" "bash -lc '${CMD}'"
  echo "[OK] Lanzada sesión tmux: ${SESS}  ->  adjuntar: tmux attach -t ${SESS}"
done

echo "[DONE] Todas las sesiones fueron lanzadas."
