#!/usr/bin/env bash
# Finds the drive labelled "Glory" (must be on /dev/sda), mounts it if needed,
# builds the trainer, and starts/resumes training. Extra args are passed through:
#   ./run.sh --games 256 --depth 4
set -euo pipefail
cd "$(dirname "$0")"
LABEL="Glory"

DEV=$(lsblk -nrpo NAME,LABEL | awk -v l="$LABEL" '$2==l {print $1; exit}')
if [[ -z "${DEV:-}" ]]; then echo "No partition labelled '$LABEL' found. Is the drive plugged in?"; exit 1; fi
case "$DEV" in /dev/sda*) ;; *) echo "Found '$LABEL' at $DEV, but expected /dev/sda*. Refusing to continue."; exit 1;; esac

MNT=$(lsblk -nro MOUNTPOINT "$DEV" | head -n1)
if [[ -z "$MNT" ]]; then
  if command -v udisksctl >/dev/null; then
    udisksctl mount -b "$DEV" >/dev/null
    MNT=$(lsblk -nro MOUNTPOINT "$DEV" | head -n1)
  else
    MNT="/mnt/$LABEL"; sudo mkdir -p "$MNT"; sudo mount "$DEV" "$MNT"
  fi
fi
if [[ ! -w "$MNT" ]]; then sudo chown "$USER" "$MNT" 2>/dev/null || { echo "$MNT is not writable"; exit 1; }; fi
echo "Using $DEV mounted at $MNT" >&2

if [[ ! -x glory_nnue || glory_nnue.c -nt glory_nnue ]]; then
  gcc -O3 -march=native -ffast-math -pthread -o glory_nnue glory_nnue.c -lm
fi
exec ./glory_nnue --dir "$MNT" "$@"
