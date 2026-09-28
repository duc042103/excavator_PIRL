#!/usr/bin/env bash
# Prepare the excavator USD used by every script in this repo.
#
#   scripts/setup_assets.sh                      # auto: assets/ folder, else download
#   scripts/setup_assets.sh /path/to/model.urdf  # a specific URDF
#   scripts/setup_assets.sh /path/to/folder      # a folder that contains the URDF + meshes/
#
# Steps
#   1. find MathScavator9000_flat.SLDASM.urdf (+ its meshes/ folder):
#        argument  ->  assets/MathScavator9000_flat/  ->  git clone of
#        github.com/mathworks-robotics/autonomous-excavator
#   2. repair it and convert it to USD with scripts/convert_urdf.py
#      (runs inside Isaac Sim through isaaclab.sh)
#   3. the result is assets/usd/excavator.usd -- the default path of the task
#
# Environment
#   ISAACLAB_PATH   Isaac Lab checkout (default: ~/IsaacLab)
#   EXCAVATOR_USD   output USD (default: <repo>/assets/usd/excavator.usd)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
OUT_USD="${EXCAVATOR_USD:-$REPO_ROOT/assets/usd/excavator.usd}"
SRC_REPO_URL="https://github.com/mathworks-robotics/autonomous-excavator.git"
URDF_NAME_HINT="MathScavator9000_flat"

die() { echo "[setup_assets] ERROR: $*" >&2; exit 1; }
log() { echo "[setup_assets] $*"; }

# pick the best URDF below a folder: prefer the MathScavator9000_flat one
find_urdf() {
    local dir="$1" hit
    hit="$(find "$dir" -type f -iname "*${URDF_NAME_HINT}*.urdf" 2>/dev/null | sort | head -n 1)"
    [[ -z "$hit" ]] && hit="$(find "$dir" -type f -iname "*.urdf" 2>/dev/null | sort | head -n 1)"
    echo "$hit"
}

# ---------------------------------------------------------------- 1. source --
URDF=""
if [[ $# -ge 1 ]]; then
    if [[ -f "$1" ]]; then
        URDF="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    elif [[ -d "$1" ]]; then
        URDF="$(find_urdf "$(cd "$1" && pwd)")"
        [[ -n "$URDF" ]] || die "no .urdf file found under '$1'"
    else
        die "'$1' does not exist"
    fi
elif [[ -d "$REPO_ROOT/assets/$URDF_NAME_HINT" ]]; then
    URDF="$(find_urdf "$REPO_ROOT/assets/$URDF_NAME_HINT")"
fi

if [[ -z "$URDF" ]]; then
    SRC_DIR="$REPO_ROOT/assets/_src/autonomous-excavator"
    if [[ ! -d "$SRC_DIR/.git" ]]; then
        log "no URDF given and none in assets/ -- cloning $SRC_REPO_URL"
        mkdir -p "$(dirname "$SRC_DIR")"
        git clone --depth 1 "$SRC_REPO_URL" "$SRC_DIR" \
            || die "clone failed. Copy the MathScavator9000_flat folder (urdf/ + meshes/) into $REPO_ROOT/assets/ and re-run."
    fi
    URDF="$(find_urdf "$SRC_DIR")"
    [[ -n "$URDF" ]] || die "no URDF found in $SRC_DIR. Copy the MathScavator9000_flat folder (urdf/ + meshes/) into $REPO_ROOT/assets/ and re-run."
fi
log "URDF: $URDF"

# -------------------------------------------------------------- 2. convert --
[[ -x "$ISAACLAB_PATH/isaaclab.sh" ]] \
    || die "Isaac Lab not found at '$ISAACLAB_PATH' (set ISAACLAB_PATH=/path/to/IsaacLab)"

mkdir -p "$(dirname "$OUT_USD")"
log "converting to $OUT_USD (starts Isaac Sim headless, ~1-2 min)"
"$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/convert_urdf.py" \
    --input "$URDF" --output "$OUT_USD"

[[ -f "$OUT_USD" ]] || die "conversion finished but $OUT_USD was not written -- see the Isaac Sim log above"

# ----------------------------------------------------------------- 3. done --
log "done: $OUT_USD"
if [[ "$OUT_USD" != "$REPO_ROOT/assets/usd/excavator.usd" ]]; then
    log "non-default location: export EXCAVATOR_USD=$OUT_USD before training"
fi
