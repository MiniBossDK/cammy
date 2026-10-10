#!/usr/bin/env bash
# Load the v4l2loopback module so Cammy can publish a virtual camera (Linux).
# Needs root for modprobe; run it yourself so sudo can ask for your password.
#
#   scripts/setup_virtual_camera.sh [video_nr]     (default /dev/video10)
#
# exclusive_caps=1 is required: Chrome, Zoom and Meet only list a loopback
# device as a camera when it advertises capture capability alone.
set -euo pipefail

NR="${1:-10}"
LABEL="Cammy"
DEV="/dev/video${NR}"

if [[ -e "$DEV" ]] && grep -qx "$LABEL" "/sys/class/video4linux/video${NR}/name" 2>/dev/null; then
    echo "$LABEL virtual camera already exists at $DEV"
    exit 0
fi

if lsmod | grep -q '^v4l2loopback'; then
    echo "v4l2loopback is already loaded, but not as $DEV ($LABEL)." >&2
    echo "Unload it first (stop apps using it), then rerun:  sudo modprobe -r v4l2loopback" >&2
    exit 1
fi

sudo modprobe v4l2loopback devices=1 video_nr="$NR" card_label="$LABEL" exclusive_caps=1
echo "Created $DEV ($LABEL). Run:  python -m cammy --virtual-camera --virtual-device $DEV"
echo
echo "To load it at every boot:"
echo "  echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf"
echo "  echo 'options v4l2loopback devices=1 video_nr=$NR card_label=$LABEL exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf"
