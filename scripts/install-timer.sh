#!/usr/bin/env bash
# Installs the daily ingestion as a systemd user timer.
#
# This replaces the Cloud Scheduler job that used to trigger the ingestion
# Cloud Function. A user timer needs lingering enabled, otherwise it only runs
# while you are logged in.
set -euo pipefail

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/systemd"

mkdir -p "$UNIT_DIR"
install -m 644 "$SRC_DIR/quantdev-ingestion.service" "$UNIT_DIR/"
install -m 644 "$SRC_DIR/quantdev-ingestion.timer" "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable --now quantdev-ingestion.timer

if ! loginctl show-user "$USER" --property=Linger | grep -q 'Linger=yes'; then
  echo
  echo "Lingering is off, so the timer will not fire while you are logged out."
  echo "Enable it with:  sudo loginctl enable-linger $USER"
fi

echo
systemctl --user list-timers quantdev-ingestion.timer --no-pager
