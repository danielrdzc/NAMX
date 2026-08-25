#!/usr/bin/env bash
# Instala NAMX como servicio para que arranque solo al encender el Pi.
set -euo pipefail

cd "$(dirname "$0")"

echo "Tarjetas de audio disponibles:"
cat /proc/asound/cards
echo
echo "Los servicios usan 'hw:USB'. Si tu Scarlett aparece con otro nombre entre"
echo "corchetes, edita namx-jack.service antes de continuar."
echo
read -rp "Continuar? [s/N] " ok
[[ "$ok" == "s" || "$ok" == "S" ]] || exit 1

sudo cp namx-jack.service namx.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable namx-jack.service namx.service
sudo systemctl restart namx-jack.service namx.service

sleep 3
systemctl --no-pager --lines=8 status namx-jack.service || true
echo
systemctl --no-pager --lines=12 status namx.service || true
