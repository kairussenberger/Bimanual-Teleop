#!/usr/bin/env bash
# Detect this Mac's LAN IP, (re)generate a self-signed cert for it, and print the
# exact Quest URL + launch command. Re-run this whenever your Wi-Fi IP changes.
set -euo pipefail
cd "$(dirname "$0")/.."

IFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
IP="$(ipconfig getifaddr "${IFACE:-en0}" 2>/dev/null || true)"
[ -z "${IP:-}" ] && IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
[ -z "${IP:-}" ] && IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
if [ -z "${IP:-}" ]; then echo "Could not find a LAN IP — are you on Wi-Fi?"; exit 1; fi

openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 825 \
  -subj "/CN=$IP" -addext "subjectAltName=IP:$IP,DNS:localhost" >/dev/null 2>&1

cat <<EOF

  Cert ready for $IP (cert.pem / key.pem).

  1) On the Mac, run:
       uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer

  2) On the Quest 3 browser, open:
       https://$IP:8012
     Tap "Advanced -> proceed" on the cert warning, then "Enter VR" and raise
     your hands. The robot follows on the Mac screen.

  Both devices must be on the SAME Wi-Fi. If the page won't load, the network is
  isolating clients (common on eduroam/campus) — use a phone hotspot, or ngrok.
EOF
