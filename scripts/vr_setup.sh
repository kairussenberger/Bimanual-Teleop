#!/usr/bin/env bash
# Detect this Mac's LAN IP, (re)generate a self-signed cert for it, and print the
# exact Vuer/WebXR Quest URL + launch command. Re-run this whenever your Wi-Fi IP
# changes. ORBIT/native Quest input does not need this helper.
set -euo pipefail
cd "$(dirname "$0")/.."

# Free a stale server still holding :8012 (the "address already in use" error).
STALE="$(lsof -ti tcp:8012 2>/dev/null || true)"
if [ -n "$STALE" ]; then echo "freeing port 8012 (killing $STALE)"; kill -9 $STALE 2>/dev/null || true; sleep 1; fi

IFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
IP="$(ipconfig getifaddr "${IFACE:-en0}" 2>/dev/null || true)"
[ -z "${IP:-}" ] && IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
[ -z "${IP:-}" ] && IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
if [ -z "${IP:-}" ]; then echo "Could not find a LAN IP — are you on Wi-Fi?"; exit 1; fi

openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 825 \
  -subj "/CN=$IP" -addext "subjectAltName=IP:$IP,DNS:localhost" >/dev/null 2>&1

cat <<EOF

  Port 8012 free, cert ready for $IP (cert.pem / key.pem).

  1) On the Mac, run:
       uv run python -m bimanual_teleop.launch.run_teleop --vr vuer --clutch gesture

  2) On the Quest 3 browser, open EXACTLY this (NOT the "vuer.ai?ws=..." line the
     launcher prints — that one is mixed-content and won't connect):
       https://$IP:8012
     Tap "Advanced -> proceed" on the cert warning, then "Enter VR" and raise
     your hands. Python publishes robot state to the Unity render channel.

  Both devices must be on the SAME network. A phone hotspot (this looks like one)
  works; isolated eduroam/campus Wi-Fi does NOT.
EOF
