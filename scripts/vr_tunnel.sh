#!/usr/bin/env bash
# Public HTTPS tunnel to the local Vuer server (works on ANY network, incl.
# isolated campus Wi-Fi). Start this ONCE and leave it running; bookmark the URL
# on the Quest. Then run `run_teleop --vr vuer` (restart it freely; the URL stays
# the same). No account needed.
#
# Shows ONLY the link (cloudflared's log spam is hidden) and saves it to
# .vr_url so you can always retrieve it:   cat ~/Developer/bimanual-teleop/.vr_url
set -euo pipefail
cd "$(dirname "$0")/.."
command -v cloudflared >/dev/null || { echo "cloudflared not found — run: brew install cloudflared"; exit 1; }

echo "Starting tunnel... (the link appears in a few seconds)"
# --protocol http2 + --edge-ip-version 4: eduroam throttles QUIC (UDP 7844), which
# causes endless "context canceled / control stream failure" drops. TCP/HTTP2 is stable.
cloudflared tunnel --url http://localhost:8012 --protocol http2 --edge-ip-version 4 2>&1 | while IFS= read -r line; do
  url=$(printf '%s' "$line" | grep -oE 'https://[-a-z0-9]+\.trycloudflare\.com' || true)
  if [ -n "${url:-}" ]; then
    printf '%s\n' "$url" > .vr_url
    printf '\n=========================================================\n'
    printf '  OPEN THIS ON THE QUEST 3 BROWSER (and bookmark it):\n\n      %s\n\n' "$url"
    printf '  Then run: uv run python -m bimanual_teleop.launch.run_teleop --vr vuer --clutch gesture\n'
    printf '  (URL also saved to .vr_url — leave this terminal running)\n'
    printf '=========================================================\n\n'
  elif printf '%s' "$line" | grep -qE 'ERR |error|failed|fatal'; then
    printf 'tunnel: %s\n' "$line"      # surface problems, hide the rest of the noise
  fi
done
