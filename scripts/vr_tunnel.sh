#!/usr/bin/env bash
# Public HTTPS tunnel to the local Vuer server (works on ANY network, incl.
# isolated campus Wi-Fi). Start this ONCE and leave it running; bookmark the URL
# on the Quest. Then run the sim with `--vr vuer --http` (restart it freely; the
# URL stays the same). No account needed.
set -euo pipefail
command -v cloudflared >/dev/null || { echo "cloudflared not found — run: brew install cloudflared"; exit 1; }

# Run cloudflared, watch its output for the URL, and print it in a big banner.
cloudflared tunnel --url http://localhost:8012 2>&1 | while IFS= read -r line; do
  echo "$line"
  url=$(printf '%s' "$line" | grep -oE 'https://[-a-z0-9]+\.trycloudflare\.com' || true)
  if [ -n "${url:-}" ]; then
    printf '\n========================================================\n'
    printf '  OPEN THIS ON THE QUEST 3 (and bookmark it):\n    %s\n' "$url"
    printf '========================================================\n\n'
  fi
done
