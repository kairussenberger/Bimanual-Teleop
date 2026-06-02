#!/usr/bin/env bash
# Fallback: run the public HTTPS tunnel in its OWN terminal (use this if the
# --tunnel banner gets buried under viewer logs). Start the sim first with:
#   uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer --tunnel
# (or without --tunnel but set vr.tunnel: true), then run this and copy the
# https://<...>.trycloudflare.com URL onto the Quest. No account needed.
exec cloudflared tunnel --url http://localhost:8012
