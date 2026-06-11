#!/usr/bin/env python
"""Stream the Mac screen (your dashboard) INTO the Quest's ORBIT app.

ORBIT renders a video panel in-headset and SUBs for it on tcp://127.0.0.1:10505
(via adb reverse) — normally fed by a robot camera. Without a stream the panel
is blank and the operator is blind to the dashboard while wearing the headset,
which makes solo operation miserable. This script captures the screen with
ffmpeg (hardware HEVC via VideoToolbox), packetizes it in ORBIT's wire format,
and publishes it — so the dashboard (banner prompts, TRACKED chips, robot view)
is visible inside VR.

Wire format (reverse-engineered from CameraOneStreamer.cs, verified constants):

    ZMQ PUB, one packet per HEVC access unit:
      'O''R''B''T' | version=1 (u8) | flags (u8: 1=keyframe, 2=config)
      | width u16 LE | height u16 LE | ptsUs u64 LE | payloadLen u32 LE
      | payload (Annex-B HEVC access unit)

Every frame is encoded all-intra with VPS/SPS/PPS repeated (dump_extra), so any
frame is a clean decoder entry point (the app's own config uses gop=1).

    uv run python scripts/headset_view.py                 # whole main screen
    uv run python scripts/headset_view.py --fps 30 --width 1280
    uv run python scripts/headset_view.py --list-screens  # pick a display

First run: grant the terminal Screen Recording permission
(System Settings → Privacy & Security → Screen Recording), then rerun.
Requires: ffmpeg (brew install ffmpeg). Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time

MAGIC = b"ORBT"
VERSION = 1
FLAG_KEYFRAME = 0x01
FLAG_CONFIG = 0x02
PORT = 10505

# HEVC NAL unit types (nal_unit_header first byte >> 1, 6 bits)
NAL_AUD = 35
NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34
IDR_TYPES = {19, 20}            # IDR_W_RADL, IDR_N_LP


def _adb_reverse() -> None:
    if not shutil.which("adb"):
        print(f"[headset-view] adb not found — run `adb reverse tcp:{PORT} tcp:{PORT}` yourself")
        return
    r = subprocess.run(["adb", "reverse", f"tcp:{PORT}", f"tcp:{PORT}"],
                       capture_output=True, timeout=5)
    print(f"[headset-view] adb reverse tcp:{PORT}: {'ok' if r.returncode == 0 else 'FAILED'}")


def _split_nals(buf: bytearray):
    """Yield (start, end, nal_type) for each Annex-B NAL in buf (complete ones)."""
    i = 0
    starts = []
    n = len(buf)
    while i < n - 3:
        if buf[i] == 0 and buf[i + 1] == 0 and (buf[i + 2] == 1 or
                                                (buf[i + 2] == 0 and i < n - 4 and buf[i + 3] == 1)):
            sc = 3 if buf[i + 2] == 1 else 4
            starts.append((i, i + sc))
            i += sc
        else:
            i += 1
    for k, (s, p) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else None
        if e is None:
            yield s, None, (buf[p] >> 1) & 0x3F
        else:
            yield s, e, (buf[p] >> 1) & 0x3F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", default="1", help="avfoundation screen index (see --list-screens)")
    ap.add_argument("--list-screens", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1440, help="encoded width")
    ap.add_argument("--height", type=int, default=810,
                    help="encoded height (screen is letterboxed to fit — the header must carry exact dims)")
    ap.add_argument("--bitrate", default="12M")
    ap.add_argument("--lavfi", default=None, help=argparse.SUPPRESS)   # self-test source (e.g. testsrc)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found — brew install ffmpeg")
        return 1
    if args.list_screens:
        subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       stderr=None)
        return 0

    import zmq
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)            # latest-wins: never build latency
    pub.bind(f"tcp://127.0.0.1:{PORT}")
    _adb_reverse()

    vf = (f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
          f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2")
    if args.lavfi:
        src_args = ["-re", "-f", "lavfi", "-i", f"{args.lavfi}=rate={args.fps}"]
    else:
        src_args = ["-f", "avfoundation", "-capture_cursor", "1",
                    "-framerate", str(args.fps), "-i", f"{args.screen}:none"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *src_args,
           "-vf", vf,
           "-c:v", "hevc_videotoolbox", "-realtime", "1",
           "-b:v", args.bitrate, "-g", "1",                       # all-intra: gop 1
           "-bsf:v", "hevc_metadata=aud=insert,dump_extra=freq=keyframe",
           "-f", "hevc", "-"]
    print("[headset-view] " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr)

    buf = bytearray()
    t0 = time.monotonic()
    sent = 0
    w_out, h_out = args.width, args.height
    last_log = 0.0
    last_cfg = -10.0

    def send_config(now: float) -> None:
        # Minimal valid reprojection payload (flat screen: pinhole intrinsics
        # from a chosen FOV; both eyes identical → plain 2D panel).
        hfov, vfov = 70.0, 70.0 * h_out / w_out
        fx = w_out / (2.0 * math.tan(math.radians(hfov) / 2.0))
        fy = h_out / (2.0 * math.tan(math.radians(vfov) / 2.0))
        eye = {"fx": fx, "fy": fy, "cx": w_out / 2.0, "cy": h_out / 2.0,
               "width": w_out, "height": h_out,
               "rectified_hfov_deg": hfov, "rectified_vfov_deg": vfov}
        payload = json.dumps({"type": "reprojection_config", "version": 1,
                              "backend": "headset_view", "profile": "flat",
                              "left": eye, "right": eye,
                              "generated_monotonic_us": int(now * 1e6)}).encode()
        pkt = MAGIC + struct.pack("<BBHHQI", VERSION, FLAG_CONFIG, w_out, h_out,
                                  int((now - t0) * 1e6), len(payload)) + payload
        pub.send(pkt)
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                err = proc.wait()
                print(f"\n[headset-view] ffmpeg exited ({err}). On first run grant the "
                      "terminal Screen Recording permission and rerun.")
                return 1 if err else 0
            buf.extend(chunk)
            # Access units are AUD-delimited (hevc_metadata=aud=insert): emit the
            # span between consecutive AUDs.
            cut = 0
            aud_positions = [s for s, e, ty in _split_nals(buf) if ty == NAL_AUD and e is not None]
            # need at least two AUDs to know one complete AU
            while len(aud_positions) >= 2:
                s0, s1 = aud_positions[0], aud_positions[1]
                au = bytes(buf[s0:s1])
                types = {ty for _, _, ty in _split_nals(bytearray(au))}
                # NOTE: FLAG_CONFIG is reserved for the JSON reprojection-config
                # packet — the app routes config packets AWAY from the decoder.
                flags = FLAG_KEYFRAME if types & IDR_TYPES else 0
                pts = int((time.monotonic() - t0) * 1e6)
                pkt = MAGIC + struct.pack("<BBHHQI", VERSION, flags, w_out, h_out,
                                          pts, len(au)) + au
                pub.send(pkt)
                sent += 1
                cut = s1
                aud_positions = aud_positions[1:]
            if cut:
                del buf[:cut]
            now = time.monotonic()
            if now - last_cfg > 2.0:                 # config keeps late joiners happy
                send_config(now)
                last_cfg = now
            if now - last_log > 5.0:
                last_log = now
                print(f"[headset-view] {sent} frames sent ({sent / max(now - t0, 1e-9):.1f} fps avg)",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        pub.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
