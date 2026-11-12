"""ffmpeg-based probing and GPU-accelerated (D3D11VA) frame extraction."""
import re
import shutil
import subprocess
import sys
import threading

import numpy as np

from .log import log

_FFMPEG = None
_HWACCEL_OK = {}
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    try:
        import imageio_ffmpeg  # bundled static ffmpeg - keeps the tool self-contained
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _FFMPEG = shutil.which("ffmpeg")
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found (install imageio-ffmpeg or put ffmpeg on PATH)")
    return _FFMPEG


def probe(path):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
    txt = r.stderr.decode("utf-8", errors="replace")
    info = {"duration": None, "width": None, "height": None, "dar": None, "title": None, "subtitles": []}
    # "Stream #0:3(cze): Subtitle: subrip (srt) (default) (forced)" / "Stream #0:3[0x4](ces): Subtitle: mov_text ..."
    for m in re.finditer(r"Stream #0:(\d+)(?:\[\w+\])?(?:\((\w+)\))?: Subtitle: (\w+)([^\n]*)", txt):
        info["subtitles"].append({"stream": int(m.group(1)), "lang": m.group(2) or "", "codec": m.group(3),
                                  "forced": "(forced)" in m.group(4)})
    if m := re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt):
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    head = txt.split("Stream #", 1)[0]
    if m := re.search(r"^\s{4}title\s*:\s*(.+?)\s*$", head, re.M):
        info["title"] = m.group(1)
    for line in txt.splitlines():
        if "Video:" in line and "attached pic" not in line:
            if m := re.search(r"\b(\d{2,5})x(\d{2,5})\b", line):
                info["width"], info["height"] = int(m.group(1)), int(m.group(2))
            if m := re.search(r"DAR (\d+):(\d+)", line):
                a, b = int(m.group(1)), int(m.group(2))
                if a and b:
                    info["dar"] = a / b
            break
    return info


def _out_size(info, cfg):
    w, h = info["width"] or 1280, info["height"] or 720
    dar = info["dar"] or (w / h)
    ow = int(min(max(w, cfg["min_width"]), cfg["max_width"])) // 2 * 2
    oh = int(round(ow / dar / 2) * 2)
    return ow, max(oh, 2)


def _hwaccel_candidates(setting):
    if setting in ("none", "", None, False):
        return [None]
    if setting == "auto":
        return ["d3d11va", None] if sys.platform == "win32" else ["auto", None]
    return [setting, None]


def _video_filter(fps, ow, oh, scan_cfg):
    """Deinterlace (combing on text edges wrecks OCR), sample, then scale with lanczos (sharper glyphs
    than bicubic when SD is upscaled). deinterlace: auto = only frames flagged interlaced, always, off."""
    vf = []
    mode = str(scan_cfg.get("deinterlace", "auto")).lower()
    if mode in ("auto", "always", "true", "1"):
        deint = "interlaced" if mode == "auto" else "all"
        vf.append(f"bwdif=mode=send_frame:parity=auto:deint={deint}")
    vf.append(f"fps={fps}")
    vf.append(f"scale={ow}:{oh}:flags={scan_cfg.get('scale_flags', 'lanczos')}")
    return ",".join(vf)


def frames(path, start, length, fps, info, scan_cfg):
    """Yield (timestamp_seconds, BGR ndarray) sampled at `fps` from [start, start+length)."""
    ow, oh = _out_size(info, scan_cfg)
    fsize = ow * oh * 3
    vf = _video_filter(fps, ow, oh, scan_cfg)
    for hw in _hwaccel_candidates(scan_cfg.get("hwaccel", "auto")):
        if hw and _HWACCEL_OK.get(hw) is False:
            continue
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if hw:
            cmd += ["-hwaccel", hw]
        cmd += ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path), "-an", "-sn", "-dn",
                "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
        err = []
        t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
        t.start()
        n = 0
        try:
            while True:
                buf = p.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                yield start + n / fps, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
                n += 1
        finally:
            p.stdout.close()
            p.wait()
            t.join(timeout=5)
        if n == 0 and hw:
            log(f"  hwaccel {hw} failed, falling back to software decoding", 2)
            _HWACCEL_OK[hw] = False
            continue
        if hw:
            _HWACCEL_OK[hw] = True
        if p.returncode and n == 0:
            msg = (err[0] if err else b"").decode("utf-8", "replace").strip()
            log(f"  ffmpeg error: {msg[:300]}", 1)
        return
