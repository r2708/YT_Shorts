#!/usr/bin/env python3
"""Turn (game) videos into vertical shorts with OpenShorts + the Claude CLI.

Layout this script expects (it locates everything relative to itself):

    <root>/url.json     the queue: video links to turn into shorts
    <root>/openshort/   the OpenShorts repo (pipeline, patched for LLM_PROVIDER=claude-cli)
    <root>/skill/       this skill
    <root>/output/      one folder per video; finished shorts land in <run>/final/
    <root>/.venv/       Python environment (skill/requirements.txt)

No API key is used: every model call goes through `claude -p` (the logged-in
Claude Code CLI). Run it with any Python 3.9+; the pipeline itself runs in the
project venv.

    python3 skill/scripts/make_shorts.py                  # every new link in url.json
    python3 skill/scripts/make_shorts.py --watch          # keep running, pick up links as they are added
    python3 skill/scripts/make_shorts.py "<url or file>"  # one video, right now
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "openshort"
OUTPUT = ROOT / "output"
VENV_PY = ROOT / ".venv" / "bin" / "python"
SOURCES = OUTPUT / "_sources"
QUEUE = ROOT / "url.json"
STATE = OUTPUT / "processed.json"
LOCK = OUTPUT / ".make_shorts.lock"
DEFAULT_MODEL = "claude-opus-5-5"  # Opus 5.5 picks the moments, titles and hooks
MAX_CLIPS = 40  # ceiling on shorts from one video, also in "all" mode

QUEUE_TEMPLATE = {
    "defaults": {"min_score": 70, "remove_watermark": True},
    "videos": [],
    "_help": [
        "Add video links to \"videos\", then run: python3 skill/scripts/make_shorts.py",
        "A link can be plain text, or an object with its own settings, e.g.",
        "{\"url\": \"https://youtu.be/...\", \"clips\": 5, \"mode\": \"visual\"}",
        "By default every strong moment becomes a short. Allowed settings: clips (a number, or \"all\"), min_score, min_seconds, max_seconds, mode (auto|speech|visual), "
        "format (vertical|square|horizontal), captions (true|false), hook (true|false), remove_watermark (true|false: removes a channel watermark from a corner, only for footage you may republish without it), name.",
        "\"defaults\": {...} applies the same settings to every link.",
        "Finished links are remembered in output/processed.json and skipped next time; "
        "add \"redo\": true to a link to make its shorts again.",
    ],
}

# Per-video settings a url.json entry (or its "defaults") may set.
ENTRY_KEYS = {"clips": "clips", "min_score": int, "min_seconds": int, "max_seconds": int, "mode": str,
              "format": str, "captions": bool, "hook": bool, "name": str,
              "hook_seconds": float, "facecam": bool, "full_frame": bool, "model": str,
              "remove_watermark": bool}


class RunError(Exception):
    """One video could not be turned into shorts. The message is for the user."""


def slugify(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return text[:limit].strip("-") or "video"


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source or ""))


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else slugify(url, 40)


def source_key(source: str) -> str:
    """Stable identity of a queue entry: the video id for a link, the path for a file."""
    return video_id(source) if is_url(source) else str(Path(source).expanduser().resolve())


def log(msg: str = ""):
    print(msg, flush=True)


# --- environment checks -------------------------------------------------------

def has_filter(ffmpeg: str, name: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"^\s*\S+\s+{name}\s", out, re.M) is not None


def caption_ffmpeg_dir() -> str | None:
    """Folder of an ffmpeg that can burn subtitles (needs libass), or None.

    Homebrew's plain `ffmpeg` is built without libass; `ffmpeg-full` (keg-only) has it.
    Override with FFMPEG_DIR=/path/to/bin.
    """
    cands = [os.environ.get("FFMPEG_DIR"), str(Path(shutil.which("ffmpeg") or "").parent),
             "/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
    for d in cands:
        if d and Path(d, "ffmpeg").exists() and has_filter(str(Path(d, "ffmpeg")), "ass"):
            return d
    return None


def preflight(need_captions: bool) -> str | None:
    """Fail early, with the fix, on anything that would break every video."""
    if not REPO.joinpath("main.py").exists():
        raise RunError(f"OpenShorts repo not found at {REPO}")
    if not VENV_PY.exists():
        raise RunError(
            f"Python environment missing at {VENV_PY}. Create it with:\n"
            f"  python3 -m venv {ROOT / '.venv'} && "
            f"{ROOT / '.venv/bin/pip'} install -r {ROOT / 'skill/requirements.txt'}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RunError(f"{tool} is not on PATH (brew install ffmpeg-full)")
    if not shutil.which(os.environ.get("CLAUDE_CLI_BIN") or "claude"):
        raise RunError("the `claude` CLI is not on PATH (install Claude Code and log in)")
    ff_dir = caption_ffmpeg_dir()
    if need_captions and not ff_dir:
        raise RunError("this ffmpeg cannot burn captions (built without libass). "
                       "Install one that can: `brew install ffmpeg-full`, or pass --no-captions.")
    return ff_dir


# --- download ---------------------------------------------------------------

def _source_files(vid: str) -> list[Path]:
    return [p for p in SOURCES.glob(f"{vid}.*") if p.suffix in (".mp4", ".mkv", ".webm")]


def download(url: str) -> tuple[Path, str]:
    """Fetch the source once per video id into output/_sources/. Returns (path, title)."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    vid = video_id(url)
    meta = SOURCES / f"{vid}.json"
    hits = _source_files(vid)
    if hits and meta.exists():
        log(f"Source already downloaded: {hits[0]}")
        return hits[0], json.loads(meta.read_text()).get("title") or vid
    log(f"Downloading {url} ...")
    cmd = [str(VENV_PY), "-m", "yt_dlp", "--no-playlist", "--progress", "--newline",
           "--progress-template",
           "download:PROGRESS %(progress.downloaded_bytes)s %(progress.total_bytes)s "
           "%(progress.total_bytes_estimate)s %(progress.speed)s %(progress.eta)s",
           "--retries", "5", "--fragment-retries", "5",
           # 1080p60 first (gameplay looks best at 60 fps), then the best 1080p, then anything.
           "-f", "bv*[height=1080][fps>=50][vcodec^=avc1]+ba[ext=m4a]/bv*[height=1080][fps>=50]+ba/"
                 "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4",
           "-o", str(SOURCES / f"{vid}.%(ext)s"),
           "--print", "after_move:TITLE %(title)s", "--no-simulate"]
    if not shutil.which("deno") and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]  # YouTube needs a JS runtime; yt-dlp only looks for deno
    cmd.append(url)
    code, title, tail = _run_download(cmd)
    hits = _source_files(vid)
    if code != 0 or not hits:
        for part in SOURCES.glob(f"{vid}.*"):  # no half-downloaded files left behind
            if part.suffix != ".json":
                part.unlink(missing_ok=True)
        raise RunError("yt-dlp could not download the video:\n"
                       + ("\n".join(tail)[-1200:] or "no output"))
    title = title or vid
    log(f"Downloaded: {title} ({hits[0].stat().st_size / 1e6:.0f} MB)")
    meta.write_text(json.dumps({"url": url, "title": title}))
    return hits[0], title


def _num(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None  # yt-dlp prints "NA" for what it does not know yet


def _run_download(cmd: list[str]) -> tuple[int, str, list[str]]:
    """Run yt-dlp, printing progress as it goes. Returns (exit code, title, last lines).

    YouTube serves the picture and the sound as two files, so the percentage
    runs 0-100 twice: each one is labelled, and a line is printed every 10%
    (or every 15 s on a slow connection) instead of one per network chunk.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    title, tail = "", []
    part, last_done, last_step, last_print = 1, 0.0, -1, 0.0
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("TITLE "):
                title = line[6:].strip()
                continue
            if not line.startswith("PROGRESS "):
                if line:
                    tail = (tail + [line])[-25:]
                continue
            f = line.split()
            done = _num(f[1]) if len(f) > 1 else None
            total = (_num(f[2]) if len(f) > 2 else None) or (_num(f[3]) if len(f) > 3 else None)
            speed = _num(f[4]) if len(f) > 4 else None
            eta = _num(f[5]) if len(f) > 5 else None
            if done is None:
                continue
            if done < last_done:  # a new file started
                part, last_step = part + 1, -1
            last_done = done
            label = "video" if part == 1 else "audio" if part == 2 else f"part {part}"
            now = time.time()
            if total:
                pct = min(100.0, 100.0 * done / total)
                step = int(pct // 10)
                if step == last_step and now - last_print < 15 and pct < 100:
                    continue
                if step == last_step and pct >= 100:
                    continue
                last_step = step
                msg = f"   Download ({label}): {pct:3.0f}% of {total / 1e6:.0f} MB"
            else:
                if now - last_print < 15:
                    continue
                msg = f"   Download ({label}): {done / 1e6:.0f} MB"
            if speed:
                msg += f" | {speed / 1e6:.1f} MB/s"
            if eta is not None and (not total or done < total):
                msg += f" | about {int(eta) // 60}m {int(eta) % 60:02d}s left"
            last_print = now
            log(msg)
        return proc.wait(), title, tail
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def probe(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        data = json.loads(res.stdout or "{}")
        st = (data.get("streams") or [{}])[0]
        num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
        return {"duration": float(data.get("format", {}).get("duration") or 0),
                "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
                "fps": round(float(num) / float(den or 1), 2) if float(den or 1) else 0.0}
    except (ValueError, TypeError, ZeroDivisionError):
        return {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0}


# --- one video ----------------------------------------------------------------

def make_shorts(source_arg: str, opts: argparse.Namespace, ff_dir: str | None) -> dict:
    """Download (if a link), run the pipeline, collect the shorts. Returns the summary."""
    downloaded = is_url(source_arg)
    if downloaded:
        source, title = download(source_arg)
    else:
        source = Path(source_arg).expanduser().resolve()
        if not source.is_file():
            raise RunError(f"file not found: {source}")
        title = source.stem
    info = probe(source)
    if info["duration"] <= 0:
        raise RunError(f"not a readable video: {source}")
    if info["duration"] < opts.min_seconds:
        raise RunError(f"the video is {info['duration']:.0f}s, shorter than one "
                       f"{opts.min_seconds}s short")

    run_name = opts.name or f"{slugify(title, 40)}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = OUTPUT / slugify(run_name, 80)
    work = run_dir / "work"
    final = run_dir / "final"
    work.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    # The pipeline names its files after the input, so give it a short stable name.
    job_source = work / "source.mp4"
    if not job_source.exists():
        try:
            os.link(source, job_source)
        except OSError:
            shutil.copy2(source, job_source)

    env = dict(os.environ)
    if ff_dir:
        env["PATH"] = ff_dir + os.pathsep + env.get("PATH", "")
    env.update({
        "LLM_PROVIDER": "claude-cli",
        "CLAUDE_CLI_MODEL": opts.model,
        "WHISPER_MODEL": opts.whisper_model,
        # A number: exactly that many. "all" (0): every moment the scoring pass
        # rates min_score or better, however many that is.
        "CLIP_TARGET_MIN": str(opts.clips or 1),
        "CLIP_TARGET_MAX": str(opts.clips or MAX_CLIPS),
        "SHORTLIST_MIN_SCORE": "0" if opts.clips else str(opts.min_score),
        "SHORTLIST_MAX": str(MAX_CLIPS),
        "CLIP_MIN_SECONDS": str(opts.min_seconds),
        "CLIP_MAX_SECONDS": str(opts.max_seconds),
        "AUTO_CAPTIONS": "0" if opts.no_captions else "1",
        "AUTO_HOOK": "0" if opts.no_hook else "1",
        # Hook pinned to the top of the frame, small and brief, so it never
        # sits on the action; on the facecam layout that is the free band
        # above the gameplay.
        "AUTO_HOOK_POSITION": env.get("AUTO_HOOK_POSITION", "top"),
        "HOOK_TOP_Y": env.get("HOOK_TOP_Y", "0.065"),
        "AUTO_HOOK_SCALE": env.get("AUTO_HOOK_SCALE", "0.9"),
        "AUTO_HOOK_SECONDS": f"{opts.hook_seconds:g}",
        # Streamer camera pinned to an edge: show the whole game frame and the
        # camera stacked, instead of a crop that cuts the face.
        "FACECAM_LAYOUT": "0" if opts.no_facecam else "1",
        # The whole 16:9 frame at full width in every short, blur above and
        # below: nothing at the edges (a small streamer camera, the HUD) is cut.
        "FULL_FRAME_LAYOUT": "1" if opts.full_frame else "0",
        # Lower CRF on every encode: sharper burned captions and hook text.
        "X264_CRF": env.get("X264_CRF", "15"),
        "AUTO_LAYOUT": env.get("AUTO_LAYOUT", "1"),
        "EARLY_AUDIO": "0",
        "WATERMARK": "0",
        "CLIP_WORKERS": env.get("CLIP_WORKERS", "2"),
        "PYTHONUNBUFFERED": "1",
        "ASR_LOCK_DIR": str(work),
        "FFMPEG_REAL_BIN": str(Path(ff_dir, "ffmpeg")) if ff_dir else (shutil.which("ffmpeg") or "ffmpeg"),
    })
    env.pop("GEMINI_API_KEY", None)
    env.pop("REMOVELOGO_MASK", None)
    if opts.remove_watermark:
        mask = find_watermark(source, work / "watermark_mask.png")
        if mask:
            env["REMOVELOGO_MASK"] = str(mask)
    if opts.mode == "visual":
        env["MIN_SPEECH_WORDS_PER_MIN"] = "1000000"
    elif opts.mode == "speech":
        env["MIN_SPEECH_WORDS_PER_MIN"] = "0"
        env["MIN_SPEECH_WORDS"] = "0"

    cmd = [str(VENV_PY), str(REPO / "main.py"), "-i", str(job_source),
           "-o", str(work), "--format", opts.format]
    wanted = (f"{opts.clips} shorts" if opts.clips
              else f"every moment scoring {opts.min_score}+ as a short")
    log(f"Run folder: {run_dir}")
    log(f"Source: {title} ({info['duration'] / 60:.1f} min, {info['width']}x{info['height']} "
        f"@ {info['fps']:g} fps) | {wanted} of {opts.min_seconds}-{opts.max_seconds}s "
        f"| mode {opts.mode} | Claude CLI model {opts.model}")
    log_path = run_dir / "pipeline.log"
    served: dict[int, str] = {}
    started = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                m = re.match(r"CLIP_READY (\d+) (\S+)", line)
                if m:
                    served[int(m.group(1))] = m.group(2)
                # Keep the console readable: skip per-segment transcript dumps and ffmpeg noise.
                if (re.match(r"\s*\[\d+\.\d+s -> ", line) or line.startswith(("objc[", "frame="))
                        or (line.startswith(("[lib", "  ")) and "Clip" not in line)):
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()  # Ctrl-C must not leave the pipeline rendering in the background
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise

    meta_files = sorted(work.glob("*_metadata.json"))
    if code != 0 or not meta_files:
        raise RunError(f"pipeline failed (exit {code}); see {log_path}")
    meta = json.loads(meta_files[0].read_text())
    shorts = meta.get("shorts") or []
    summary = []
    for i, clip in enumerate(shorts):
        name = served.get(i)
        src = work / name if name else None
        if not src or not src.exists():
            continue
        t = clip.get("video_title_for_youtube_short") or f"clip {i + 1}"
        dest = final / f"{i + 1:02d}-{slugify(t, 50)}.mp4"
        shutil.copy2(src, dest)
        summary.append({
            "index": i + 1,
            "file": str(dest),
            "source_start": clip.get("start"),
            "source_end": clip.get("end"),
            "seconds": round(float(clip.get("end", 0)) - float(clip.get("start", 0)), 2),
            "score": clip.get("predicted_score") or clip.get("viral_score"),
            "hook": clip.get("viral_hook_text"),
            "youtube_title": t,
            "tiktok_description": clip.get("video_description_for_tiktok"),
            "instagram_description": clip.get("video_description_for_instagram"),
            "why": clip.get("why") or "",
            # Kept so the upload metadata can be (re)written after work/ is gone.
            "transcript": _clip_transcript(meta, clip.get("start"), clip.get("end")),
        })
    result = {
        "source": source_arg, "title": title, "source_minutes": round(info["duration"] / 60, 2),
        "source_video": f"{info['width']}x{info['height']} @ {info['fps']:g} fps",
        "mode": "visual" if (meta.get("transcript") or {}).get("language") == "none" else "speech",
        "llm": f"claude-cli:{opts.model}", "requested_clips": opts.clips or "all",
        "rendered_clips": len(summary), "elapsed_seconds": round(time.time() - started),
        "run_dir": str(run_dir), "shorts": summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    lines = [f"# {title}", "", f"Source: {source_arg}", f"Picked by: {result['mode']} analysis, "
             f"Claude CLI ({opts.model})", ""]
    for s in summary:
        lines += [f"## {s['index']}. {s['youtube_title']}",
                  f"- File: `{Path(s['file']).name}`",
                  f"- Source time: {s['source_start']}s to {s['source_end']}s ({s['seconds']}s)",
                  f"- Score: {s['score']}", f"- Hook: {s['hook']}",
                  f"- TikTok: {s['tiktok_description']}",
                  f"- Instagram: {s['instagram_description']}", ""]
    (run_dir / "summary.md").write_text("\n".join(lines))

    log(f"\nDONE: {len(summary)}/{len(shorts)} shorts in {final}")
    for s in summary:
        log(f"  {s['index']}. [{s['source_start']}-{s['source_end']}s] {Path(s['file']).name}")
    log(f"Summary: {run_dir / 'summary.md'}")
    if not summary:
        raise RunError(f"no short was rendered; see {log_path}")
    if not opts.no_metadata:
        try:
            write_metadata(run_dir, result, opts.model)
        except RunError as e:  # the shorts are fine; the metadata can be remade
            log(f"NOTE: metadata.json was not written ({e}). "
                f"Make it later with: --metadata {run_dir}")
    if len(summary) < len(shorts):
        raise RunError(f"{len(shorts) - len(summary)} of {len(shorts)} clip(s) did not render; "
                       f"see {log_path}. The source was kept, run again to retry.")

    # Every clip rendered: free the disk. The source and the intermediates are
    # only needed to make the shorts. A file the user passed in is never removed.
    if not opts.keep_source:
        before = _tree_bytes(work) + (_tree_bytes(source) if downloaded else 0)
        shutil.rmtree(work, ignore_errors=True)
        if downloaded:
            source.unlink(missing_ok=True)
            (SOURCES / f"{video_id(source_arg)}.json").unlink(missing_ok=True)
            try:
                SOURCES.rmdir()  # only when empty
            except OSError:
                pass
        log(f"Cleaned up the {'downloaded video and ' if downloaded else ''}working files "
            f"(about {before / 1e6:.0f} MB). Use --keep-source to keep them.")
    if opts.clips and len(shorts) < opts.clips:
        log(f"NOTE: asked for {opts.clips} shorts, the video yielded {len(shorts)} usable highlight(s); "
            f"padded or too-short picks are dropped rather than rendered. "
            f"Try a wider --min-seconds/--max-seconds band or --mode visual for more.")
    return result


def find_watermark(source: Path, mask_path: Path) -> Path | None:
    """Locate a channel watermark burned into a corner of the source and write
    the mask the pipeline uses to rebuild those pixels. None when none is found.

    Only for footage you own or are allowed to republish without the mark."""
    log("Looking for a channel watermark in the corners of the video...")
    try:
        res = subprocess.run([str(VENV_PY), str(REPO / "watermark_box.py"), str(source), str(mask_path)],
                             capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"NOTE: watermark search failed ({e}); the shorts are made without removing it.")
        return None
    box = (res.stdout or "").split()
    if res.returncode != 0 or len(box) != 4 or not mask_path.is_file():
        log("NOTE: no corner watermark found in this video; nothing is removed.")
        return None
    log(f"Watermark found at x={box[0]} y={box[1]} ({box[2]}x{box[3]} px); it is removed from every short.")
    return mask_path


def _tree_bytes(path: Path) -> int:
    """Disk bytes under a file or folder, counting a hard-linked file once."""
    seen, total = set(), 0
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


# --- upload metadata (title, description, tags, hashtags) ----------------------

META_BATCH = 15
META_SCHEMA = {
    "type": "object", "required": ["shorts"], "additionalProperties": False,
    "properties": {"shorts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["index", "title", "description", "tags", "hashtags"],
        "properties": {
            "index": {"type": "integer"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "hashtags": {"type": "array", "items": {"type": "string"}},
        }}}},
}
META_PROMPT = """You write upload metadata for short vertical gaming videos
(YouTube Shorts, Instagram Reels, TikTok). Each short below was cut from one
long video. For EVERY short return:

- "title": catchy, at most 90 characters, no hashtags in it. Name the game when
  the source title names it.
- "description": 2-3 short sentences that make someone want to watch, plain
  text, no hashtags and no links.
- "tags": 12-18 search keywords (YouTube tags) without "#": the game, the kind
  of moment, characters or places named in the short, general gaming terms.
  Lowercase. Together at most 450 characters.
- "hashtags": 6-10 hashtags, each starting with "#", no spaces, most specific
  first, and "#shorts" among them.

Rules: use ONLY what the source title and each short's own details say. Never
invent a game, a character, a player, a channel, a score or an event that is
not there. Write in the language of the short's hook. Keep the "index" of each
short exactly as given and return one entry per short.

SOURCE VIDEO TITLE: {source_title}

SHORTS (JSON):
{shorts}

Answer ONLY with a JSON object: {{"shorts": [{{"index": ..., "title": ...,
"description": ..., "tags": [...], "hashtags": [...]}}]}}. No prose."""


def _clip_transcript(meta: dict, start, end, limit: int = 1500) -> str:
    """What is said inside [start, end] of the source, from the pipeline's transcript."""
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return ""
    words = []
    for seg in (meta.get("transcript") or {}).get("segments") or []:
        try:
            if float(seg.get("end", 0)) > start and float(seg.get("start", 0)) < end:
                words.append(str(seg.get("text") or "").strip())
        except (TypeError, ValueError):
            continue
    return " ".join(w for w in words if w)[:limit]


def _claude_json(prompt: str, schema: dict, model: str) -> dict:
    """One isolated `claude -p` call that must answer JSON matching `schema`."""
    binary = os.environ.get("CLAUDE_CLI_BIN") or "claude"
    cmd = [binary, "-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence",
           "--output-format", "json", "--model", model, "--tools", "",
           "--json-schema", json.dumps(schema)]
    env = dict(os.environ)
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"):
        env.pop(k, None)
    with tempfile.TemporaryDirectory(prefix="claude-meta-") as cwd:
        try:
            res = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                 cwd=cwd, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            raise RunError("Claude CLI did not answer within 15 minutes")
        except OSError as e:
            raise RunError(f"could not run the Claude CLI: {e}")
    try:
        envelope = json.loads((res.stdout or "").strip())
    except ValueError:
        raise RunError(f"Claude CLI gave no JSON (exit {res.returncode}): "
                       f"{(res.stderr or res.stdout or '').strip()[-300:]}")
    if envelope.get("is_error"):
        raise RunError(f"Claude CLI error: {str(envelope.get('result') or '')[:300]}")
    parsed = envelope.get("structured_output")
    if not isinstance(parsed, dict):
        text = str(envelope.get("result") or "")
        lo, hi = text.find("{"), text.rfind("}")
        if lo < 0 or hi <= lo:
            raise RunError("Claude CLI answer had no JSON object")
        try:
            parsed = json.loads(text[lo:hi + 1])
        except ValueError:
            raise RunError("Claude CLI answer was not valid JSON")
    return parsed


def _clean_meta(item: dict, fallback_title: str) -> dict:
    """Normalise one model answer: plain tags, '#'-prefixed unique hashtags."""
    tags, seen = [], set()
    for t in item.get("tags") or []:
        t = re.sub(r"\s+", " ", str(t).replace("#", "")).strip().lower()
        if t and t not in seen and sum(len(x) + 1 for x in tags) + len(t) <= 480:
            seen.add(t)
            tags.append(t)
    hashtags, seen = [], set()
    for h in item.get("hashtags") or []:
        h = "#" + re.sub(r"[^\w]", "", str(h), flags=re.UNICODE)
        if len(h) > 1 and h.lower() not in seen:
            seen.add(h.lower())
            hashtags.append(h)
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip() or fallback_title
    return {"title": title[:100],
            "description": str(item.get("description") or "").strip(),
            "tags": tags, "hashtags": hashtags}


def write_metadata(run_dir: Path, result: dict, model: str) -> Path:
    """Ask Claude for title, description, tags and hashtags of every short and
    write them to <run_dir>/metadata.json. `result` is the run's summary."""
    shorts = result.get("shorts") or []
    if not shorts:
        raise RunError("no shorts to describe")
    answers: dict[int, dict] = {}
    for at in range(0, len(shorts), META_BATCH):
        batch = shorts[at:at + META_BATCH]
        brief = [{"index": s["index"], "seconds": s.get("seconds"), "hook": s.get("hook"),
                  "working_title": s.get("youtube_title"), "what_happens": s.get("why"),
                  "caption_draft": s.get("tiktok_description"),
                  "transcript": s.get("transcript") or ""} for s in batch]
        log(f"Writing titles, descriptions, tags and hashtags for shorts "
            f"{batch[0]['index']}-{batch[-1]['index']} (Claude CLI, {model})...")
        parsed = _claude_json(META_PROMPT.format(
            source_title=result.get("title") or "",
            shorts=json.dumps(brief, indent=1, ensure_ascii=False)), META_SCHEMA, model)
        for item in parsed.get("shorts") or []:
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                answers[item["index"]] = item
    out, missing = [], []
    for s in shorts:
        item = answers.get(s["index"])
        if not item:
            missing.append(s["index"])
            continue
        entry = {"index": s["index"], "file": Path(s["file"]).name}
        entry.update(_clean_meta(item, s.get("youtube_title") or f"clip {s['index']}"))
        entry["hashtags_text"] = " ".join(entry["hashtags"])
        entry["seconds"] = s.get("seconds")
        out.append(entry)
    if not out:
        raise RunError("Claude returned no metadata")
    path = run_dir / "metadata.json"
    path.write_text(json.dumps({
        "source": result.get("source"), "source_title": result.get("title"),
        "generated_by": f"claude-cli:{model}", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shorts": out}, indent=2, ensure_ascii=False))
    log(f"Metadata: {path} ({len(out)} shorts)")
    if missing:
        log(f"NOTE: no metadata came back for short(s) {missing}; "
            f"run again with --metadata {run_dir}")
    return path


def metadata_for_run(run_arg: str, model: str) -> None:
    """(Re)make metadata.json for a finished run folder, from its summary.json."""
    run_dir = Path(run_arg).expanduser()
    if not run_dir.is_dir() and (OUTPUT / run_arg).is_dir():
        run_dir = OUTPUT / run_arg
    summary = run_dir / "summary.json"
    if not summary.is_file():
        raise RunError(f"no summary.json in {run_dir}; give a finished run folder under output/")
    try:
        result = json.loads(summary.read_text())
    except ValueError:
        raise RunError(f"{summary} is not valid JSON")
    write_metadata(run_dir, result, model)


# --- url.json queue -----------------------------------------------------------

def load_queue() -> tuple[list[dict], dict]:
    """(entries, defaults) from url.json. Creates an empty template if it is missing."""
    if not QUEUE.exists():
        QUEUE.write_text(json.dumps(QUEUE_TEMPLATE, indent=2) + "\n")
        log(f"Created {QUEUE}. Add video links to \"videos\" and run again.")
        return [], {}
    try:
        data = json.loads(QUEUE.read_text() or "{}")
    except json.JSONDecodeError as e:
        raise RunError(f"{QUEUE} is not valid JSON (line {e.lineno}, column {e.colno}): {e.msg}")
    if isinstance(data, list):
        raw, defaults = data, {}
    elif isinstance(data, dict):
        raw = data.get("videos") or data.get("urls") or []
        defaults = data.get("defaults") or {}
    else:
        raise RunError(f"{QUEUE} must hold a list of links, or {{\"videos\": [...]}}")
    if not isinstance(raw, list) or not isinstance(defaults, dict):
        raise RunError(f"{QUEUE}: \"videos\" must be a list and \"defaults\" an object")
    entries, seen = [], set()
    for n, item in enumerate(raw, 1):
        entry = {"url": item} if isinstance(item, str) else item
        if not isinstance(entry, dict) or not str(entry.get("url") or "").strip():
            log(f"url.json: entry {n} skipped, it needs a link (a string, or {{\"url\": ...}})")
            continue
        entry = dict(entry, url=str(entry["url"]).strip())
        key = source_key(entry["url"])
        if key in seen:
            continue  # the same video listed twice
        seen.add(key)
        entries.append(entry)
    return entries, defaults


def entry_options(base: argparse.Namespace, defaults: dict, entry: dict) -> argparse.Namespace:
    """Command-line options, overridden by url.json defaults, then by the entry."""
    opts = argparse.Namespace(**vars(base))
    for src in (defaults, entry):
        for key, kind in ENTRY_KEYS.items():
            if key not in src or src[key] is None:
                continue
            value = src[key]
            try:
                if kind == "clips":
                    value = parse_clips(value)
                else:
                    value = bool(value) if kind is bool else kind(value)
            except (TypeError, ValueError):
                want = "a number or \"all\"" if kind == "clips" else kind.__name__
                raise RunError(f"url.json: \"{key}\" should be {want}, got {src[key]!r}")
            if key == "captions":
                opts.no_captions = not value
            elif key == "hook":
                opts.no_hook = not value
            elif key == "facecam":
                opts.no_facecam = not value
            else:
                setattr(opts, key, value)
    validate(opts)
    return opts


def parse_clips(value) -> int:
    """"all" (every strong moment) -> 0, otherwise the exact number asked for."""
    if isinstance(value, str) and value.strip().lower() in ("all", "auto", "every"):
        return 0
    if isinstance(value, bool):
        raise ValueError(value)
    return int(value)


def validate(opts: argparse.Namespace):
    if opts.clips < 0 or opts.clips > MAX_CLIPS:
        raise RunError(f"clips must be \"all\" or a number from 1 to {MAX_CLIPS}")
    if not 1 <= opts.min_score <= 100:
        raise RunError("min_score must be between 1 and 100")
    if opts.min_seconds < 5 or opts.max_seconds > 180 or opts.min_seconds > opts.max_seconds:
        raise RunError("clip length must satisfy 5 <= min_seconds <= max_seconds <= 180")
    if opts.mode not in ("auto", "speech", "visual"):
        raise RunError("mode must be auto, speech or visual")
    if opts.format not in ("vertical", "square", "horizontal"):
        raise RunError("format must be vertical, square or horizontal")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(STATE)


def pending(entries: list[dict], state: dict) -> list[dict]:
    """Entries that still need shorts: new ones, failed ones, and "redo": true."""
    todo = []
    for e in entries:
        rec = state.get(source_key(e["url"]))
        done = rec and rec.get("status") == "done" and Path(rec.get("run_dir", "")).is_dir()
        if not done or e.get("redo"):
            todo.append(e)
    return todo


def run_queue(base: argparse.Namespace, ff_dir: str | None, announce_empty: bool = True) -> int:
    """Make shorts for every pending url.json entry. Returns how many failed."""
    entries, defaults = load_queue()
    state = load_state()
    todo = pending(entries, state)
    if not todo:
        if announce_empty:
            log(f"Nothing to do: {len(entries)} link(s) in {QUEUE.name}, all already have shorts."
                if entries else f"No links in {QUEUE}. Add some to \"videos\".")
        return 0
    log(f"{len(todo)} video(s) to process from {QUEUE.name} ({len(entries) - len(todo)} already done).")
    failed = 0
    for n, entry in enumerate(todo, 1):
        url, key = entry["url"], source_key(entry["url"])
        log(f"\n===== [{n}/{len(todo)}] {url} =====")
        rec = {"url": url, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            result = make_shorts(url, entry_options(base, defaults, entry), ff_dir)
            rec.update(status="done", title=result["title"], run_dir=result["run_dir"],
                       shorts=result["rendered_clips"], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            if entry.get("redo"):
                clear_redo(url)
        except RunError as e:
            failed += 1
            rec.update(status="failed", error=str(e)[:600],
                       attempts=int((state.get(key) or {}).get("attempts", 0)) + 1)
            log(f"FAILED: {e}")
        state = load_state()  # another edit may have landed during a long render
        state[key] = rec
        save_state(state)
    log(f"\nQueue finished: {len(todo) - failed} done, {failed} failed. Record: {STATE}")
    return failed


def clear_redo(url: str):
    """Drop "redo": true from a finished entry so it is not remade on every run."""
    try:
        data = json.loads(QUEUE.read_text())
        items = data if isinstance(data, list) else (data.get("videos") or data.get("urls") or [])
        changed = False
        for item in items:
            if isinstance(item, dict) and str(item.get("url", "")).strip() == url and item.pop("redo", None):
                changed = True
        if changed:
            QUEUE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass


def watch(base: argparse.Namespace, ff_dir: str | None, every: int):
    """Stay running; whenever url.json changes, make shorts for the new links."""
    log(f"Watching {QUEUE} (checked every {every}s). Add a link and save the file. Ctrl-C to stop.")
    last = None
    # Failed links are retried only when url.json changes again, not on every tick.
    while True:
        try:
            stamp = QUEUE.stat().st_mtime_ns if QUEUE.exists() else 0
        except OSError:
            stamp = 0
        if stamp != last:
            try:
                run_queue(base, ff_dir, announce_empty=last is None)
            except RunError as e:
                log(f"ERROR: {e}")
            try:
                last = QUEUE.stat().st_mtime_ns if QUEUE.exists() else 0
            except OSError:
                last = 0
            log(f"Waiting for new links in {QUEUE.name} ...")
        time.sleep(every)


# --- single-instance lock -----------------------------------------------------

def acquire_lock():
    """Two runs at once would fight over CPU, memory and the same source files."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or 0)
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            pass  # stale lock from a run that died
        except PermissionError:
            raise RunError(f"another make_shorts run is active (pid in {LOCK})")
        else:
            raise RunError(f"another make_shorts run is active (pid {pid}). "
                           f"Wait for it, or stop it first.")
    LOCK.write_text(str(os.getpid()))


def release_lock():
    try:
        if LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?",
                    help="a video link or local file to process now; omit to process url.json")
    ap.add_argument("--watch", action="store_true",
                    help="keep running and process links as they are added to url.json")
    ap.add_argument("--watch-interval", type=int, default=30, help="seconds between url.json checks (default 30)")
    ap.add_argument("--clips", type=parse_clips, default=0, metavar="all|N",
                    help="\"all\" (default): a short for every strong moment in the video; "
                         "or an exact number of shorts")
    ap.add_argument("--min-score", type=int, default=70,
                    help="with --clips all: how strong (0-100) a moment must score to become a short (default 70)")
    ap.add_argument("--min-seconds", type=int, default=20, help="shortest clip (default 20)")
    ap.add_argument("--max-seconds", type=int, default=45, help="longest clip (default 45)")
    ap.add_argument("--mode", choices=["auto", "speech", "visual"], default="auto",
                    help="auto: transcript when there is commentary, frames+audio when not; "
                         "visual: always pick highlights from the footage itself; "
                         "speech: always use the transcript")
    ap.add_argument("--model", default=os.environ.get("CLAUDE_CLI_MODEL", DEFAULT_MODEL),
                    help=f"Claude CLI model that analyses the video: an id or an alias "
                         f"like sonnet/haiku (default {DEFAULT_MODEL})")
    ap.add_argument("--whisper-model", default=os.environ.get("WHISPER_MODEL", "small"),
                    help="speech model: tiny/base (fast, low memory), small (default), medium (best, slow)")
    ap.add_argument("--format", default="vertical", choices=["vertical", "square", "horizontal"])
    ap.add_argument("--no-captions", action="store_true", help="do not burn subtitles")
    ap.add_argument("--no-hook", action="store_true", help="do not burn the hook text overlay")
    ap.add_argument("--hook-seconds", type=float, default=3.0,
                    help="how long the hook text stays at the top of the frame (default 3)")
    ap.add_argument("--no-facecam", action="store_true",
                    help="do not use the stacked game + streamer-camera layout")
    ap.add_argument("--full-frame", action=argparse.BooleanOptionalAction, default=True,
                    help="show the whole source frame at full width with blur above and below "
                         "(default: on; --no-full-frame crops to fill the 9:16 frame)")
    ap.add_argument("--remove-watermark", action=argparse.BooleanOptionalAction, default=True,
                    help="find a channel watermark burned into a corner of the video and remove it "
                         "from the shorts (default: on; --no-remove-watermark keeps it). "
                         "Only for footage you may republish without it")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the downloaded video and working files after the shorts are made")
    ap.add_argument("--no-metadata", action="store_true",
                    help="do not write metadata.json (title, description, tags, hashtags per short)")
    ap.add_argument("--metadata", metavar="RUN_DIR",
                    help="only (re)write metadata.json for a finished run folder under output/, then exit")
    ap.add_argument("--name", help="run folder name under output/ (single video only; default: title + time)")
    args = ap.parse_args()

    if args.metadata:
        try:
            metadata_for_run(args.metadata, args.model)
            return 0
        except RunError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    try:
        validate(args)
        if args.watch and args.source:
            raise RunError("--watch reads url.json; do not pass a video with it")
        ff_dir = preflight(need_captions=not args.no_captions)
        acquire_lock()
    except RunError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    try:
        if args.source:
            make_shorts(args.source, args, ff_dir)
            return 0
        args.name = None  # a queue of videos cannot share one folder name
        if args.watch:
            watch(args, ff_dir, max(5, args.watch_interval))
            return 0
        return 2 if run_queue(args, ff_dir) else 0
    except RunError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
