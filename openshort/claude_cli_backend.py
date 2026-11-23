"""Claude Code CLI as the LLM: no Gemini, OpenAI or Anthropic API key needed.

``LLM_PROVIDER=claude-cli`` routes every model call of the pipeline through the
locally installed, already logged-in ``claude`` binary in print mode
(``claude -p``), so the job runs on the user's Claude subscription:

- the two transcript passes of ``main.get_viral_clips`` (via ``llm_backend``),
- the silent/low-speech vision path (``main.get_visual_clips``): Claude cannot
  take a video file, so it is shown timestamped contact sheets of the footage
  plus a loudness curve of the audio (``visual_pass``),
- the frame-based helpers (``layout_picker``, ``screencast_layout``,
  ``hook_grounding``), which hand their JPEG stills to ``generate_json``.

Each call is one fresh, isolated CLI process: safe mode (no CLAUDE.md, plugins,
hooks or MCP servers of the host), no session persistence, structured output
enforced with ``--json-schema`` and validated again with the same pydantic
model Gemini's ``response_schema`` uses. Images are written to a private temp
directory that is the process's working directory, and ``Read`` is the only
tool it gets.

Env:
    CLAUDE_CLI_BIN      path of the binary (default: ``claude`` on PATH)
    CLAUDE_CLI_MODEL    model alias or id (default: ``sonnet``)
    CLAUDE_CLI_TIMEOUT  seconds per call (default: 900)
    CLAUDE_VISION_SEGMENT_SECONDS / CLAUDE_VISION_FRAME_SECONDS  vision sampling
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 900.0
PROVIDER_NAMES = ("claude-cli", "claude_cli", "claude", "claude-code")

# Vision sampling: one frame every FRAME_SECONDS, tiled GRID x GRID per sheet,
# one CLI call per SEGMENT_SECONDS of footage (10 min at 3 s = 200 frames =
# 13 sheets of 16). Sheets stay under ~1568 px so the API does not downscale.
VISION_SEGMENT_SECONDS = float(os.environ.get("CLAUDE_VISION_SEGMENT_SECONDS", "600"))
VISION_MAX_SEGMENTS = int(os.environ.get("CLAUDE_VISION_MAX_SEGMENTS", "12"))
VISION_FRAME_SECONDS = float(os.environ.get("CLAUDE_VISION_FRAME_SECONDS", "3"))
SHEET_GRID = 4
TILE_W, TILE_H = 390, 220

Image = Union[bytes, Tuple[str, bytes]]


def active() -> bool:
    return (os.environ.get("LLM_PROVIDER") or "").strip().lower() in PROVIDER_NAMES


def binary() -> str:
    return (os.environ.get("CLAUDE_CLI_BIN") or "").strip() or "claude"


def model_name() -> str:
    return (os.environ.get("CLAUDE_CLI_MODEL") or "").strip() or DEFAULT_MODEL


def available() -> bool:
    return shutil.which(binary()) is not None


def _timeout() -> float:
    try:
        return float(os.environ.get("CLAUDE_CLI_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _inline_refs(schema: dict) -> dict:
    """Pydantic emits ``$defs`` + ``$ref``; inline them so the CLI's schema
    validator gets one plain object whatever it supports."""
    defs = schema.get("$defs") or {}

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return walk(defs.get(ref.split("/")[-1], {}))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _extract_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except ValueError:
        lo, hi = text.find("{"), text.rfind("}")
        if lo < 0 or hi <= lo:
            raise RuntimeError("Claude CLI answer did not contain a JSON object")
        return json.loads(text[lo:hi + 1])


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  images: Optional[Sequence[Image]] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One ``claude -p`` call that must answer JSON matching ``schema``.

    ``images``: JPEG bytes, or ``(label, bytes)`` pairs, shown to the model in
    order. Returns ``(parsed_dict, cost_analysis)`` like
    ``llm_backend.generate_json``. Raises RuntimeError on any failure; the
    messages carry the tokens ``main._run_gemini_stage`` treats as transient.
    """
    if not available():
        raise RuntimeError(
            f"Claude CLI not found ('{binary()}'). Install Claude Code and log in, "
            "or set CLAUDE_CLI_BIN.")
    model = model or model_name()
    workdir = tempfile.mkdtemp(prefix="claude-llm-")
    try:
        full_prompt = prompt
        tools = ""
        if images:
            lines = []
            for i, img in enumerate(images):
                label, data = img if isinstance(img, tuple) else (f"Image {i + 1}", img)
                name = f"img_{i:03d}.jpg"
                with open(os.path.join(workdir, name), "wb") as f:
                    f.write(data)
                lines.append(f"- {label}: ./{name}")
            full_prompt = (
                "You are given image files in the current directory. Use the Read "
                "tool to LOOK at every one of them, in order, before answering "
                "(read several per turn). Do not answer from the file names.\n"
                + "\n".join(lines) + "\n\n" + prompt)
            tools = "Read"
        full_prompt += ("\n\nAnswer ONLY with the JSON object described above, "
                        "matching the required schema. No prose.")
        cmd = [binary(), "-p", "--safe-mode", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json",
               "--model", model, "--tools", tools,
               "--json-schema", json.dumps(_inline_refs(schema.model_json_schema()))]
        if tools:
            cmd += ["--allowedTools", tools]
        env = dict(os.environ)
        # A nested call must not inherit the parent session's identity.
        for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"):
            env.pop(k, None)
        try:
            res = subprocess.run(cmd, input=full_prompt, capture_output=True, text=True,
                                 cwd=workdir, env=env, timeout=_timeout())
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude CLI Deadline exceeded after {_timeout():.0f}s")
        out = (res.stdout or "").strip()
        if not out:
            raise RuntimeError("Claude CLI empty response body "
                               f"(exit {res.returncode}): {(res.stderr or '').strip()[-300:]}")
        try:
            envelope = json.loads(out)
        except ValueError:
            raise RuntimeError(f"Claude CLI empty response body (not JSON): {out[:200]}")
        if envelope.get("is_error"):
            detail = str(envelope.get("result") or envelope.get("subtype") or "")[:300]
            status = envelope.get("api_error_status")
            tag = "overloaded " if status in (429, 500, 529, 503) else ""
            raise RuntimeError(f"Claude CLI {tag}error: {detail}")
        parsed = envelope.get("structured_output")
        if not isinstance(parsed, dict):
            parsed = _extract_json(str(envelope.get("result") or ""))
        validated = schema.model_validate(parsed).model_dump()
        usage = envelope.get("usage") or {}
        cost = {
            "input_tokens": int(usage.get("input_tokens") or 0)
                            + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "thinking_tokens": 0,
            # Runs on the Claude subscription: nothing is billed per call. The
            # CLI's list-price figure is kept for information only.
            "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0,
            "list_price_usd": float(envelope.get("total_cost_usd") or 0.0),
            "model": f"claude-cli:{model}",
            "price_estimated": False,
            "local": True,
        }
        return validated, cost
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Vision: pick clips from footage with little or no speech (gameplay, montages)
# ---------------------------------------------------------------------------

VISUAL_PROMPT = """
You are a senior short-form video editor cutting highlights for TikTok / Reels /
YouTube Shorts. This footage has little or no usable speech (typically gameplay
or a montage), so judge it by what you SEE and by how loud it gets.

The images are CONTACT SHEETS of the footage: one frame every {step:g} seconds,
left to right, top to bottom. Every tile has its timestamp in SECONDS burned
into its top-left corner. The footage shown runs from 0 to {video_duration:g} s.
{audio}
Pick the {min_clips}-{max_clips} MOST engaging moments: kills and multi-kills,
clutches, boss fights and finishing blows, wins and losses, big plays, crashes,
fails, jump scares, reveals, cutscene payoffs, funny or striking shots. Prefer a
moment with a build-up and a payoff over a static menu, loading screen, lobby,
inventory or map. Loudness peaks usually mark the action; confirm them on the
frames. Spread the picks over the footage instead of clustering them.

TIME CONTRACT - STRICT:
- Timestamps in ABSOLUTE SECONDS as burned into the tiles (usable with ffmpeg -ss/-to).
- 0 <= start < end <= {video_duration:g}.
- Each clip {min_secs:g} to {max_secs:g} seconds long. If the footage is shorter
  than {min_secs:g}s, return one clip spanning all of it.
- Start a few seconds BEFORE the action begins so the build-up is in, and end
  just after the payoff lands. Never start or end on a menu or loading screen.

For each clip write catchy copy in {language}: `viral_hook_text` (max 10 words,
names what actually happens on screen), a TikTok and an Instagram description
with hashtags, a YouTube title of at most 100 characters, and `predicted_score`
0-100 for how likely it stops a viewer scrolling. Describe only what the frames
show: never invent a game title, player name or score you cannot read on screen.
Order clips best to worst.

Return {{"shorts": [{{"start": ..., "end": ..., "predicted_score": ...,
"video_description_for_tiktok": ..., "video_description_for_instagram": ...,
"video_title_for_youtube_short": ..., "viral_hook_text": ...}}]}}.
"""


def vision_segment_seconds() -> float:
    return VISION_SEGMENT_SECONDS


def _contact_sheets(video_path: str, start: float, end: float, step: float) -> List[Tuple[str, bytes]]:
    """Timestamped contact sheets of ``[start, end]``; times are relative to ``start``."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    tiles = []
    try:
        t = 0.0
        while start + t < end - 0.05:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                scale = min(TILE_W / w, TILE_H / h)
                small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)
                tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                y0 = (TILE_H - small.shape[0]) // 2
                x0 = (TILE_W - small.shape[1]) // 2
                tile[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
                label = f"{t:.0f}s"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(tile, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)
                cv2.putText(tile, label, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2, cv2.LINE_AA)
                tiles.append((t, tile))
            t += step
    finally:
        cap.release()

    per = SHEET_GRID * SHEET_GRID
    sheets = []
    for i in range(0, len(tiles), per):
        chunk = tiles[i:i + per]
        canvas = np.zeros((SHEET_GRID * TILE_H, SHEET_GRID * TILE_W, 3), dtype=np.uint8)
        for k, (_, tile) in enumerate(chunk):
            r, c = divmod(k, SHEET_GRID)
            canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            sheets.append((f"Contact sheet {len(sheets) + 1} "
                           f"({chunk[0][0]:.0f}s to {chunk[-1][0]:.0f}s)", buf.tobytes()))
    return sheets


def _loudness(video_path: str, start: float, end: float, bucket: float = 2.0) -> List[int]:
    """Audio level per ``bucket`` seconds as 0-9 (9 = loudest of this stretch),
    or [] when the file has no decodable audio."""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-i", video_path, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception:
        return []
    pcm = np.frombuffer(res.stdout or b"", dtype=np.int16).astype(np.float32)
    n = int(8000 * bucket)
    if pcm.size < n:
        return []
    rms = np.sqrt((pcm[:pcm.size // n * n].reshape(-1, n) ** 2).mean(axis=1) + 1e-9)
    db = 20 * np.log10(rms / 32768.0 + 1e-9)
    lo, hi = float(np.percentile(db, 5)), float(db.max())
    if hi - lo < 3:  # flat track: music bed or silence, no signal in it
        return []
    return [int(v) for v in np.clip((db - lo) / (hi - lo) * 9, 0, 9).round()]


def visual_pass(video_path: str, start: float, end: float, language: str,
                min_clips: int, max_clips: int, min_secs: float, max_secs: float,
                hint: str = "", schema: Optional[Type[BaseModel]] = None):
    """Pick clips inside ``[start, end]`` of ``video_path`` from contact sheets.

    Returns ``(shorts, cost)`` with times relative to ``start`` and clamped to
    the stretch, the contract of ``main._visual_pass``."""
    import gemini_worker

    duration = round(float(end) - float(start), 3)
    step = max(VISION_FRAME_SECONDS, duration / 320.0)  # never more than 20 sheets
    sheets = _contact_sheets(video_path, start, end, step)
    if not sheets:
        raise RuntimeError("no readable frames for the vision pass")
    levels = _loudness(video_path, start, end)
    audio = ""
    if levels:
        audio = ("\nAUDIO_LOUDNESS, one digit per 2 seconds from second 0 "
                 "(0 = quiet, 9 = loudest): " + "".join(str(v) for v in levels) + "\n")
    lang = language if language and language not in ("none", "unknown") else "English"
    prompt = VISUAL_PROMPT.format(
        step=step, video_duration=duration, audio=audio, language=lang,
        min_clips=min_clips, max_clips=max_clips,
        min_secs=min_secs, max_secs=max_secs) + (hint or "")
    parsed, cost = generate_json(prompt, schema or gemini_worker.VisualResponse, images=sheets)
    clean = []
    for s in parsed.get("shorts") or []:
        s["start"] = max(0.0, float(s.get("start", 0)))
        s["end"] = min(duration, float(s.get("end", 0)))
        if s["end"] - s["start"] > max_secs:
            s["end"] = s["start"] + max_secs
        if s["end"] - s["start"] >= 1.0:
            clean.append(s)
    return clean, cost


def segment_count(duration: float) -> int:
    return max(1, min(VISION_MAX_SEGMENTS, math.ceil(float(duration) / VISION_SEGMENT_SECONDS)))
