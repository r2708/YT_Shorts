---
name: game-shorts
description: Turn a long gaming video (YouTube link or local file) into vertical 9:16 Shorts / Reels / TikToks. Finds the key highlights and renders captioned, hooked clips locally with OpenShorts, using the Claude CLI as the LLM (no OpenAI, Gemini or Anthropic API key). Use when the user shares a game video link and asks for shorts, reels, clips or highlights.
---

# Game Shorts

Makes shorts from a video with the local OpenShorts pipeline. Every model call
(highlight picking, titles, hooks, layout choice) goes through `claude -p`, so
it runs on the logged-in Claude Code subscription. Nothing is uploaded anywhere
and no API key is needed.

## Folders

```
<root>/url.json     the queue: video links to turn into shorts
<root>/openshort/   OpenShorts repo, patched: LLM_PROVIDER=claude-cli (claude_cli_backend.py)
<root>/skill/       this skill (SKILL.md, scripts/make_shorts.py, requirements.txt)
<root>/output/      results, one folder per video; processed.json remembers finished links
<root>/.venv/       Python environment for the pipeline
```

`<root>` is the folder that contains this `skill/` directory. Resolve it from
this file's real path; do not assume the current directory.

## Make shorts

A 25-minute source takes roughly 12-20 minutes on a laptop (transcription and
rendering are local). Run it in the background and report when it finishes.
Only one run at a time: a second one refuses to start while the first is active.

### From url.json (the normal way)

`<root>/url.json` holds the links. When the user gives links, add them to
`"videos"` (keep what is already there) and run the script with no source:

```json
{
  "defaults": { "min_score": 70 },
  "videos": [
    "https://youtu.be/XXXXXXXXXXX",
    { "url": "https://youtu.be/YYYYYYYYYYY", "clips": 5, "mode": "visual" }
  ]
}
```

```bash
python3 <root>/skill/scripts/make_shorts.py            # every link that has no shorts yet
python3 <root>/skill/scripts/make_shorts.py --watch    # stay running; new links are picked up when the file is saved
```

- By default every strong moment of a video becomes a short; the count is not
  fixed. Set `"clips": 5` on a link only when the user wants an exact number.
- A link can be a plain string or an object with its own settings: `clips`,
  `min_score`,
  `min_seconds`, `max_seconds`, `mode`, `format`, `captions`, `hook`,
  `hook_seconds`, `facecam`, `full_frame`, `name`. `"defaults"` applies to every link.
- Finished links are recorded in `output/processed.json` and skipped next time.
  A failed link is retried on the next run. `"redo": true` on a link makes its
  shorts again (the flag is removed once it succeeds).
- One bad link does not stop the others; the script exits 2 if any failed and
  the reason is in `processed.json`.

### One video right now

```bash
python3 <root>/skill/scripts/make_shorts.py "<video url or file path>"
```

This does not touch url.json or processed.json.

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--clips all\|N` | all | `all`: a short for every strong moment in the video (however many there are, up to 40); a number: exactly that many |
| `--min-score` | 70 | with `all`: how strong (0-100) a moment must score to become a short. Raise it for fewer, stronger shorts; lower it for more |
| `--min-seconds` / `--max-seconds` | 20 / 45 | clip length band |
| `--mode auto\|speech\|visual` | auto | how highlights are found (below) |
| `--model` | claude-fable-5-2 | Claude CLI model that analyses the video (Fable 5.2 by default). `sonnet` or `haiku` are faster and lighter on the subscription's usage limits |
| `--whisper-model` | small | speech model; use `base` or `tiny` on a machine with 8 GB RAM or when transcription is slow |
| `--format` | vertical | `vertical` 9:16, `square` 1:1, `horizontal` 16:9 |
| `--no-captions`, `--no-hook` | off | skip burned subtitles / the hook text overlay |
| `--hook-seconds` | 3 | how long the hook text stays on screen (top of the frame) |
| `--no-facecam` | off | turn off the stacked game + streamer-camera layout |
| `--no-full-frame` | full frame on | by default the whole 16:9 source frame is shown at full width with blur above and below, so nothing at the edges is cut (url.json: `"full_frame": true`). `--no-full-frame` goes back to crops that fill more of the 9:16 frame but trim the sides |
| `--no-remove-watermark` | removal on | by default the script will find a channel watermark burned into a corner of the source and rebuild those pixels in every short (url.json: `"remove_watermark": true`). The user has confirmed they hold the rights to their footage; pass `--no-remove-watermark` (url.json: `"remove_watermark": false`) to keep the mark |
| `--no-clean-source` | clean-up on | by default black cinematic bars and the source's own burned-in subtitles (game dialogue subtitles) are cut off the source before the shorts are made, so a short fills the frame and carries one layer of captions (url.json: `"clean_source": true`). Costs one re-encode of the source, only when something is found |
| `--keep-source` | off | keep the downloaded video and the `work/` intermediates; by default both are deleted once every short has rendered |
| `--name` | title + time | run folder name under `output/` |

Downloads prefer 1080p at 60 fps and fall back to the best 1080p; shorts keep
the source frame rate.

### Layout, hook and captions

- **Streamer facecam**: when a webcam window is pinned to an edge of the stream,
  the clip is laid out as: free band (hook text), the whole game frame at full
  width, the camera enlarged below it, free band (captions). The face is never
  cut by the 9:16 crop. If no stable webcam window is found, the whole 16:9
  frame is still shown at full width over a blurred backdrop (full-frame
  layout), so a small camera inside the picture stays whole; only
  `--no-full-frame` brings back the side-trimming crop layouts.
- **Hook**: pinned to the top of the frame, small, 3 seconds, so it does not sit
  on the action. On the facecam layout it is in the band above the gameplay.
- **Sharpness**: every encode runs at CRF 15 (`X264_CRF`), which keeps burned
  text crisp; files are larger than the pipeline default.
- Burning captions needs an ffmpeg built with libass. Homebrew's plain `ffmpeg`
  has none: `brew install ffmpeg-full` (the script finds it by itself).

### Picking the mode for a game video

- `auto` (default): if the video has commentary, highlights come from the
  transcript (what the player or caster says, scored window by window, then cut
  on word boundaries). If speech is missing or sparse it switches to `visual`
  by itself.
- `visual`: Claude looks at timestamped contact sheets of the footage (one
  frame every 3 s) plus a loudness curve of the audio, and picks kills,
  clutches, boss fights, wins, fails and other on-screen payoffs. Use it for
  no-commentary gameplay, montages, or when the user wants moments chosen by
  what happens on screen rather than by what is said.
- `speech`: force the transcript path.

When the user says only "make shorts from this game video", use `auto`. If the
transcript picks come back as talk about the game rather than gameplay moments,
re-run with `--mode visual`.

## Output

```
output/<run>/final/01-<title>.mp4 ...   the finished shorts (captions + hook burned in)
output/<run>/summary.md                 per short: source time range, score, hook, YouTube title, TikTok / Instagram descriptions
output/<run>/summary.json               the same, machine-readable
output/<run>/metadata.json              upload copy per short, written by Claude: title, description, tags (keywords), hashtags
                                        (skip with --no-metadata; remake for a finished run with --metadata output/<run>)
output/<run>/pipeline.log               full pipeline log
output/<run>/work/                      intermediates; deleted after a fully successful run unless --keep-source
output/_sources/                        downloaded sources; deleted after a fully successful run unless --keep-source
```

After a run, tell the user: how many shorts were made, the folder, and for each
short its title, source time range and hook (read `summary.md`). The titles,
hooks and descriptions are model-written suggestions: say so, and do not
present them as verified facts about the game or the players.

## Checking the result

Before reporting success, confirm the files exist and are real videos:

```bash
for f in <root>/output/<run>/final/*.mp4; do ffprobe -v error -show_entries stream=codec_type,width,height:format=duration -of csv=p=0 "$f"; done
```

Each should be 1080x1920 (for `vertical`) with a video and an audio stream and a
duration inside the requested band. To eyeball a pick, grab a frame:
`ffmpeg -ss 5 -i <clip> -frames:v 1 /tmp/frame.jpg` and look at it.

## Troubleshooting

- `Python environment missing`: `python3 -m venv <root>/.venv && <root>/.venv/bin/pip install -r <root>/skill/requirements.txt` (it pins `av`, which the repo's own file leaves loose and which breaks transcription when it floats).
- `yt-dlp could not download`: update it (`<root>/.venv/bin/pip install -U yt-dlp`) and retry; for age-restricted or private videos ask the user for the file and pass its path instead of the URL.
- `Claude CLI ... error`: run `claude -p "hi"` to check the login; a rate limit clears on its own, re-run later (the downloaded source is reused).
- A clip failed to render: the script reports it and keeps the source; run again to retry.
- `another make_shorts run is active`: wait for it to finish, or stop that process first.
- Do not set `GEMINI_API_KEY`; the script removes it so the run never leaves the Claude CLI.
