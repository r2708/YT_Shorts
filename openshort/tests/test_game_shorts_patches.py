"""Changes made for the game-shorts skill: placeholder clips, facecam layout,
the CRF override and the top-pinned hook."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_inset
import ffmpeg_utils


def _clip(start, end, title, hook, wid):
    return {"start": start, "end": end, "video_title_for_youtube_short": title,
            "viral_hook_text": hook, "source_window_id": wid}


def test_drop_placeholder_clips_removes_padding_only():
    import main
    clips = [
        _clip(160, 205, "They Came to Collect", "Why not him?", "w3"),
        _clip(1560, 1569.14, "x", "x", "w24"),          # the GTA 6 padding clip
        _clip(10, 40, "Real title", "-", "w1"),          # filler hook
        _clip(10, 40, "खेल खत्म", "अब क्या?", "w2"),      # non-Latin text is real text
        _clip(50, 55, "Real title", "Real hook", "w5"),  # far under the floor
    ]
    kept, dropped = main.drop_placeholder_clips(clips, 20)
    assert [c["source_window_id"] for c in kept] == ["w3", "w2"]
    assert dropped == {"w24", "w1", "w5"}


def test_drop_placeholder_clips_handles_empty():
    import main
    assert main.drop_placeholder_clips([], 20) == ([], set())
    assert main.drop_placeholder_clips(None, 20) == ([], set())


def test_facecam_filtergraph_keeps_whole_frame_and_fits():
    # 4:3 camera pinned to the left edge of a 1080p stream (PUBG test clip).
    graph = camera_inset.facecam_filtergraph(1920, 1080, 1080, 1920, (4, 487, 374, 282))
    assert "scale=1080:608" in graph                      # full game width, nothing cropped
    assert "crop=w=374:h=282:x=4:y=486" in graph          # the camera, even coordinates
    top = int(graph.split("overlay=x=0:y=")[1].split("[")[0])
    cam_h = int(graph.split("scale=")[-1].split(":")[1].split("[")[0].split(":")[0])
    assert top >= 0.15 * 1920                             # room for the hook above the game
    assert top + 608 + cam_h <= 1920                      # everything inside the frame
    assert cam_h <= 0.30 * 1920 + 2


def test_facecam_filtergraph_wide_camera_never_exceeds_width():
    graph = camera_inset.facecam_filtergraph(1920, 1080, 1080, 1920, (1400, 0, 520, 200))
    cam_w = int(graph.split("y=0,scale=")[1].split(":")[0]) if "y=0,scale=" in graph else 0
    assert 0 < cam_w <= 1080


def test_x264_crf_override_only_lowers(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    monkeypatch.setenv("X264_CRF", "15")
    args = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    assert args[args.index("-crf") + 1] == "15"
    monkeypatch.setenv("X264_CRF", "30")                  # never makes it worse
    args = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    assert args[args.index("-crf") + 1] == "18"
    monkeypatch.setenv("X264_CRF", "junk")
    args = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    assert args[args.index("-crf") + 1] == "18"


def test_facecam_layout_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("FACECAM_LAYOUT", raising=False)
    assert importlib.reload(camera_inset).FACECAM_ENABLED is False
    monkeypatch.setenv("FACECAM_LAYOUT", "1")
    assert importlib.reload(camera_inset).FACECAM_ENABLED is True
    monkeypatch.delenv("FACECAM_LAYOUT", raising=False)
    importlib.reload(camera_inset)
