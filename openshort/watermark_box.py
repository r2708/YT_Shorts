"""Find a channel watermark burned into a corner of a video.

A watermark is the one thing in a corner that never moves or changes while the
picture behind it does. Frames are sampled across the whole file; an edge that
is present in nearly all of them, inside a corner zone, belongs to the mark.

    python watermark_box.py <video>              prints "x y w h" (source pixels) or nothing
    python watermark_box.py <video> <mask.png>   also writes the mask of the mark's own pixels

The mask is for ffmpeg's ``removelogo``, which rebuilds only the masked pixels
from their neighbours: a boxed ``delogo`` smears whatever HUD text shares the
corner. ``ffmpeg_utils.cut_clip`` applies it when ``REMOVELOGO_MASK`` is set.
"""
from __future__ import annotations

import sys

ZONE_W, ZONE_H = 0.30, 0.16     # corner zone searched, as a share of the frame
MAX_W, MAX_H = 0.26, 0.10       # a larger persistent shape is a HUD, not a watermark
SAMPLES = 36


def _frames(video_path: str, samples: int):
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
            ok, frame = cap.read()
            if ok and frame is not None:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def detect(video_path: str, samples: int = SAMPLES):
    """``(x, y, w, h)`` of the watermark in source pixels, or None."""
    import cv2
    import numpy as np

    edges, lumas = [], []
    for gray in _frames(video_path, samples):
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edges.append(cv2.magnitude(gx, gy))
        lumas.append(g)
    if len(edges) < 8:
        return None
    stack = np.stack(edges)
    h, w = stack.shape[1:]
    # Present in 4 frames out of 5: survives the scenes where the mark sits on
    # a background of its own brightness.
    persistent = np.percentile(stack, 20, axis=0)
    # A static picture (menu, black bars, a paused shot) is persistent too:
    # only keep pixels whose surroundings actually change between frames.
    moving = cv2.blur(np.stack(lumas).std(axis=0), (61, 61)) > 6.0
    mask = ((persistent > 18.0) & moving).astype(np.uint8)

    zw, zh = int(w * ZONE_W), int(h * ZONE_H)
    best = None
    for zx, zy in ((0, 0), (w - zw, 0), (0, h - zh), (w - zw, h - zh)):
        zone = mask[zy:zy + zh, zx:zx + zw]
        # Join the letters of one mark into one blob.
        joined = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        for k in range(1, n):
            bx, by, bw, bh, _ = stats[k]
            pixels = int(zone[by:by + bh, bx:bx + bw].sum())
            if pixels < 60 or bw > w * MAX_W or bh > h * MAX_H or bw < 24 or bh < 8:
                continue
            if best is None or pixels > best[0]:
                best = (pixels, zx + bx, zy + by, bw, bh)
    if best is None:
        return None
    _, x, y, bw, bh = best
    # Undo the dilation, then pad for the soft edge and shadow of the glyphs.
    x, y, bw, bh = x + 12, y + 4, bw - 24, bh - 8
    pad = max(6, int(round(h * 0.008)))
    x0, y0 = max(1, x - pad), max(1, y - pad)
    x1, y1 = min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)
    return x0, y0, x1 - x0, y1 - y0


def build_mask(video_path: str, mask_path: str, box=None, samples: int = 160):
    """Write a source-sized PNG, white on the watermark's own pixels, for
    ``removelogo``. Returns the box, or None when there is no mark to mask.

    Each pixel of a translucent mark is ``background * (1 - a) + a * colour``:
    across frames it follows its surroundings with a slope below 1 and a
    positive offset. Pixels that fit that line are the mark; HUD text that is
    only sometimes there does not."""
    import cv2
    import numpy as np

    box = box or detect(video_path)
    if not box:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        margin = 14
        x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
        x1, y1 = min(fw, box[0] + box[2] + margin), min(fh, box[1] + box[3] + margin)
        crops = []
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
            ok, frame = cap.read()
            if ok and frame is not None:
                crops.append(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        cap.release()
    if len(crops) < 20:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    stack = np.stack(crops)
    around = np.stack([cv2.GaussianBlur(cv2.morphologyEx(c, cv2.MORPH_OPEN, kernel), (0, 0), 4)
                       for c in crops])
    am, pm = around.mean(0), stack.mean(0)
    slope = ((around - am) * (stack - pm)).mean(0) / (((around - am) ** 2).mean(0) + 1e-6)
    offset = pm - slope * am
    alpha = 1.0 - slope
    glyphs = ((alpha > 0.05) & (alpha < 0.6) & (offset > 4)).astype(np.uint8)
    # Only inside the box the persistent edges gave: the margin was there for
    # the background estimate, and HUD text next to the mark stays untouched.
    inside = np.zeros_like(glyphs)
    inside[box[1] - y0:box[1] + box[3] - y0, box[0] - x0:box[0] + box[2] - x0] = 1
    glyphs *= inside
    # The mark is the largest group of such pixels; stray ones are noise.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(glyphs, np.ones((5, 15), np.uint8)), connectivity=8)
    if n < 2:
        return None
    glyphs = glyphs * (labels == 1 + int(np.argmax(stats[1:, 4])))
    if int(glyphs.sum()) < 60:
        return None
    glyphs = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    full = np.zeros((fh, fw), np.uint8)
    full[y0:y1, x0:x1] = glyphs * 255
    if not cv2.imwrite(mask_path, full):
        return None
    ys, xs = np.nonzero(full)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        found = build_mask(sys.argv[1], sys.argv[2])
    else:
        found = detect(sys.argv[1]) if len(sys.argv) > 1 else None
    if found:
        print(" ".join(str(v) for v in found))
