"""Render the 'hook' video: FocusTrack vs OSTrack vs SiamDT on Anti-UAV410 test sequence
3700000000002_153139_2. No method explanation on screen, only the result.

FocusTrack boxes: our own run (focustrack_single, ep30).
OSTrack / SiamDT boxes: raw results published by the Anti-UAV410 authors.
'LOST' = IoU with the ground-truth box is 0 in that frame (ground truth itself is not drawn).
"""
import os
import numpy as np, cv2, imageio
from PIL import Image, ImageDraw, ImageFont
import json

SEQ = "3700000000002_153139_2"
IMG = rf"D:\data\antiuav410\test\{SEQ}"
GT_FILE = rf"F:\seminar\code\FocusTrack\annos\test\{SEQ}.txt"
RES = r"D:\data\au410_results\x\tracking_results"
TRK = [  # name, file, color (RGB)
    ("FocusTrack", rf"F:\seminar\code\FocusTrack\output\test\tracking_results\focustrack\focustrack_single_030\antiuav410\{SEQ}.txt", (40, 230, 70)),
    ("OSTrack", rf"{RES}\Defaults\OSTrack\vitb_384_mae_ce_32x4_ep300\{SEQ}.txt", (255, 150, 0)),
    ("SiamDT", rf"{RES}\Trained_with_antiuav410\SiamDT\{SEQ}.txt", (240, 60, 240)),
]
OUT = r"F:\seminar\hook\hook_FocusTrack.mp4"
FPS = 25
W, H = 1920, 1080
FONT = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_R = r"C:\Windows\Fonts\segoeui.ttf"
F0, F_END = 430, 686          # 1-based frame range shown
INSET_SRC = 56                # source crop size (px) for the zoomed inset


def load_boxes(path):
    txt = open(path).read().strip()
    if txt.startswith("{"):
        return np.array([b if len(b) == 4 else [0, 0, 0, 0] for b in json.loads(txt)["res"]], float)
    rows = [l.replace("\t", ",").split(",") for l in txt.splitlines()]
    return np.array([[float(v) for v in r][:4] for r in rows], float)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


gt = np.loadtxt(GT_FILE, delimiter=",").reshape(-1, 4)
boxes = {n: load_boxes(p)[: len(gt)] for n, p, _ in TRK}
lost = {n: np.array([iou(boxes[n][i], gt[i]) == 0 for i in range(len(gt))]) for n in boxes}

# moment SiamDT drops the drone for good (last frame on target before F_END, +1)
on = np.where(~lost["SiamDT"][: F_END])[0]
FAIL = int(on[on < F_END - 1].max()) + 2       # 1-based first frame of the final loss
print("SiamDT final loss at frame", FAIL)

# smoothed inset centre that follows the drone (ground-truth centre)
cx = gt[:, 0] + gt[:, 2] / 2; cy = gt[:, 1] + gt[:, 3] / 2
k = np.ones(9) / 9
cx_s = np.convolve(np.pad(cx, 4, mode="edge"), k, "valid"); cy_s = np.convolve(np.pad(cy, 4, mode="edge"), k, "valid")


def font(sz, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_R, sz)


def text_card(lines, sub=None):
    im = Image.new("RGB", (W, H), (12, 14, 18)); d = ImageDraw.Draw(im)
    y = H // 2 - 70 * len(lines) // 2 - (40 if sub else 0)
    for t, sz, col in lines:
        f = font(sz); w = d.textlength(t, font=f)
        d.text(((W - w) / 2, y), t, font=f, fill=col); y += int(sz * 1.35)
    if sub:
        f = font(30, False); w = d.textlength(sub, font=f)
        d.text(((W - w) / 2, y + 30), sub, font=f, fill=(150, 155, 165))
    return np.array(im)


MAIN_H = H
MAIN_W = int(640 * MAIN_H / 512)        # 1350
S = MAIN_H / 512                          # 2.109 scale of main view
PANEL_X = MAIN_W + 30
INSET = W - PANEL_X - 30                  # 510 px square inset


def render(fi, show_boxes=True, freeze_note=None, speed_note=None, teaser=False, reveal=1.0):
    """fi: 0-based frame index."""
    raw = cv2.cvtColor(cv2.imread(os.path.join(IMG, f"{fi+1:06d}.jpg")), cv2.COLOR_BGR2RGB)
    main = cv2.resize(raw, (MAIN_W, MAIN_H), interpolation=cv2.INTER_CUBIC)
    canvas = Image.new("RGB", (W, H), (12, 14, 18)); canvas.paste(Image.fromarray(main), (0, 0))
    d = ImageDraw.Draw(canvas)

    # inset crop around the drone
    c0x = int(np.clip(cx_s[fi] - INSET_SRC / 2, 0, 640 - INSET_SRC)); c0y = int(np.clip(cy_s[fi] - INSET_SRC / 2, 0, 512 - INSET_SRC))
    z = INSET / INSET_SRC
    crop = cv2.resize(raw[c0y:c0y + INSET_SRC, c0x:c0x + INSET_SRC], (INSET, INSET), interpolation=cv2.INTER_CUBIC)
    iy = 150
    if not teaser or reveal > 0:
        canvas.paste(Image.fromarray(crop), (PANEL_X, iy))
        # inset region on main frame + connector
        rx0, ry0, rx1, ry1 = c0x * S, c0y * S, (c0x + INSET_SRC) * S, (c0y + INSET_SRC) * S
        d.rectangle([rx0, ry0, rx1, ry1], outline=(255, 255, 255), width=2)
        d.line([rx1, ry0, PANEL_X, iy], fill=(255, 255, 255, 120), width=1)
        d.line([rx1, ry1, PANEL_X, iy + INSET], fill=(255, 255, 255), width=1)
        d.rectangle([PANEL_X, iy, PANEL_X + INSET, iy + INSET], outline=(255, 255, 255), width=3)
        d.text((PANEL_X, iy - 50), f"ZOOM ×{z:.0f}", font=font(32), fill=(255, 255, 255))

    if show_boxes:
        for n, _, col in reversed(TRK):
            x, y, w, h = boxes[n][fi]
            if w <= 0: continue
            lw = 4 if n == "FocusTrack" else 3
            pad = 3
            d.rectangle([x * S - pad, y * S - pad, (x + w) * S + pad, (y + h) * S + pad], outline=col, width=lw)
            f = font(26); tw = d.textlength(n, font=f)
            ty = y * S - pad - 34 if n != "SiamDT" else (y + h) * S + pad + 2
            d.rectangle([x * S - pad, ty, x * S - pad + tw + 10, ty + 32], fill=col)
            d.text((x * S - pad + 5, ty - 1), n, font=f, fill=(0, 0, 0))
            # boxes in the inset
            bx0, by0 = (x - c0x) * z + PANEL_X, (y - c0y) * z + iy
            bx1, by1 = bx0 + w * z, by0 + h * z
            if bx1 > PANEL_X and by1 > iy and bx0 < PANEL_X + INSET and by0 < iy + INSET:
                d.rectangle([max(bx0, PANEL_X), max(by0, iy), min(bx1, PANEL_X + INSET), min(by1, iy + INSET)], outline=col, width=5)

        # legend / status
        ly = iy + INSET + 40
        for n, _, col in TRK:
            d.rectangle([PANEL_X, ly + 8, PANEL_X + 34, ly + 42], fill=col)
            d.text((PANEL_X + 50, ly), n, font=font(40), fill=(255, 255, 255))
            st, sc = ("LOST", (255, 70, 70)) if lost[n][fi] else ("on target", (120, 230, 140))
            f = font(36); d.text((PANEL_X + INSET - d.textlength(st, font=f), ly + 3), st, font=f, fill=sc)
            ly += 70

    d.text((PANEL_X, 40), f"frame {fi+1}", font=font(30, False), fill=(170, 175, 185))
    d.text((PANEL_X, H - 60), "Anti-UAV410 test set · infrared", font=font(24, False), fill=(130, 135, 145))
    if speed_note:
        f = font(40); d.rectangle([20, 20, 40 + d.textlength(speed_note, font=f), 80], fill=(0, 0, 0))
        d.text((30, 24), speed_note, font=f, fill=(255, 220, 60))
    if freeze_note:
        f = font(64); tw = d.textlength(freeze_note, font=f)
        d.rectangle([(MAIN_W - tw) / 2 - 30, H - 170, (MAIN_W + tw) / 2 + 30, H - 70], fill=(0, 0, 0))
        d.text(((MAIN_W - tw) / 2, H - 165), freeze_note, font=f, fill=(255, 80, 80))
    if teaser:
        f = font(70); t = "Can you find the drone?"; tw = d.textlength(t, font=f)
        d.rectangle([(MAIN_W - tw) / 2 - 30, 30, (MAIN_W + tw) / 2 + 30, 140], fill=(0, 0, 0))
        d.text(((MAIN_W - tw) / 2, 38), t, font=f, fill=(255, 255, 255))
    return np.array(canvas)


wr = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8,
                        ffmpeg_params=["-movflags", "+faststart"])


def hold(img, sec):
    for _ in range(int(sec * FPS)): wr.append_data(img)


def fade(a, b, sec=0.4):
    n = int(sec * FPS)
    for i in range(n): wr.append_data((a * (1 - i / n) + b * (i / n)).astype(np.uint8))


# 1) title
title = text_card([("Three trackers.", 96, (255, 255, 255)), ("One 8-pixel drone.", 96, (255, 255, 255))],
                  sub="A real infrared test video from the Anti-UAV410 benchmark")
hold(title, 3.0)
# 2) teaser: frame without boxes and without inset
teas = render(F0 - 1, show_boxes=False, teaser=True, reveal=0)
fade(title, teas); hold(teas, 2.2)
# 3) reveal: inset + boxes on the same frame
first = render(F0 - 1)
fade(teas, first, 0.5); hold(first, 1.2)
# 4) play at real speed until just before SiamDT's final loss
for fi in range(F0 - 1, FAIL - 12):
    wr.append_data(render(fi, speed_note="1×"))
# 5) slow motion (4x slower) through the failure
for fi in range(FAIL - 12, FAIL + 6):
    img = render(fi, speed_note="SLOW MOTION ¼×")
    for _ in range(4): wr.append_data(img)
# 6) freeze frame
fz = render(FAIL + 5, freeze_note="OSTrack: lost.   SiamDT: lost.")
hold(fz, 2.2)
# 7) continue: FocusTrack keeps it
for fi in range(FAIL + 5, F_END):
    wr.append_data(render(fi, speed_note="1×"))
last = render(F_END - 1)
# 8) end card
end = text_card([("Same video. Same drone.", 84, (255, 255, 255)),
                 ("Two trackers lost it.", 84, (255, 140, 140)),
                 ("FocusTrack didn't.", 84, (120, 240, 140)),
                 ("How?", 110, (255, 220, 60))])
fade(last, end, 0.6); hold(end, 3.0)
wr.close()
print("wrote", OUT)
