"""Rank Anti-UAV410 test sequences by how badly published trackers lose the drone.

For each sequence: per-frame IoU vs ground truth for OSTrack (default weights) and
SiamDT (trained on Anti-UAV410, the dataset authors' own tracker). We look for long
runs where the drone is visible but the tracker's IoU is 0 (= lost).
"""
import json, os, sys
import numpy as np

GT_DIR = r"F:\seminar\code\FocusTrack\annos\test"
RES = r"D:\data\au410_results\x\tracking_results"
TRACKERS = {
    "OSTrack": os.path.join(RES, r"Defaults\OSTrack\vitb_384_mae_ce_32x4_ep300"),
    "SiamDT": os.path.join(RES, r"Trained_with_antiuav410\SiamDT"),
}


def load_boxes(path):
    txt = open(path).read().strip()
    if txt.startswith("{"):
        return np.array([b if len(b) == 4 else [0, 0, 0, 0] for b in json.loads(txt)["res"]], float)
    rows = [l.replace("\t", ",").replace(" ", ",").split(",") for l in txt.splitlines()]
    return np.array([[float(v) for v in r if v != ""][:4] or [0, 0, 0, 0] for r in rows], float)


def iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0)


def longest_run(mask):
    best = cur = 0; start = bstart = 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0: start = i
            cur += 1
            if cur > best: best, bstart = cur, start
        else:
            cur = 0
    return best, bstart


rows = []
for f in sorted(os.listdir(GT_DIR)):
    if not f.endswith(".txt"): continue
    seq = f[:-4]
    gt = np.loadtxt(os.path.join(GT_DIR, f), delimiter=",").reshape(-1, 4)
    vis = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    row = {"seq": seq, "n": len(gt), "size": float(np.median(np.sqrt(gt[vis, 2] * gt[vis, 3]))) if vis.any() else 0}
    ok = True
    for name, d in TRACKERS.items():
        p = os.path.join(d, seq + ".txt")
        if not os.path.exists(p): ok = False; break
        pr = load_boxes(p)[: len(gt)]
        if len(pr) < len(gt): pr = np.vstack([pr, np.zeros((len(gt) - len(pr), 4))])
        io = iou(pr, gt)
        lost = vis & (io == 0)
        row[name + "_sa"] = float(np.mean(io[vis]))
        row[name + "_lost"] = float(lost[vis].mean())
        row[name + "_run"], row[name + "_runstart"] = longest_run(lost)
    if ok:
        rows.append(row)

rows.sort(key=lambda r: -min(r["OSTrack_lost"], r["SiamDT_lost"]))
print(f"{'seq':28s} {'n':>5s} {'size':>5s} | OST lost% run@start | SiamDT lost% run@start")
for r in rows[:30]:
    print(f"{r['seq']:28s} {r['n']:5d} {r['size']:5.1f} | {100*r['OSTrack_lost']:5.1f} {r['OSTrack_run']:5d}@{r['OSTrack_runstart']:<5d} | "
          f"{100*r['SiamDT_lost']:5.1f} {r['SiamDT_run']:5d}@{r['SiamDT_runstart']:<5d}")
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "rank_failures.json"), "w"), indent=1)
