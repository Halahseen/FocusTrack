"""Per-sequence comparison FocusTrack (our run) vs OSTrack vs SiamDT (published raw results).
Prints overall lost%% and, per 50-frame window, which trackers are on the drone, to find
a moment where the others fail and FocusTrack does not."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rank_failures import load_boxes, iou, GT_DIR, TRACKERS  # noqa (also prints ranking; ignore)

FT = r"F:\seminar\code\FocusTrack\output\test\tracking_results\focustrack\focustrack_single_030\antiuav410"
TR = dict(TRACKERS, FocusTrack=FT)

for f in sorted(os.listdir(FT)):
    if f.endswith("_time.txt"): continue
    seq = f[:-4]
    gt = np.loadtxt(os.path.join(GT_DIR, seq + ".txt"), delimiter=",").reshape(-1, 4)
    vis = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    ious = {}
    for k, d in TR.items():
        pr = load_boxes(os.path.join(d, seq + ".txt"))[: len(gt)]
        ious[k] = iou(pr, gt)
    print(f"\n== {seq}  " + "  ".join(f"{k}: lost {100*np.mean(ious[k][vis]==0):.0f}% SA {100*np.mean(ious[k][vis]):.0f}" for k in TR))
    # windows where FocusTrack is on target >90% and both others are lost >90%
    W = 50
    for s in range(0, len(gt) - W, W):
        v = vis[s:s + W]
        if v.mean() < 0.9: continue
        on = {k: np.mean(ious[k][s:s + W][v] > 0) for k in TR}
        if on["FocusTrack"] > 0.9 and on["OSTrack"] < 0.1 and on["SiamDT"] < 0.1:
            print(f"   frames {s+1}-{s+W}: FT on {on['FocusTrack']:.2f}, OST {on['OSTrack']:.2f}, SiamDT {on['SiamDT']:.2f}")
