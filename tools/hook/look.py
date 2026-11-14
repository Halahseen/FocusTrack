"""Quick look: box sizes + IoU along time for one sequence, and a contact sheet of frames."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from compare3 import TR, load_boxes, iou, GT_DIR  # noqa

seq = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]
gt = np.loadtxt(os.path.join(GT_DIR, seq + ".txt"), delimiter=",").reshape(-1, 4)
B = {k: load_boxes(os.path.join(d, seq + ".txt"))[: len(gt)] for k, d in TR.items()}
for s in range(0, len(gt), 100):
    print(s + 1, "GT", gt[s].astype(int), " ".join(f"{k}:{B[k][s].astype(int)} iou={iou(B[k][s:s+1], gt[s:s+1])[0]:.2f}" for k in B))
col = {"OSTrack": (255, 160, 0), "SiamDT": (255, 0, 255), "FocusTrack": (0, 0, 255)}
tiles = []
for fr in frames:
    im = cv2.imread(rf"D:\data\antiuav410\test\{seq}\{fr:06d}.jpg")
    x, y, w, h = gt[fr - 1].astype(int); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for k, c in col.items():
        x, y, w, h = B[k][fr - 1].astype(int); cv2.rectangle(im, (x, y), (x + w, y + h), c, 1)
    cv2.putText(im, str(fr), (10, 30), 0, 1, (255, 255, 255), 2)
    tiles.append(im)
cv2.imwrite(sys.argv[3], np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
