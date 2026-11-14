"""Per-sequence comparison: our FocusTrack boxes vs. the authors' published raw results.
Per-sequence score = official Anti-UAV410 state accuracy (same function as experiments/anti_uav.py).

usage: python compare_with_authors.py DATASET_TEST_DIR OURS_DIR AUTHORS_DIR OUT_JSON
"""
import glob, json, os, sys
import numpy as np


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w, h = min(ax1, bx1) - max(a[0], b[0]), min(ay1, by1) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def sa(pred, lab):  # identical logic to the official eval()
    m = []
    for p, g, e in zip(pred, lab['gt_rect'], lab['exist']):
        if not e:
            m.append(0.0)  # FocusTrack always outputs a box, never "no target"
        elif len(g) < 4 or sum(g) == 0:
            continue
        else:
            m.append(iou(p, g))
    return float(np.mean(m))


test_dir, ours_dir, auth_dir, out = sys.argv[1:5]
rows = []
for lf in sorted(glob.glob(os.path.join(test_dir, '*', 'IR_label.json'))):
    seq = os.path.basename(os.path.dirname(lf))
    lab = json.load(open(lf))
    o = np.loadtxt(os.path.join(ours_dir, seq + '.txt'))
    a = np.loadtxt(os.path.join(auth_dir, seq + '.txt'))
    diff = np.any(o != a, axis=1)
    first = int(np.argmax(diff)) + 1 if diff.any() else None
    far = np.linalg.norm((o[:, :2] + o[:, 2:] / 2) - (a[:, :2] + a[:, 2:] / 2), axis=1) > 20
    rows.append(dict(seq=seq, frames=len(o), identical=bool(not diff.any()), first_diff_frame=first,
                     frames_differ=int(diff.sum()), frames_far_apart_20px=int(far.sum()),
                     sa_ours=sa(o, lab), sa_authors=sa(a, lab),
                     absent_frames=int(len(lab['exist']) - sum(lab['exist']))))
for r in rows:
    r['sa_delta'] = r['sa_ours'] - r['sa_authors']

d = np.array([r['sa_delta'] for r in rows])
tot = sum(r['frames'] for r in rows)
summary = dict(
    sequences=len(rows),
    identical_sequences=sum(r['identical'] for r in rows),
    frames_with_any_difference_pct=100 * sum(r['frames_differ'] for r in rows) / tot,
    frames_far_apart_20px_pct=100 * sum(r['frames_far_apart_20px'] for r in rows) / tot,
    seqs_ours_better=int((d > 0.005).sum()), seqs_authors_better=int((d < -0.005).sum()),
    seqs_within_0_5pt=int((abs(d) <= 0.005).sum()),
    mean_sa_delta_pts=100 * float(d.mean()),
    delta_from_top5_seqs_pts=100 * float(np.sort(d)[:5].sum() / len(d)),
    absent_frames_pct=100 * sum(r['absent_frames'] for r in rows) / tot,
    worst=sorted(rows, key=lambda r: r['sa_delta'])[:6],
    best=sorted(rows, key=lambda r: -r['sa_delta'])[:4],
)
json.dump(dict(summary=summary, per_sequence=rows), open(out, 'w'), indent=1)
s = {k: v for k, v in summary.items() if k not in ('worst', 'best')}
print(json.dumps(s, indent=1))
for name in ('worst', 'best'):
    print(name)
    for r in summary[name]:
        print(f"  {r['seq']:28s} SA ours {100*r['sa_ours']:5.1f}  authors {100*r['sa_authors']:5.1f}  "
              f"delta {100*r['sa_delta']:+5.1f}  first diff @ frame {r['first_diff_frame']}  far>20px {r['frames_far_apart_20px']}/{r['frames']}")
