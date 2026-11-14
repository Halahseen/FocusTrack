"""Draw FocusTrack results vs. ground truth into MP4 videos, and compute Anti-UAV410 metrics.

  red box   = FocusTrack prediction
  green box = ground truth (IR_label.json)

Metrics follow the official Anti-UAV410 toolkit (github.com/HwangBo94/Anti-UAV410):
  * Precision (P@20): fraction of frames whose center-location error <= 20 px
  * Success (AUC):    mean over 21 IoU thresholds in [0,1] of the fraction of frames with IoU > threshold
  * State Accuracy:   mean over frames of  IoU            if the target is visible (exist=1)
                                           1/0 (said "no target"?) if it is not (exist=0)

usage (from the repository root):
  python tools/make_videos_and_metrics.py [--out_dir results]
"""
import argparse
import glob
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')   # repository root
DATA_DIR = ROOT + '/data/antiuav410/test'
RESULT_DIR = ROOT + '/output/test/tracking_results/focustrack/focustrack_single_030/antiuav410'
VIDEO_DIR = ROOT + '/results'   # metrics go here, videos go to <VIDEO_DIR>/videos
SCALE = 2          # targets are ~8x8 px, so upscale frames to make boxes visible
FPS = 25

RED, GREEN, WHITE = (0, 0, 255), (0, 255, 0), (255, 255, 255)   # OpenCV uses BGR


# ---------------------------------------------------------------- metrics
def iou_xywh(a, b):
    """IoU between rows of a and b, boxes as (x, y, w, h)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def center_error(a, b):
    ca = a[:, :2] + (a[:, 2:] - 1) / 2       # same center convention as the official toolkit
    cb = b[:, :2] + (b[:, 2:] - 1) / 2
    return np.sqrt(((ca - cb) ** 2).sum(1))


def seq_metrics(pred, gt, exist):
    visible = exist.astype(bool) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    ious = iou_xywh(pred, gt)
    # Success / Precision: computed on frames where the target is visible
    thr_iou = np.linspace(0, 1, 21)
    success_curve = np.array([(ious[visible] > t).mean() for t in thr_iou])
    ce = center_error(pred, gt)[visible]
    precision_curve = np.array([(ce <= t).mean() for t in range(51)])
    # State accuracy: every frame counts. FocusTrack always outputs a box, so on frames where
    # the target is absent it scores 0 (the toolkit gives 1 only for an empty prediction).
    pred_says_absent = (pred[:, 2] <= 0) | (pred[:, 3] <= 0)
    sa_per_frame = np.where(visible, ious, np.where(exist.astype(bool), np.nan, pred_says_absent.astype(float)))
    return dict(
        frames=len(gt),
        visible_frames=int(visible.sum()),
        precision_20px=float(precision_curve[20]),
        success_auc=float(success_curve.mean()),
        success_rate_iou50=float(success_curve[10]),
        state_accuracy=float(np.nanmean(sa_per_frame)),
        mean_iou=float(ious[visible].mean()),
        _success_curve=success_curve, _precision_curve=precision_curve,
    )


# ---------------------------------------------------------------- video
def draw_box(img, box, color, thick):
    x, y, w, h = [v * SCALE for v in box]
    cv2.rectangle(img, (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h))), color, thick)


def put_text(img, text, org, color, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)  # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def make_video(seq, frames, pred, gt, exist, ious):
    out_path = os.path.join(VIDEO_DIR, 'videos', seq + '.mp4')
    writer = imageio.get_writer(out_path, fps=FPS, codec='libx264', quality=8, macro_block_size=16,
                                ffmpeg_log_level='error')
    for i, f in enumerate(frames):
        img = cv2.imread(f)
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        if exist[i]:
            draw_box(img, gt[i], GREEN, 2)
        draw_box(img, pred[i], RED, 2)
        # dark panel behind the labels (the raw frames have burned-in camera text in that corner)
        panel = img[0:122, 0:250]
        img[0:122, 0:250] = (panel * 0.25).astype(np.uint8)
        put_text(img, f'Frame {i + 1}/{len(frames)}', (12, 30), WHITE)
        put_text(img, 'FocusTrack', (12, 60), RED, 0.6)
        put_text(img, 'Ground truth', (12, 85), GREEN, 0.6)
        put_text(img, f'IoU {ious[i]:.2f}' if exist[i] else 'target absent', (12, 110), WHITE, 0.6)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    writer.close()
    return out_path


def main():
    global VIDEO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=VIDEO_DIR)
    VIDEO_DIR = ap.parse_args().out_dir
    os.makedirs(os.path.join(VIDEO_DIR, 'videos'), exist_ok=True)
    rows = []
    for label_file in sorted(glob.glob(DATA_DIR + '/*/IR_label.json')):
        seq_dir = os.path.dirname(label_file)
        seq = os.path.basename(seq_dir)
        pred_file = os.path.join(RESULT_DIR, seq + '.txt')
        if not os.path.exists(pred_file):
            continue
        label = json.load(open(label_file))
        gt = np.array([r if len(r) == 4 else [0, 0, 0, 0] for r in label['gt_rect']], dtype=float)
        exist = np.array(label['exist'], dtype=int)
        pred = np.loadtxt(pred_file, delimiter='\t').reshape(-1, 4)
        frames = sorted(glob.glob(seq_dir + '/*.jpg'), key=lambda p: int(os.path.basename(p)[:-4]))
        assert len(frames) == len(gt) == len(pred), (seq, len(frames), len(gt), len(pred))

        m = seq_metrics(pred, gt, exist)
        rows.append((seq, m))
        video = make_video(seq, frames, pred, gt, exist, iou_xywh(pred, gt))
        print(f'video: {video}')

    # ---- report (per sequence, plus mean over sequences like the toolkit's "overall")
    keys = ['precision_20px', 'success_auc', 'state_accuracy', 'success_rate_iou50', 'mean_iou']
    lines = [f"{'sequence':28s} {'frames':>6s} " + ' '.join(f'{k:>18s}' for k in keys)]
    for seq, m in rows:
        lines.append(f"{seq:28s} {m['frames']:6d} " + ' '.join(f'{100 * m[k]:17.2f}%' for k in keys))
    lines.append(f"{'MEAN over sequences':28s} {sum(m['frames'] for _, m in rows):6d} " +
                 ' '.join(f'{100 * np.mean([m[k] for _, m in rows]):17.2f}%' for k in keys))
    report = '\n'.join(lines)
    print('\n' + report)
    with open(os.path.join(VIDEO_DIR, 'metrics.txt'), 'w') as f:
        f.write(report + '\n')
    with open(os.path.join(VIDEO_DIR, 'metrics.json'), 'w') as f:
        json.dump({seq: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
                   for seq, m in rows}, f, indent=1)


if __name__ == '__main__':
    main()
