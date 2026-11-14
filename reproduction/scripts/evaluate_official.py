"""Evaluate FocusTrack raw results with the OFFICIAL Anti-UAV410 toolkit, unmodified.

It only calls ExperimentAntiUAV410.report() from https://github.com/HwangBo94/Anti-UAV410
(experiments/anti_uav.py), which computes:
  - State Accuracy (SA): mean over sequences of the per-frame score
    (IoU when the drone is visible; 1/0 for reporting / not reporting "no target" when it is absent)
  - Success AUC: mean of the success curve over 21 IoU thresholds in [0, 1]
  - Precision: share of frames with center error <= 20 px
All three are averaged per sequence first, then over the sequences.

usage (run from inside the Anti-UAV410 repo, the toolkit uses relative paths for annos/):
  python evaluate_official.py DATASET_ROOT RESULTS_DIR REPORT_DIR
"""
import json, os, sys, glob
import numpy as np

sys.path.insert(0, os.getcwd())
from experiments import ExperimentAntiUAV410  # official code

dataset_root, results_dir, report_dir = sys.argv[1:4]

exp = ExperimentAntiUAV410(root_dir=dataset_root, subset='test', report_dir=report_dir)
n_seq = len(exp.dataset)
missing = [s for s in exp.dataset.seq_names if not os.path.isfile(os.path.join(results_dir, s + '.txt'))]
assert not missing, f'missing results for {len(missing)} sequences: {missing[:5]}'

perf = exp.report([{'name': 'FocusTrack', 'path': results_dir, 'mode': 1}])['FocusTrack']

# SA is printed/written by report() but not stored in performance.json; read it back from its file
sa_line = open(os.path.join(report_dir, 'AntiUAV410', 'test', 'state_accuracy_scores.txt')).read().splitlines()[-1]
sa = float(sa_line.split(':')[-1])

# speed: FocusTrack writes <seq>_time.txt next to the results (seconds per frame)
times = np.concatenate([np.loadtxt(f) for f in glob.glob(os.path.join(results_dir, '*_time.txt'))])
fps_total = len(times) / times[1:].sum() if len(times) > 1 else float('nan')

summary = {
    'sequences': n_seq,
    'frames': int(sum(len(json.load(open(f))['exist']) for f in exp.dataset.anno_files)),
    'AUC': round(100 * perf['overall']['success_score'], 2),
    'P': round(100 * perf['overall']['precision_score'], 2),
    'SA': round(100 * sa, 2),
    'FPS_overall': round(float(fps_total), 1),
    'per_attribute_AUC': {k: round(100 * perf[k]['att_success_score'], 2)
                          for k in ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC', 'TS', 'SS', 'MS', 'NS']},
    'per_attribute_P': {k: round(100 * perf[k]['att_precision_score'], 2)
                        for k in ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC', 'TS', 'SS', 'MS', 'NS']},
}
with open(os.path.join(report_dir, 'final_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
