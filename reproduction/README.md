# Reproducing FocusTrack on the full Anti-UAV410 test set

**Setup:** `focustrack_single` with the official weights (`FocusTrack_ep0030.pth.tar`), all **120** test sequences
(**129,691** frames), Windows 11, NVIDIA RTX A4000, Python 3.9, torch 2.0.0+cu118, run on 2026-11-14.
Total run time was 1 h 10 min (~41 FPS for the model itself, ~38 FPS counting the first, initialization frame of each sequence).

**Evaluation:** the official Anti-UAV410 toolkit ([HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410),
commit `8a8eb04`, `experiments/anti_uav.py`), **unmodified**, called by `scripts/evaluate_official.py`.

## Results

| | AUC (Success) | P (20 px) | SA (State Acc.) |
|---|---|---|---|
| Paper (Table) | 62.8 | 86.2 | 63.9 |
| Authors' published raw results, official eval | 62.64 | 86.37 | 63.89 |
| **Our run, official eval** | **61.98** | **86.13** | **63.17** |
| Our run − paper | −0.82 | −0.07 | −0.73 |

## Why the numbers differ

1. **The evaluation is correct.** Applying the same evaluation code to the authors' own published boxes
   gives SA 63.89, which matches the paper's 63.9. AUC and P differ from the paper by only 0.16–0.17 points, probably
   because of rounding or a slightly different script or run behind the paper's table. So most of the gap
   comes from the tracker output, not from the metric.
2. **Our boxes diverge from the authors' boxes from the very first frames.** None of the 120 sequences is identical.
   In every sequence the first difference appears at frame 2–4, as a 1-pixel change (for example `206 312` vs `205 312`).
   Such tiny differences come from floating-point arithmetic. They depend on the GPU model
   (RTX A4000 here, using TF32 convolutions on Ampere), the cuDNN/cuBLAS kernels, and library versions (OpenCV, the
   `cvxopt` QP solver used by the non-uniform sampling grid).
3. **A tracker amplifies tiny differences.** Each frame's search crop depends on the previous box, so a 1-px
   change can decide whether the tracker locks onto a distractor or stays on the drone. For 15% of the frames, our box
   and the authors' box are more than 20 px apart. But these errors go **both ways**:
   our run is better on 38 sequences, the authors' on 55, and 27 are within ±0.5 points.
   Per-sequence differences reach ±40 SA points, yet they almost cancel out over the set. The 5 worst sequences
   alone account for −0.90 points of SA, which is more than the whole −0.72 gap.
4. **This is not a bug in the Windows port.**
   - The checkpoint loads with `strict=True`, so the network is identical.
   - Our run is deterministic: the result file for `3700000000002_153139_2` from Nov 7 is byte-identical to today's.
   - A systematic bug (for example, wrong preprocessing) would make us worse almost everywhere, not better on 38 sequences.

**Conclusion:** FocusTrack reproduces within about 1 point on every metric (−0.8 AUC, −0.1 P, −0.7 SA). That is the
normal run-to-run spread for a single-object tracker on a different machine.

## Files

| Path | What |
|---|---|
| `raw_results/` | our tracker output: `<seq>.txt` (x y w h per frame) and `<seq>_time.txt` |
| `official_report/final_metrics.json` | AUC / P / SA / FPS + per-attribute scores (our run) |
| `official_report/AntiUAV410/test/` | official toolkit output: `performance.json`, `state_accuracy_scores.txt`, success/precision plots (PDF, overall and per attribute) |
| `official_report_authors_results/` | the same evaluation applied to the authors' published raw results |
| (not included) | the authors' raw results: [Google Drive](https://drive.google.com/drive/folders/1xRXHlMYzs_02WYTcLrOMRtZwryvp9KF0) |
| `compare_with_authors.json` | per-sequence comparison, ours vs. authors |
| `tracker_run.log` | console log of the tracking run |
| `scripts/` | `fetch_test_split.py` (download the test split only), `evaluate_official.py`, `compare_with_authors.py` |

## How to reproduce

```powershell
# tracker (from the FocusTrack fork, local.py pointing data_dir to the folder that holds antiuav410/test)
python tracking/test.py --tracker_name focustrack --tracker_param focustrack_single --dataset_name antiuav410_test --threads 0 --num_gpus 1 --run_id 30
# evaluation (cwd = Anti-UAV410 repo; needs wget and shapely installed)
python evaluate_official.py <data>/antiuav410 <raw_results_dir> <report_dir>
```
