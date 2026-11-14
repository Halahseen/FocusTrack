# Hook video: FocusTrack vs OSTrack vs SiamDT

[`hook_FocusTrack.mp4`](hook_FocusTrack.mp4) (25 s, 1920×1080) is the opening clip for the seminar talk.

- **Sequence:** Anti-UAV410 test, `3700000000002_153139_2`, frames 430–686. The drone is about 8×8 px.
- **FocusTrack:** our own run (`focustrack_single`, epoch 30). Raw boxes: [`focustrack_3700000000002_153139_2.txt`](focustrack_3700000000002_153139_2.txt).
- **OSTrack** (ViT-B 384, default weights) and **SiamDT** (trained on Anti-UAV410): raw results published by the [Anti-UAV410 authors](https://github.com/HwangBo94/Anti-UAV410) (the "tracking results" download).
- **"LOST" / "on target":** whether the tracker's box overlaps the ground-truth box in that frame (IoU > 0). Ground truth itself is not drawn. The zoom inset follows the ground-truth position.

## What happens in the whole sequence (not only the clip)
- OSTrack loses the drone at frame ~273 and never recovers.
- SiamDT switches back and forth between the drone and a bright house until frame 523, then stays lost.
- FocusTrack **also** loses the drone at frames ~276–340, then recovers and tracks it until frame ~686. After an abrupt camera move there, it loses it as well.
- Lost frames (visible drone, IoU = 0) over the whole sequence: OSTrack 81%, SiamDT 73%, FocusTrack 57%.

This is a hand-picked sequence chosen to show the difference. On the 7 hardest test sequences we ran, FocusTrack is not always better.

## How the sequence was chosen (`tools/hook/`)
1. `rank_failures.py`: ranks all 120 test sequences by how often OSTrack and SiamDT lose the drone, using the ground truth in `annos/test`.
2. `fetch_sequences.py`: downloads only the selected sequences from the 9.4 GB dataset zip, using HTTP range requests.
3. FocusTrack was run on those 7 sequences with `tracking/test.py`.
4. `compare3.py` and `look.py`: per-frame comparison of the three trackers and contact sheets. The check "FocusTrack overlaps the drone" alone was misleading on `new21_train_newfix`, where FocusTrack's box covered almost the whole image, so those frames were inspected visually.
5. `make_hook.py`: renders the video.

Paths in the scripts are absolute paths of the machine they were run on.
