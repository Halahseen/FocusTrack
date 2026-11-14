# FocusTrack on Anti-UAV410 — Windows run

[עברית](#עברית) | [English](#english)

> **Credit.** All tracking code belongs to the authors of
> *FocusTrack: A Self-Adaptive Local Sampling Algorithm for Efficient Anti-UAV Tracking*
> ([paper](https://arxiv.org/abs/2504.13604), [official code](https://github.com/vero1925/FocusTrack)).
> This repository only adds the changes needed to run it on Windows, plus the results of a
> small seminar experiment (Sapir College, seminar on multi-object tracking in UAV video).
> The original repository has no license file; the code is shared here for study purposes only.
> The original README is kept as [`README_original.md`](README_original.md).

---

<div dir="rtl">

## עברית

### מה זה FocusTrack?
FocusTrack הוא tracker לעקיבה אחרי רחפן יחיד בסרטוני אינפרא-אדום (Anti-UAV). הוא מבוסס על Vision Transformer (ViT-B), ובכל פריים הוא מחפש את המטרה בחלון קטן סביב המיקום הקודם שלה (local tracking).

החידוש במאמר הוא **דגימה מקומית מסתגלת**. במקום חלון חיפוש בגודל קבוע, הוא:
- מגדיל את אזור החיפוש כשהביטחון של המודל יורד (למשל כשהרחפן הולך לאיבוד);
- משתמש ברשת דגימה לא-אחידה, כדי "להתמקד" במטרה הקטנה ולהגדיל אותה בלי לאבד הקשר.

ככה הוא מתמודד עם מטרות זעירות (לפעמים 8×8 פיקסלים בלבד) ועם תנועה מהירה, ועדיין רץ מהר.

### מה הורץ כאן
- הקונפיג `focustrack_single` עם המשקולות הרשמיות (`FocusTrack_ep0030.pth.tar`).
- 3 רצפים מה-test של Anti-UAV410.
- כרטיס מסך NVIDIA RTX A4000 על Windows 11, בקצב של כ-**41 FPS**.

### תוצאות

| רצף | פריימים | Precision (20px) | Success (AUC) | State Accuracy |
|---|---|---|---|---|
| 02_6319_1500-2999 | 1500 | 80.00% | 50.27% | 51.07% |
| 20190925_111757_1_1 | 1000 | 100.00% | 78.85% | 80.42% |
| 3700000000002_130905_1 | 1500 | 99.87% | 72.29% | 73.72% |
| **ממוצע** | 4000 | **93.29%** | **67.13%** | **68.40%** |

**מה מודדים** (הנוסחאות לפי קוד ההערכה הרשמי של [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410)):
- **Precision:** אחוז הפריימים שבהם מרכז הקופסה נמצא עד 20 פיקסלים ממרכז האמת.
- **Success:** השטח מתחת לעקומת ה-IoU, ממוצע על פני 21 ספים בין 0 ל-1.
- **State Accuracy:** ה-IoU הממוצע, כשבפריים שבו הרחפן לא נראה, ה-tracker צריך לדווח "אין מטרה".

3 רצפים מתוך 120 הם לא מדגם מייצג, אז אין להשוות אותם ישירות לטבלה במאמר.

**דוגמת כישלון:** ברצף `02_6319`, בפריימים 1124 עד 1389 בערך, הרחפן עף מול בניין. ה-tracker נתפס לחלון בבניין ואיבד אותו, ובסביבות פריים 1390 חזר אליו.

**סרטונים:** בתיקייה [`results/videos/`](results/videos). אדום = FocusTrack, ירוק = האמת. התמונה הוגדלה פי 2.

הסיכום המלא נמצא ב-[`results/summary.md`](results/summary.md).

### התיקונים ל-Windows ולמה הם עובדים
1. **נתיבים שבורים ב-`local.py`.**
   - `create_default_local_file.py` כתב נתיבי Windows עם `\` לתוך מחרוזת Python, ושם `\a`, ‏`\t` ו-`\n` הם תווים מיוחדים. למשל `\antiuav410` הפך לתו "צלצול" ואחריו `ntiuav410`.
   - **התיקון:** בזמן הכתיבה כל `\` מוחלף ב-`/`, ש-Windows מקבל גם כן. השינוי נמצא ב-`lib/test/evaluation/environment.py` וב-`lib/train/admin/environment.py`.
   - בנוסף, `realpath` הוחלף ב-`abspath`, כדי שכונן רשת ממופה (למשל `F:`) לא יהפוך לנתיב רשת `\\server\...`.
2. **`mmseg` לא מותקן.**
   - ה-decoder משתמש בו רק כמחלקת בסיס (`BaseDecodeHead`) שמגדירה כמה שדות. ההתקנה של mmseg ב-Windows קשה מאוד, כי היא דורשת את `mmcv` המקומפל.
   - **התיקון:** ב-`lib/models/layers/decoder.py` נוספה מחלקה חלופית מינימלית, שמשמשת רק אם mmseg לא מותקן.
   - **ההוכחה שהיא נכונה:** ה-tracker טוען את המשקולות עם `strict=True`, כלומר כל פרמטר חסר או מיותר היה עוצר את ההרצה עם שגיאה, והטעינה עוברת.
3. **חבילות שחסרות ב-README.** הקוד מייבא אותן גם בזמן test, אז צריך להתקין אותן:
   - `jpeg4py`: מספיק ה-wrapper. אם ספריית ה-C שלה חסרה, הקוד עובר ל-OpenCV.
   - `visdom`: צריכה `setuptools<81` ו-`--no-build-isolation`.
   - `cvxopt`: משמשת את רשת הדגימה הלא-אחידה.
4. **גרסאות.**
   - נשמרו הגרסאות של ה-README: Python 3.9 ו-PyTorch 2.0.0 עם CUDA 11.8.
   - דרייבר חדש יותר (כאן CUDA 13.2) מריץ אותן, כי דרייברים של NVIDIA תואמים לאחור. גם לא צריך להתקין CUDA Toolkit, כי ה-wheel של PyTorch כולל אותו.
   - `timm` ננעל על 0.6.13, כי הקוד משתמש ב-API ישן שלה.
5. **`--threads 0`.** ריבוי תהליכים (multiprocessing) עם CUDA נוטה להיתקע ב-Windows, אז ההרצה נעשית בתהליך יחיד.
6. **דגל ההתקנה של uv.** קובץ הדרישות מוסיף את האינדקס של PyTorch, ו-uv לוקח כל חבילה רק מהאינדקס הראשון שבו היא מופיעה. באינדקס של PyTorch יש גרסאות ישנות של `packaging` ו-`numpy`, ולכן ההתקנה נכשלת. **התיקון:** `--index-strategy unsafe-best-match`.
7. **התיקייה `lib/train/data`.** בריפו הקודם שלי (שאינו fork) היא חסרה, ובלעדיה `import lib.train.data` נכשל. ה-fork הזה מבוסס על הקוד המקורי, אז כל 10 הקבצים שלה נמצאים בו.

### איך מריצים
ההוראות המלאות (באנגלית) נמצאות [למטה](#how-to-run).

</div>

---

## English

### What is FocusTrack?
FocusTrack is a single-object tracker for infrared anti-UAV video, built on a ViT-B backbone. It is a *local* tracker: each frame it looks for the drone in a small window around its last position.

The paper's contribution is **self-adaptive local sampling**:
- the search region is enlarged when the model's confidence drops (for example, when the drone is lost);
- a non-uniform sampling grid "zooms in" on the tiny target without losing context.

This helps with very small targets (sometimes ~8×8 px) and fast motion, while staying fast.

### What was run here
- Config `focustrack_single` with the official weights (`FocusTrack_ep0030.pth.tar`).
- 3 sequences from the Anti-UAV410 **test** split.
- NVIDIA RTX A4000, Windows 11, about **41 FPS**.

### Results

| Sequence | Frames | Precision (20px) | Success (AUC) | State Accuracy |
|---|---|---|---|---|
| 02_6319_1500-2999 | 1500 | 80.00% | 50.27% | 51.07% |
| 20190925_111757_1_1 | 1000 | 100.00% | 78.85% | 80.42% |
| 3700000000002_130905_1 | 1500 | 99.87% | 72.29% | 73.72% |
| **Mean** | 4000 | **93.29%** | **67.13%** | **68.40%** |

**What the metrics mean** (formulas follow the official [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) evaluation code):
- **Precision:** share of frames whose center error is ≤ 20 px.
- **Success:** area under the IoU success curve, averaged over 21 thresholds from 0 to 1.
- **State Accuracy:** mean IoU, where on frames without a visible target the tracker must report "no target".

Three sequences out of 120 are not representative, so do not compare these numbers directly with the paper's table.

**Failure example:** in `02_6319`, frames ~1124–1389, the drone passes in front of a building. The tracker locks onto a window of the building, then recovers around frame 1390.

**Videos:** in [`results/videos/`](results/videos). Red = FocusTrack, green = ground truth, frames upscaled 2×.

### Windows fixes and why they work
1. **Broken paths in the generated `local.py`.**
   - `create_default_local_file.py` wrote Windows paths with `\` into plain Python string literals, where `\a`, `\t` and `\n` are escape sequences. For example, `\antiuav410` became a BEL character followed by `ntiuav410`.
   - **Fix:** paths are converted to `/` when they are written, which Windows also accepts. The change is in `lib/test/evaluation/environment.py` and `lib/train/admin/environment.py`.
   - Also, `realpath` was replaced with `abspath`, so a mapped network drive (e.g. `F:`) is not turned into a `\\server\share` path.
2. **`mmseg` is not installed.**
   - The decoder only uses mmseg's `BaseDecodeHead` as a base class that stores a few attributes. Installing mmseg on Windows is very hard, because it requires the compiled `mmcv`.
   - **Fix:** `lib/models/layers/decoder.py` adds a minimal stand-in base class, used only when mmseg is not installed.
   - **Why we know it is correct:** the tracker loads the checkpoint with `strict=True`, so any missing or extra weight would raise an error, and loading succeeds.
3. **Packages missing from the README.** The code imports them even at test time, so they must be installed:
   - `jpeg4py`: the wrapper is enough. If its C library is missing, the code falls back to OpenCV.
   - `visdom`: needs `setuptools<81` and `--no-build-isolation`.
   - `cvxopt`: used by the non-uniform sampling grid.
4. **Versions.**
   - The README versions are kept: Python 3.9 and PyTorch 2.0.0 with CUDA 11.8.
   - A newer driver (here CUDA 13.2) runs them, because NVIDIA drivers are backward compatible. No CUDA Toolkit install is needed, because the PyTorch wheel bundles it.
   - `timm` is pinned to 0.6.13, because the code uses its old API.
5. **`--threads 0`.** CUDA with multiprocessing tends to hang on Windows, so the run uses a single process.
6. **uv install flag.** The requirements file adds the PyTorch index, and uv takes each package only from the first index that has it. The PyTorch index carries old `packaging` / `numpy`, so the install fails. **Fix:** `--index-strategy unsafe-best-match` (see *How to run*).
7. **`lib/train/data`.** My earlier, non-fork repository was missing this package, so `import lib.train.data` failed. This fork is based on the original code, so all 10 files are present.

### How to run

**1. Environment** (no conda needed; [uv](https://github.com/astral-sh/uv) or plain `pip` both work):
```powershell
uv venv venv --python 3.9
uv pip install --python venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-windows.txt
uv pip install --python venv\Scripts\python.exe --no-build-isolation visdom
```
`--index-strategy unsafe-best-match` is required. The requirements file adds the PyTorch index
(for the `+cu118` wheels). By default uv takes each package only from the *first* index that has it,
and the PyTorch index also carries old copies of `packaging`, `numpy`, `pillow` and others. Without the flag
the install fails with `there is no version of packaging==26.3`. (Plain `pip` does not need it.)

**2. Paths:**
```powershell
venv\Scripts\python.exe tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

**3. Weights** (not included): download `focustrack_single/FocusTrack_ep0030.pth.tar` from the
[authors' Google Drive](https://drive.google.com/drive/folders/1ndhwjIIymsppHknSjLUd40NmPmQ4XZhi) and put it at:
```
output/checkpoints/train/focustrack/focustrack_single/FocusTrack_ep0030.pth.tar
```

**4. Data** (not included): download [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) and place the test sequences as:
```
data/antiuav410/test/<sequence_name>/000001.jpg ... IR_label.json
```
The archive is 9.4 GB. Any subset of the test sequences works; the code runs on whatever is in that folder.

**5. Run the tracker:**
```powershell
venv\Scripts\python.exe tracking/test.py --tracker_name focustrack --tracker_param focustrack_single --dataset_name antiuav410_test --threads 0 --num_gpus 1 --run_id 30
```
Results are written to `output/test/tracking_results/focustrack/focustrack_single_030/antiuav410/`.

**6. Videos and metrics:**
```powershell
venv\Scripts\python.exe tools/make_videos_and_metrics.py
```
This writes `results/metrics.txt`, `results/metrics.json` and `results/videos/*.mp4`.

### Repository contents
| Path | What |
|---|---|
| `lib/`, `tracking/`, `experiments/`, `annos/` | FocusTrack code (with the Windows fixes above) |
| `tools/make_videos_and_metrics.py` | draws the videos and computes Precision / Success / State Accuracy |
| `results/` | `summary.md` (Hebrew), `metrics.txt`, `metrics.json`, `videos/` |
| `requirements-windows.txt` | exact package versions used |
| `README_original.md` | the authors' original README |

Not included: the dataset, the model weights, and the Python environment.
