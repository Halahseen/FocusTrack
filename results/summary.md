# סיכום: הרצת FocusTrack על Anti-UAV410

**המאמר:** FocusTrack: A Self-Adaptive Local Sampling Algorithm for Efficient Anti-UAV Tracking ([arXiv 2504.13604](https://arxiv.org/abs/2504.13604))
**הקוד:** https://github.com/vero1925/FocusTrack
**תאריך ההרצה:** 2026-11-04

כל שמונת השלבים הושלמו: FocusTrack (קונפיג `focustrack_single`) רץ על 3 רצפים מה-test של Anti-UAV410 בכ-41 FPS, נוצרו 3 סרטוני MP4 עם הקופסאות, וחושבו המדדים.

## התוצאות

| רצף | פריימים | Precision (20px) | Success (AUC) | State Accuracy |
|---|---|---|---|---|
| 02_6319_1500-2999 | 1500 | 80.00% | 50.27% | 51.07% |
| 20190925_111757_1_1 | 1000 | 100.00% | 78.85% | 80.42% |
| 3700000000002_130905_1 | 1500 | 99.87% | 72.29% | 73.72% |
| **ממוצע** | 4000 | **93.29%** | **67.13%** | **68.40%** |

- **מה מודדים:**
  - **Precision:** אחוז הפריימים שבהם מרכז הקופסה של ה-tracker נמצא עד 20 פיקסלים ממרכז האמת.
  - **Success (AUC):** ממוצע על פני 21 ספי IoU בין 0 ל-1 של אחוז הפריימים שבהם ה-IoU גבוה מהסף. זה השטח מתחת לעקומת ה-Success.
  - **State Accuracy (SA):** ממוצע על כל הפריימים. בפריים שהרחפן נראה בו נכנס ה-IoU, ובפריים שהוא לא נראה בו נכנס 1 אם ה-tracker דיווח "אין מטרה" ו-0 אם לא.
- **מקור הנוסחאות:** הן מועתקות מקוד ההערכה הרשמי של Anti-UAV410 (`Evaluation_for_SA.py`, `experiments/anti_uav.py`), כולל חישוב המרכז `x + (w-1)/2`. לכן אפשר להשוות אותן לטבלה במאמר. אבל 3 רצפים מתוך 120 הם לא מדגם מייצג.
- **מהירות:** בערך 41 FPS על NVIDIA RTX A4000. ההרצה של כל שלושת הרצפים לקחה 4 דקות.
- **דוגמת כישלון טובה להצגה:** ברצף `02_6319`, בפריימים 1124 עד 1389 בערך, הרחפן עף מול בניין. ה-tracker נתפס לחלון בבניין ואיבד אותו, ובסביבות פריים 1390 חזר אליו. זה מתאים לתכונה "רקע עמוס" (DBC) שמסומנת לרצף הזה ב-dataset.

## איפה הכול נמצא
- **סרטונים ומדדים:** `F:\seminar\results_videos\`. יש שם 3 קבצי MP4 (בקידוד H.264, נפתחים גם ב-PowerPoint) ואת `metrics.txt` ו-`metrics.json`.
  - אדום = FocusTrack, ירוק = האמת. בפינה מופיעים מספר הפריים וה-IoU.
  - הגדלתי את התמונה פי 2, כי הרחפנים בגודל של בערך 8×8 פיקסלים בלבד.
- **הסקריפט שמייצר סרטונים ומדדים:** `F:\seminar\code\make_videos_and_metrics.py`.
- **הקוד:** `F:\seminar\code\FocusTrack`.
- **המשקולות:** `output\checkpoints\train\focustrack\focustrack_single\FocusTrack_ep0030.pth.tar` (1.4GB). ירדו אוטומטית מה-Drive.
- **הנתונים:** `data\antiuav410\test\` (3 רצפים).
- **תוצאות ה-tracker הגולמיות:** `output\test\tracking_results\focustrack\focustrack_single_030\antiuav410\`.
- **סביבת Python:** `D:\envs\focustrack`, כי בכונן F יש רק 5GB פנויים.

## מה נשבר ולמה התיקון עובד
1. **אין git, conda ו-Python במחשב.** הורדתי את הריפו כקובץ ZIP מ-GitHub (אותו קוד, בלי היסטוריה). במקום conda השתמשתי ב-`uv`, כלי שמוריד בעצמו Python 3.9 ויוצר סביבה.
2. **התאמת הגרסאות ל-CUDA.** הדרייבר כאן הוא CUDA 13.2, וה-README מבקש 11.8. דרייברים של NVIDIA תואמים לאחור, ו-PyTorch מגיע עם ספריות ה-CUDA בתוכו (לא צריך להתקין CUDA Toolkit), אז **שמרתי את הגרסאות המקוריות**: Python 3.9 ו-torch 2.0.0+cu118.
   - torchvision היא 0.15.1, כי זו הגרסה ב-pip שמתאימה ל-torch 2.0.0.
   - `timm` ננעל על 0.6.13, כי הקוד משתמש ב-API ישן שלה.
3. **ה-dataset שוקל 9.4GB, יותר מהמקום הפנוי.** ל-ZIP יש "תוכן עניינים" (Central Directory) בסוף הקובץ, עם המיקום של כל קובץ בתוכו. קראתי רק אותו, ואז הורדתי רק את הבייטים של 3 הרצפים (HTTP Range). בסך הכול ירדו כ-55MB.
4. **`create_default_local_file.py` כתב נתיבים שבורים.** הוא כותב נתיבי Windows עם `\` לתוך מחרוזת Python, ושם `\a`, ‏`\t` ו-`\n` הם תווים מיוחדים. למשל `\antiuav410` הפך לתו "צלצול" ואחריו `ntiuav410`.
   - בנוסף, כונן F הוא כונן רשת ממופה, והסקריפט החליף אותו בנתיב רשת מלא (`\\YARDEN\...`).
   - הקוד נכתב ל-Linux, ושם זה לא קורה.
   - **התיקון:** כתבתי מחדש את `lib/test/evaluation/local.py` עם `/`, כמו שה-README מרשה.
5. **חבילות חסרות שה-README לא מזכיר או שמסומנות כאופציונליות:** `jpeg4py`, `visdom` ו-`cvxopt` פשוט הותקנו.
   - ההתקנה של visdom נכשלה בהתחלה, כי היא צריכה את `pkg_resources`, שהוסר מגרסאות חדשות של setuptools. התקנתי `setuptools<81` ובניתי אותה עם `--no-build-isolation`.
   - `jpeg4py` מספיק כ-wrapper. אם ספריית ה-C שלה חסרה, הקוד עצמו עובר לקריאת תמונות עם OpenCV.
6. **`mmseg` חסר.** הקוד משתמש בו רק כמחלקת בסיס ל-decoder (`BaseDecodeHead`), ומה שהוא לוקח ממנה זה כמה שדות. ההתקנה שלו ב-Windows קשה מאוד, כי הוא דורש את `mmcv` המקומפל.
   - **התיקון:** כתבתי ב-`lib/models/layers/decoder.py` מחלקה חלופית מינימלית, שמשמשת רק אם mmseg לא מותקן.
   - **ההוכחה שהיא נכונה:** ה-tracker טוען את המשקולות עם `strict=True`, כלומר כל פרמטר חסר או מיותר היה עוצר את ההרצה עם שגיאה, והטעינה עברה בהצלחה.

## דברים שכדאי לדעת
- **הרצה מחדש של FocusTrack** (מתוך `F:\seminar\code\FocusTrack`):
  ```
  D:\envs\focustrack\Scripts\python.exe tracking/test.py --tracker_name focustrack --tracker_param focustrack_single --dataset_name antiuav410_test --threads 0 --num_gpus 1 --run_id 30
  ```
  השתמשתי ב-`--threads 0` כי ריבוי תהליכים עם CUDA נוטה להיתקע ב-Windows.
- **יצירת הסרטונים והמדדים מחדש:**
  ```
  D:\envs\focustrack\Scripts\python.exe F:\seminar\code\make_videos_and_metrics.py
  ```
- **תיקייה ריקה שנשארה:** ב-`D:\focustrack_test_write` נשארה תיקייה ריקה מבדיקת הרשאות כתיבה, ולא הייתה לי הרשאה למחוק אותה. אפשר למחוק אותה ידנית.
