# 網球場地自動偵測

從單張圖片或固定鏡頭影片偵測底線、單雙打邊線、發球線與中央發球線。使用 OpenCV 的白線擷取、Hough 直線偵測和標準球場模板配對，可在 CPU 執行。

## 執行

如只需要球場偵測，在專案根目錄執行下列命令。若已完成主專案安裝，可以直接執行偵測命令。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-court.txt
.\.venv\Scripts\python.exe working/court/detect_court.py ./match.mp4 --output ./court_output
```

也可以輸入 `.jpg` 或 `.png`。影片預設均勻抽取 5 幀。`--start` 和 `--end` 可以指定取樣時段（秒，含起點、不含終點），例如只用前 5 秒內的固定構圖。

```powershell
python working/court/detect_court.py ./match.mp4 --start 0 --end 5 --samples 5 --output ./court_output
```

`--max-width` 設定偵測時的最大影像寬度，預設 960。線段配對的分數偏低時，程式會嘗試較小尺度，以減少圍欄或壓縮細節的干擾。輸出座標一律以原始解析度表示。

### 匯出疊線影片

加上 `--overlay-video` 就會逐幀疊上與預覽圖相同的球場線和四角，匯出 `court_overlay.mp4`，並保留原片音軌。需要將支援 `libx264` 的 FFmpeg 加入 `PATH`，可以用 `ffmpeg -version` 確認終端機找得到它。若指定起訖時間，MP4 也會裁成相同區段。

```powershell
python working/court/detect_court.py ./match.mp4 --start 0 --end 5 --overlay-video --output ./court_output
```

### 跑有切鏡的完整影片

`track_video.py` 逐幀讀取影片，在球場標線足以校正時追蹤位置，也支援近端底線或角點被裁出的低視角。切鏡或場線失去足夠驗證時暫停疊線，直到能重新確認球場。輸出的 `court_overlay.mp4` 保留全片影格和原始音軌。匯出同樣使用 FFmpeg。

```powershell
python working/court/track_video.py ./highlights.mp4 --output ./court_output
```

## 輸出

- `court_overlay.png` 顯示參考影格、球場線及雙打外框四角。
- `court_overlay.mp4` 是加上 `--overlay-video` 時輸出的 H.264 影片。
- `court.json` 包含 `image_to_court`、`court_to_image`、原始影像尺寸、球場四角、參考影格索引，以及抽樣和一致性計數。影格索引從 0 開始，圖片的索引為 `null`。
- `score` 是全場模板線貼合亮線的比例（每條線等權）。`validation_score` 衡量未用於初始配對的球場線貼合度。兩者是幾何比對分數。

同一固定構圖可共用同一組矩陣。從底線後方拍攝時，俯視座標原點在遠端左側的雙打外框角，`X` 往右、`Y` 往近端。單位為公尺。

## 接到既有球追蹤

假設 `(x, y)` 是原始影格上的估計觸地點。若你的追蹤器使用縮圖或裁切圖，先轉回原始畫面座標。`image_to_court` 對應的是球場平面，直接輸入空中的球心會得到相機視線與地面的交點。

```python
import json
from pathlib import Path

import cv2
import numpy as np

calibration = json.loads(Path("court_output/court.json").read_text(encoding="utf-8"))
image_to_court = np.asarray(calibration["image_to_court"], dtype=np.float64)
pixel = np.array([[[820.0, 613.0]]], dtype=np.float64)  # 換成原始影格的觸地點
court_x, court_y = cv2.perspectiveTransform(pixel, image_to_court)[0, 0]
```

實作以直立、從底線一側觀看的全場畫面為目標。球場只露出一小部分、白線對比不足或抽樣影格的定位結果分歧時，程式可能回傳失敗。遇到平移、變焦或切鏡時，需各自取得對應構圖的校正。第一次使用新機位時，請用疊線圖檢查遠端底線和發球線的對齊情況。

球場幾何依循 [ITF 球場規格](https://itf.uberflip.com/i/623773-rules-of-tennis/5)，定位流程參考 [Farin 等人的球場模型校正](https://www.dirk-farin.net/publications/data/Farin2004b.pdf)。
