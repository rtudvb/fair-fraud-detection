# Notebooks

正式實驗都由指令執行（`make benchmark` 等，見 [`docs/experiments.md`](../docs/experiments.md)），
結果存成 `results/` 與 `reports/paper/` 裡的表格和圖。Notebook 用來**看結果、補充探索、寫報告時查數字**，
不放核心運算邏輯（邏輯都在 `src/frauddetect/`，才能測試、才能重跑）。

命名：`<編號>_<主題>.ipynb`，編號對應實驗（`docs/proposal.md` §7）。

## 範本

```python
import pandas as pd
from frauddetect.config import load_config
from frauddetect.viz import use_report_style

config = load_config()
use_report_style()
results = config["paths"]["results"]

# 例：讀 E2 的模型比較表
default = pd.read_csv(results / "benchmark" / "e2_default.csv")
default[["model", "family", "tpr_at_fpr", "tpr_ci_low", "tpr_ci_high", "fpr_ratio"]]
```

```python
# 例：讀快取的模型分數，自己畫圖或算指標
from frauddetect.experiments import make_context
from frauddetect.experiments.scoring import get_scores

ctx = make_context("notebook")
scores = get_scores(ctx, "lightgbm", protocol="final").part("test")  # 已跑過就直接讀快取
```
