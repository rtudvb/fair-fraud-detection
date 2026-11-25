# 實驗操作手冊

所有實驗都已寫好並用假資料測試過。裝好環境（[`compute.md`](compute.md)）、下載資料（`make data`）後，
照這份手冊執行即可。每個實驗都是一個指令，結果自動存成報告要用的圖表。

## 先做這一步：用假資料確認整條流程

```bash
make pipeline-check     # 約 5–10 分鐘；所有實驗用假資料各跑一次
```

輸出在 `results/_synthetic/`，**只用來確認程式能跑，數字沒有意義，不能放進報告**。
看到最後一個 `=== app-data ===` 且沒有紅字 `ERROR`（TabPFN／TabICL 授權未設定時的錯誤除外）就代表正常。

## 實驗一覽

| 指令 | 實驗 | 回答 | 主要圖表（`reports/paper/figures/`） | 報告章節 | 粗估時間（RTX 2080 Ti） |
|---|---|---|---|---|---|
| `make eda` | E0 資料探索 | 背景 | `e0_monthly`、`e0_age`、`e0_missing` | 3.2 | 數分鐘 |
| `make reproduce` | E1 重現原論文 | RQ1 | `e1_figure1` | 4.1 | 3–8 小時（可中斷續跑） |
| `make benchmark` | E2 十個模型（預設參數） | RQ2 | `e2_tpr`、`e2_cost_vs_performance`、`e2_fairness_vs_performance` | 4.2 | 1–3 小時 |
| `make tuned` | E3 調參組 | RQ2 | `e3_default_vs_tuned` | 4.2 | 4–10 小時（可續跑） |
| `make learning-curve` | E4 學習曲線 | RQ2、RQ6 | `e4_learning_curve` | 4.2 | 1–3 小時（可續跑） |
| `make fairness` | E5–E6 年齡公平性 | RQ3 | `e5_group_fpr`、`e5_cutoff_sensitivity`、`e6_tradeoff` | 4.3 | 20–60 分鐘 |
| `make decision` | E7 三級分流 | RQ4 | `e7_cost_breakdown`、`e7_tiers`、`e7_cost_surface` | 4.4 | 數分鐘 |
| `make explain` | E8 SHAP 與規則 | RQ5 | `e8_shap_importance`、`e8_shap_beeswarm`、`e8_shap_dependence` | 4.5 | 10–20 分鐘 |
| `make monitor` | E9 上線監控 | 第五章 | `e9_monthly` | 5.5 | 數分鐘 |
| `make small-bank` | E10 小銀行困境 | RQ6 | `e10_small_bank` | 4.6 | 30–90 分鐘 |
| `make app-data` | E11 決策支援系統資料 | 展示 | （`app/data/score_histogram.json`） | 5.6 | 數分鐘 |

時間是粗估，實際依 CPU、記憶體與設定而定。表格（`reports/paper/tables/*.md`）與完整數字（`results/<實驗>/*.csv`）同時產生。

## 建議執行順序（時間不夠時）

後面的實驗會重複使用前面算好的模型分數，所以順序有影響：

1. **第一天**：`make eda` → `make benchmark` → `make fairness` → `make decision` → `make explain` → `make monitor`
   （核心結果都出來了，可以開始寫第四章）
2. **第一個晚上**（`tmux` 裡）：`make reproduce`
3. **第二個晚上**：`make tuned`，接著 `make learning-curve`
4. **11/15 檢查點後**：`make small-bank`（決定做 RQ6 的話）、`make app-data`

一次全部跑：`make experiments`（約 1–2 個晚上）。

## 長時間執行與中斷

- **用 tmux**：`tmux new -s exp` → `source .venv/bin/activate` → `make experiments`，
  按 `Ctrl+b` 再按 `d` 離開；`tmux attach -t exp` 回來看。
- **可以中斷續跑**：E1、E3、E4 每完成一個模型就存檔，中斷後重下同一個指令，會從停下的地方繼續。
- **模型分數有快取**：存在 `results/scores/`。同一個模型不會重複訓練；想重算某個模型，刪掉對應檔案即可。
- **單一模型失敗不會中斷整批**：失敗原因寫在表格的 `status` 欄和終端機的 `ERROR` 訊息。
- **只跑部分模型**：`python -m frauddetect run benchmark --models lightgbm catboost`
- **跳過某些實驗**：`python -m frauddetect run all --skip tuned small-bank`

## 常見狀況

| 狀況 | 處理 |
|---|---|
| `CUDA out of memory` | 把 `configs/experiments.yaml` 的 `max_train_rows`（TabPFN、TabICL）調小，或調小 `runtime.predict_batch_size`；仍不行就改在 Kaggle 跑該模型 |
| TabPFN 授權錯誤 | 見 [`compute.md`](compute.md) 第 7-2 步 |
| 記憶體不足（程式被系統砍掉） | 調大 `.wslconfig` 的 `memory`（[`compute.md`](compute.md) 第 4 步） |
| 某模型跑太久 | 先用 `--models` 排除它，最後再單獨跑 |
| 想改成本假設 | 改 `configs/costs.yaml`，重跑 `make decision`（幾分鐘） |

## 每次執行留下的紀錄

`results/<實驗>/meta.json` 會記下：執行時間、Python／PyTorch 版本、GPU 型號、git commit；
`config.yaml` 是當次的完整設定。報告的「實驗環境」與「代價」比較就用這些資料。

## 台灣詐騙統計（E0 選用）

把 [`templates/taiwan_fraud_stats.csv`](templates/taiwan_fraud_stats.csv) 複製到
`data/external/taiwan_fraud_stats.csv`，依 165 打詐儀表板或警政署公開資料補齊各期數字並保留 `source` 欄，
再跑一次 `make eda` 就會產生 `e0_taiwan_fraud_stats` 圖表。

## 互動 Demo

```bash
make app-data     # 由 E7 用的模型分數產生彙總分布（不含任何原始資料）
make app          # 瀏覽器開啟 http://localhost:8501
```

沒有 `app/data/score_histogram.json` 時，Demo 會顯示標註清楚的示範資料。

## 哪些東西會上傳 GitHub

| 會上傳 | 不會上傳（`.gitignore`） |
|---|---|
| `reports/paper/figures/`、`reports/paper/tables/`（圖表與彙總表） | `data/`（BAF 原始資料，授權禁止散布） |
| `app/data/score_histogram.json`（彙總分布） | `results/`（含逐筆分數，與原始資料一一對應） |
| 程式碼、設定檔、文件 | 權杖、`kaggle.json` |
