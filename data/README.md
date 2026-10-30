# 資料（不進版控）

本資料夾的資料檔**不會**上傳到 GitHub（見 `.gitignore`）。

## BAF（Bank Account Fraud）資料集

- 來源：Jesus et al. (2022), NeurIPS Datasets & Benchmarks
- 下載：[Kaggle — bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- 授權：**CC BY-NC-ND 4.0**（非商業、不得散布改作版本）→ 原始資料與加工後的資料都不能放上 GitHub

### 下載方式

**方式一：Kaggle API（本機）**

1. 在 Kaggle 帳號設定頁建立 API token，把 `kaggle.json` 放到 `~/.kaggle/`
   （Windows + WSL 的詳細步驟見 [`docs/compute.md`](../docs/compute.md) 第 7-1 步）
2. 執行：

```bash
pip install -e ".[data]"
make data        # 下載到 data/raw/ 並轉成 Parquet
make validate    # 檢查 6 個資料集的欄位與各月詐騙率
```

**方式二：Kaggle Notebook（建議用來跑 GPU 實驗）**

在 Notebook 右側「Add Input」加入這個資料集，再把設定檔的 `paths.raw` 指向
`/kaggle/input/bank-account-fraud-dataset-neurips-2022`。

**方式三：手動下載**

從 Kaggle 網頁下載後解壓縮到 `data/raw/`，檔名應為 `Base.csv`、`Variant I.csv` … `Variant V.csv`，
再執行 `make validate`。

## 台灣開放資料（第二資料來源）

政府資料開放平臺「警政署電信詐欺犯罪資料」等，手動下載後放到 `data/external/`，
並在報告中註明下載日期。
