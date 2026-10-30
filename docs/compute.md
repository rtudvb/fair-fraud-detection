# 運算環境設定（Windows 實驗室桌機 + RTX 2080）

| 環境 | 用途 |
|---|---|
| **實驗室桌機（Windows + RTX 2080，主力）** | 所有實驗；長時間任務（600 次 LightGBM、學習曲線）可以整晚跑 |
| Kaggle Notebook（備援） | 2080 的 8GB 顯示記憶體不夠時，用 16GB GPU 跑基礎模型的大資料量 |

**做法**：在 Windows 上裝 **WSL2（Windows 內建的 Ubuntu Linux）**，程式在 Ubuntu 裡跑，
用 Windows 的 VS Code 連進去編輯。這樣 `make`、`tmux` 等指令都能用，GPU 也能直接使用，
而且和 CI、Kaggle 的 Linux 環境一致，不會出現「我電腦可以跑、別的地方不行」。

> 慣例：`PS>` 開頭的指令在 **Windows PowerShell** 執行；`$` 開頭的在 **Ubuntu（WSL）終端機** 執行。
> 每一步最後都有「✅ 檢查」，確認通過再往下做。

**預計時間**：第一次約 1–1.5 小時（大部分在等下載）。

---

## 第 0 步：開始前確認

1. **共用的電腦先問**：跟學長姐或老師確認你可以安裝 WSL、使用 GPU，以及有沒有系統管理員權限。
2. **Windows 版本**：按 `Win + R`，輸入 `winver`。需要 **Windows 11**，或 **Windows 10 版本 21H2 以上**。
3. **硬碟空間**：`C:` 槽至少剩 **40 GB**（WSL + 套件約 15 GB、資料約 5 GB、模型權重與暫存檔）。
4. **記憶體**：工作管理員（`Ctrl + Shift + Esc`）→ 效能 → 記憶體，記下總容量（第 4 步會用到）。
5. **虛擬化**：工作管理員 → 效能 → CPU，右下角「**虛擬化：已啟用**」。
   如果是「已停用」，要進 BIOS 開啟（Intel 叫 VT-x，AMD 叫 SVM），不確定就請學長姐或助教幫忙。

✅ 檢查：Windows 版本符合、C 槽空間夠、虛擬化已啟用。

---

## 第 1 步：更新 NVIDIA 驅動程式（在 Windows 上）

WSL2 會直接使用 **Windows 的** NVIDIA 驅動程式，所以只要在 Windows 裝好驅動。

1. 到 NVIDIA 官網驅動程式下載頁，選 **GeForce RTX 20 Series → RTX 2080 → Windows**，
   下載最新版 **Game Ready** 或 **Studio** 驅動程式並安裝（或用 NVIDIA App 更新）。
2. 安裝完重新開機。
3. 開 PowerShell 執行：

   ```powershell
   PS> nvidia-smi
   ```

✅ 檢查：表格中看得到 **NVIDIA GeForce RTX 2080**，右上角有 **Driver Version** 和 **CUDA Version**。
把 CUDA Version 的數字記下來（第 6 步會用到）。

> ⚠️ 之後**不要在 Ubuntu 裡安裝任何 NVIDIA 驅動程式**，會把 WSL 的 GPU 支援弄壞。

---

## 第 2 步：安裝 WSL2 與 Ubuntu

1. 開始功能表搜尋 PowerShell → **右鍵 → 以系統管理員身分執行**：

   ```powershell
   PS> wsl --install -d Ubuntu-24.04
   ```

2. 完成後**重新開機**。開機後會自動跳出 Ubuntu 視窗（沒有的話從開始功能表開「Ubuntu 24.04」），
   設定 Linux 的使用者名稱與密碼。輸入密碼時畫面不會顯示字元，這是正常的；請記住這組密碼（`sudo` 會用到）。
3. 回到 PowerShell（一般權限即可）：

   ```powershell
   PS> wsl --update
   PS> wsl -l -v
   ```

✅ 檢查：`wsl -l -v` 顯示 `Ubuntu-24.04`，**VERSION 是 2**。
（若是 1，執行 `wsl --set-version Ubuntu-24.04 2`。）

---

## 第 3 步：在 Ubuntu 裡安裝基本工具

開「Ubuntu 24.04」終端機：

```bash
$ sudo apt update && sudo apt upgrade -y
$ sudo apt install -y build-essential git make curl tmux python3-venv python3-pip gh
```

✅ 檢查：

```bash
$ nvidia-smi          # 要看到 RTX 2080（這是 Windows 驅動提供的）
$ python3 --version   # 3.10 以上（Ubuntu 24.04 內建 3.12）
$ git --version && make --version | head -1 && tmux -V
```

> `nvidia-smi` 在 Ubuntu 裡找不到或看不到 GPU：回第 1 步更新 Windows 驅動，
> 再在 PowerShell 執行 `wsl --update` 與 `wsl --shutdown`，重開 Ubuntu。

---

## 第 4 步：調整 WSL 可用的記憶體

WSL 預設只能用電腦一半的記憶體。BAF 單一資料集 100 萬筆，同時載入多個資料集或跑大模型時可能不夠。

1. 在 Windows 檔案總管網址列輸入 `%UserProfile%` 進入使用者資料夾，新增檔案 **`.wslconfig`**
   （注意前面有點、沒有 `.txt` 副檔名；用記事本存檔時「存檔類型」選「所有檔案」）。內容：

   ```ini
   [wsl2]
   # 依電腦記憶體調整：總記憶體 32GB 填 24GB；16GB 填 12GB（留一些給 Windows）
   memory=24GB
   swap=16GB
   ```

2. PowerShell 執行 `wsl --shutdown`，等 10 秒再開 Ubuntu。

✅ 檢查：Ubuntu 執行 `free -h`，`Mem:` 的 total 接近你設定的數字。

---

## 第 5 步：VS Code 與 GitHub

### 5-1 安裝 VS Code（Windows）

1. 安裝 Windows 版 VS Code（安裝時勾選「加入 PATH」）。
2. 擴充套件（左側方塊圖示）安裝：**WSL**、**Python**、**Jupyter**、**Ruff**。

### 5-2 設定 Git 與登入 GitHub（Ubuntu）

```bash
$ git config --global user.name "Chang Ren-Yu"
$ git config --global user.email "168074462+rtudvb@users.noreply.github.com"
$ gh auth login      # 依序選 GitHub.com → HTTPS → Yes → Login with a web browser
```

✅ 檢查：`gh auth status` 顯示 Logged in to github.com account **rtudvb**。

### 5-3 下載專案（放在 Ubuntu 的家目錄，**不要**放在 `/mnt/c/...`）

```bash
$ mkdir -p ~/projects && cd ~/projects
$ git clone https://github.com/rtudvb/fair-fraud-detection.git
$ cd fair-fraud-detection
$ code .
```

> 放在 `/mnt/c`（Windows 的 C 槽）會讓讀寫資料慢上好幾倍。想用 Windows 檔案總管看 Ubuntu 的檔案，
> 網址列輸入 `\\wsl$\Ubuntu-24.04\home\<你的 Linux 帳號>\projects`。

✅ 檢查：VS Code 視窗左下角顯示 **「WSL: Ubuntu-24.04」**。

---

## 第 6 步：建立 Python 環境與安裝套件

在 VS Code 裡開終端機（`` Ctrl + ` ``），確認目前在專案資料夾：

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate          # 之後每次開新終端機都要執行（提示字元前會出現 (.venv)）
$ python -m pip install --upgrade pip
$ pip install torch                   # PyTorch（含 GPU 支援，約 2–3 GB）
```

✅ 檢查 PyTorch 抓得到 GPU：

```bash
$ python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最後要看到 `True NVIDIA GeForce RTX 2080`。

> **出現 `False` 或錯誤**：比較兩個數字——第 1 步記下的 `nvidia-smi` **CUDA Version**，
> 和上面印出的 `torch.version.cuda`。前者必須**大於或等於**後者。不夠的話先更新 Windows 驅動；
> 仍不行就改裝較舊 CUDA 版本的 PyTorch（到 PyTorch 官網「Get Started」選 Linux / Pip / 對應的 CUDA 版本，
> 複製它給的安裝指令）。

接著安裝專案的其他套件：

```bash
$ make install-all
$ make models          # 10 個模型都要顯示 installed
$ make test            # 單元測試全部通過
```

✅ 檢查：`make models` 全部 installed、`make test` 全部通過（一排綠點、最後 100%）。

最後讓 VS Code 使用這個環境：`Ctrl + Shift + P` → **Python: Select Interpreter** → 選 `.venv/bin/python`。

---

## 第 7 步：帳號與權杖（Kaggle、Hugging Face）

權杖（token）等同密碼：**只放在自己電腦，不要寫進程式碼、不要 commit、不要傳給別人**。

### 7-1 Kaggle（下載資料用）

1. 登入 Kaggle → 右上角頭像 → **Settings** → API → **Create New Token**，
   會下載 `kaggle.json` 到 Windows 的「下載」資料夾。
2. 複製到 Ubuntu（把 `<Windows使用者名稱>` 換成你的）：

   ```bash
   $ mkdir -p ~/.kaggle
   $ cp /mnt/c/Users/<Windows使用者名稱>/Downloads/kaggle.json ~/.kaggle/
   $ chmod 600 ~/.kaggle/kaggle.json
   ```

✅ 檢查：`ls -l ~/.kaggle/kaggle.json` 顯示 `-rw-------`。

### 7-2 Hugging Face（基礎模型權重用）

1. 註冊／登入 Hugging Face。
2. 打開 <https://huggingface.co/Prior-Labs/tabpfn_3_5>，**接受使用條款**（TabPFN-3.5 權重為受限下載）。
3. 頭像 → Settings → **Access Tokens** → Create new token，類型選 **Read**，複製 `hf_` 開頭的字串。
4. 寫進 Ubuntu 的環境設定（只存在你的電腦上）：

   ```bash
   $ echo 'export HF_TOKEN=hf_你的權杖' >> ~/.bashrc
   $ source ~/.bashrc
   ```

✅ 檢查：`echo $HF_TOKEN` 印出 `hf_` 開頭的字串。

> TabPFN 若在第 8 步仍要求授權，依錯誤訊息到 <https://ux.priorlabs.ai> 登入接受授權，
> 取得權杖後同樣用 `echo 'export TABPFN_TOKEN=...' >> ~/.bashrc` 加入。

---

## 第 8 步：總檢查（GPU + 10 個模型）

```bash
$ make smoke
```

這個指令會印出 Python、PyTorch、GPU 資訊，並用一小份假資料讓 10 個模型各訓練一次
（第一次會下載 TabPFN 與 TabICL 的權重，需要幾分鐘）。

✅ 檢查：

- `GPU` 那行顯示 **NVIDIA GeForce RTX 2080, 8.0 GB**，下一行是 `GPU test matrix multiply OK`
- 最後一行是 **`10/10 models OK`**

> 假資料是隨機產生的，AUC 數字沒有意義（甚至可能低於 0.5），只用來確認「跑得起來」。

| 狀況 | 原因與處理 |
|---|---|
| `GPU NOT available` | 回第 6 步檢查 `torch.cuda.is_available()` |
| `tabpfn` FAIL，訊息含 `GatedRepo` 或 `authentication` | 第 7-2 步：確認已接受使用條款、`echo $HF_TOKEN` 有值 |
| `tabicl` FAIL，訊息含 `Proxy` 或 `Connection` | 連不到 Hugging Face（校園網路或防火牆），換網路或請網管協助 |
| 其他模型 FAIL | 重新執行 `make install-all`；仍失敗就把錯誤訊息貼給我 |

---

## 第 9 步：下載資料

```bash
$ make data        # 從 Kaggle 下載 BAF 並轉成 Parquet
$ make validate    # 檢查 6 個資料集
```

✅ 檢查：`make validate` 的 6 個資料集都顯示 `[OK]`、各約 1,000,000 rows，各月詐騙率約 0.01；
Variant III、V 會有「extra columns」提示，這是正常的。

資料只會存在這台電腦的 `data/raw/`，不會上傳到 GitHub。

接著執行 `make pipeline-check`（所有實驗用假資料跑一次，約 5–10 分鐘），通過後就可以照
[`experiments.md`](experiments.md) 開始跑正式實驗。

---

## 第 10 步：跑長時間實驗

### 用 tmux，關掉視窗也不會中斷

```bash
$ tmux new -s exp               # 開一個叫 exp 的工作階段
$ source .venv/bin/activate
$ <執行實驗的指令>
#   按 Ctrl + b，放開，再按 d → 離開畫面，但程式繼續跑
$ tmux attach -t exp            # 之後回來查看進度
```

### 讓電腦整晚不中斷

- Windows 設定 → 系統 → 電源：**接上電源時，睡眠設為「永不」**（螢幕關閉可以設定時間）。
- Windows Update → **暫停更新**（實驗期間避免自動重新開機）。
- 跑實驗時**保留一個 Ubuntu 終端機或 VS Code 視窗開著**（可以最小化）：
  所有 WSL 視窗都關閉後，Windows 可能自動關掉 WSL，連帶中斷實驗。
- 開跑前用 `nvidia-smi` 確認沒有其他人正在用 GPU。

### 從家裡連回實驗室桌機（選用）

1. 實驗室桌機的 VS Code：左下角帳號圖示 → **Turn on Remote Tunnel Access**，用 GitHub 帳號登入。
2. 家裡的 VS Code 安裝 **Remote - Tunnels** 擴充套件，用同一個 GitHub 帳號登入後就能連到桌機；
   連上後按 `Ctrl + Shift + P` → **WSL: Connect to WSL** 進入 Ubuntu。
3. 使用前先確認學校／實驗室的網路規定是否允許。

---

## 備援：Kaggle Notebook

2080 跑基礎模型的大資料量出現 `CUDA out of memory` 時，改用 Kaggle 的 16GB GPU：

1. Kaggle 帳號完成手機驗證（才能開 GPU 與網路）。
2. 新增 Notebook → Settings：**Accelerator 選 GPU、Internet 打開**；Add Input 加入資料集
   `sgpjesus/bank-account-fraud-dataset-neurips-2022`。
3. Add-ons → **Secrets** 加入 `HF_TOKEN`（與 `TABPFN_TOKEN`，若需要）。
4. Notebook 開頭：

   ```python
   # 取得程式碼（repo 為 private 時，改用上傳 zip 或 GitHub token）
   !git clone https://github.com/rtudvb/fair-fraud-detection.git
   %cd fair-fraud-detection
   !pip install -q -e ".[gbdt,dl,fm]"

   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

   from pathlib import Path
   from frauddetect.config import load_config
   config = load_config()
   config["paths"]["raw"] = Path("/kaggle/input/bank-account-fraud-dataset-neurips-2022")
   ```

---

## 注意事項（兩種環境都適用）

- **只存結果，不存資料**：commit 到 GitHub 的只能是指標、彙總結果與圖表；原始資料不能散布（CC BY-NC-ND 4.0）。
- **記錄硬體**：每次實驗把 GPU 型號與執行時間寫進結果檔，報告比較「代價」時要用。
- **先小後大**：先用小資料量確認程式正確，再跑全量；學習曲線的大資料量只跑表現最好的模型。
