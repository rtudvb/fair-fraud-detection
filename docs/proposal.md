# 研究計畫書

**從梯度提升樹到表格基礎模型：開戶詐騙偵測之公平與成本分析**

*From Gradient Boosting to Tabular Foundation Models: Fairness and Cost in Account-Opening Fraud Detection*

> 專案標語：**Catch Fraud, Not Customers**（抓詐騙，不誤傷好客戶）
>
> 課程：商業智慧（期中報告）｜作者：Chang Ren-Yu｜繳交：11 月底前上傳 ee-class 討論區（目標 11/28）
>
> 版本：v1.0（2026-10-24 定案）

---

## 目錄

1. [研究背景與動機](#1-研究背景與動機)
2. [研究目的](#2-研究目的)
3. [研究問題與假說](#3-研究問題與假說)
4. [研究貢獻](#4-研究貢獻)
5. [資料](#5-資料)
6. [研究方法](#6-研究方法)
7. [實驗矩陣與優先順序](#7-實驗矩陣與優先順序)
8. [報告大綱](#8-報告大綱)
9. [專案架構與程式品質](#9-專案架構與程式品質)
10. [時程與檢查點](#10-時程與檢查點)
11. [風險與對策](#11-風險與對策)
12. [授權、研究倫理與分工](#12-授權研究倫理與分工)
13. [初步參考文獻](#13-初步參考文獻)

---

## 1. 研究背景與動機

**台灣詐騙問題嚴重，人頭帳戶是金流核心。**

- 2025 年全台詐騙財損約 893 億元、受理約 16.2 萬件（媒體引用 165 打詐儀表板）。
- 警示帳戶從 2022 年約 6.6 萬個，增加到 2025 年初突破 15 萬個。
- 2025 年詐騙集團騙取民眾的自然人憑證大量線上開戶，刑事局指出當年約 1/5 人頭帳戶與此有關；17 家銀行一度暫停自然人憑證開戶，金管會要求銀行「依風險基礎」強化驗證（視訊、跨行驗證、電信認證、交易限額）後恢復受理。

**但 AI 偵測也會誤傷好客戶。**

- 銀行 AI「誤鎖薪轉戶」引發爭議，金管會要求提供 24 小時解控服務。
- BAF 原論文發現：表現最好的模型，對 50 歲以上正常申請者的誤判率是年輕族群的 3 倍以上。

**法規與產業同時在推進。**

- 《人工智慧基本法》2026/01 公布施行，第 3 條列出「公平與不歧視」「透明與可解釋」「問責」等原則；金管會 2024 年《金融業運用 AI 指引》要求避免演算法偏見。
- 2026/08 北富銀帶領 8 家銀行完成 AI 聯合學習防詐驗證，警示帳戶辨識精確率約提升兩倍、假警報減少。

**技術也在快速演進。**

- 表格資料長期由梯度提升樹（GBDT）主導；2024–2026 年出現新一代表格深度學習（TabM、RealMLP）與表格基礎模型（TabPFN-3.5、TabICLv2）。
- 2026 年研究指出 TabPFN 的公平性改善「有限且不穩定」，新模型在高風險決策中是否「又準、又公平、又划算」仍無定論。

**研究缺口**：BAF 原作者只測了「不考慮公平性」的 LightGBM，並明確鼓勵後續研究嘗試公平性方法；新一代模型在詐騙情境下的公平與成本表現，以及如何轉成符合台灣制度的決策，據我們所知尚少研究。

## 2. 研究目的

| 層次 | 目的 |
|---|---|
| 商業 | B1 提高詐騙開戶攔截率；B2 控制誤擋好客戶比例與人工審核成本；B3 確保不同年齡客群受到公平對待；B4 產出銀行可採用的風控規則與分級審核策略 |
| 研究 | 重現 NeurIPS 2022 基準 → 比較三代模型 → 加入公平與成本 → 提出決策模型 → 轉成知識與建議 |
| 交付 | 論文格式報告（PDF）、可重現的程式碼 repo、互動式 demo |

**完成標準**

- 重現結果與原論文的差異都能說明原因。
- 10 個模型 × 預設參數組／調參組完整比較。
- 報告至少 12 張圖、8 張表、25 篇參考文獻。
- `make all` 可重新產生所有圖表；核心指標有單元測試；CI 全部通過。
- 主要比較附 bootstrap 95% 信賴區間。

## 3. 研究問題與假說

### 核心（一定做）

| RQ | 問題 |
|---|---|
| **RQ1 重現** | 能否重現 BAF 論文的發現（100 組 LightGBM；最佳模型的 FPR 比約 0.3）？ |
| **RQ2 三代模型** | 傳統機器學習、深度學習、基礎模型，哪一類在詐騙偵測最好？代價（時間、GPU）多少？資料少時誰勝出？ |
| **RQ3 年齡公平性** | 模型是否對年長者不公平？公平性處理要犧牲多少偵測率？結論是否隨年齡切點改變？ |
| **RQ4 三級分流決策** | 在公平與人工審核人力的限制下，如何設定三級分流門檻讓期望損失最小？ |
| **RQ5 知識發現** | 哪些特徵最能預警詐騙開戶？能轉成哪些風控規則？ |

### 延伸（11/15 檢查點決定去留）

| RQ | 問題 |
|---|---|
| **RQ6 小銀行困境** | 資料少的小銀行，該用基礎模型，還是加入跨行聯合學習？ |

### 假說

- **H1**：在預設參數下，基礎模型在小資料量（≤ 1 萬筆）時優於 GBDT，資料量增加後差距縮小。
- **H2**：未做公平性處理的模型，對 50 歲以上族群的誤判率顯著較高（FPR 比 < 0.8）。
- **H3**：公平性結論對年齡切點敏感；不同切點會得到不同的 FPR 比。
- **H4**：三級分流的期望損失低於「固定 5% 誤判率」的二元決策。

## 4. 研究貢獻

> 報告 1.4 節使用此表（表 A）。用語一律寫「據我們所知」，不寫「首創」。

| 貢獻 | 類型 | 內容 | 對應 RQ |
|---|---|---|---|
| 一 | 新問題 | 在極度不平衡、會隨時間變化的詐騙情境中，同時評估三代模型（含 2026 年基礎模型）的效能、公平與成本 | RQ2、RQ3 |
| 二 | 新方法 | 對應台灣數位存款帳戶分類的三級分流決策模型，加入公平與人力限制 | RQ4 |
| 三 | 新發現 | 重現時發現論文正文（>50）與官方程式碼（≥50）的年齡分組不一致，延伸為年齡切點敏感度分析 | RQ1、RQ3 |
| 延伸 | 新情境 | 小銀行困境：基礎模型 vs 跨行聯合學習 | RQ6 |

**一句話特色**：不只比誰最準——把新一代 AI 模型放進「抓詐騙 vs. 不誤傷」的取捨，並提出對應台灣數位帳戶制度的三級分流決策模型。

## 5. 資料

### 5.1 主要資料：BAF（Bank Account Fraud）

| 項目 | 內容 |
|---|---|
| 來源 | Jesus et al. (2022), NeurIPS Datasets & Benchmarks；Kaggle 下載 |
| 內容 | 以真實、匿名化的銀行線上開戶詐騙資料，經生成模型合成 |
| 規模 | 6 個資料集（Base、Variant I–V），各 100 萬筆、30 個特徵 + `month` + `fraud_bool` |
| 期間 | 8 個月（`month` 0–7），各月詐騙率 0.85%–1.5% |
| 公平性欄位 | `customer_age`（以 10 歲為一級）、`income`、`employment_status` |
| 授權 | CC BY-NC-ND 4.0 → 資料**不放進 GitHub**，只提供下載腳本 |

各變體的偏誤設計（原論文 Table 1）：

| 資料集 | 偏誤類型 |
|---|---|
| Base | 接近原始資料：年長組佔 23%、詐騙率約為年輕組 2 倍 |
| Variant I | 組別大小差異加大，詐騙率相同 |
| Variant II | 詐騙率差異加大 |
| Variant III | 某組的可分離性較高 |
| Variant IV | 詐騙率差異只出現在訓練期（時間漂移） |
| Variant V | 可分離性差異只出現在訓練期（時間漂移） |

### 5.2 第二資料來源：台灣開放資料（背景與趨勢分析）

- 政府資料開放平臺「警政署電信詐欺犯罪資料」：每季更新，含案類、發生日期、發生地點。
- 165 打詐儀表板公布的統計數字（人工整理並註明出處）。
- 用途：2024–2026 台灣詐騙趨勢分析，放在第一章背景與第三章資料。

### 5.3 選用：FiFAR（Alves et al., 2023）

以 BAF Base 為基礎，加入 50 位模擬審核員的判斷與人力限制。若時間允許，用來估計三級分流模型中「人工審核的抓出率」。

## 6. 研究方法

### 6.1 研究流程

```
BAF 資料 ─→ 驗證與前處理 ─→ 時間切分 ─→ 10 個模型 ─→ 評估（效能／公平／成本）
                                              │
                    ┌─────────────────────────┼───────────────────────┐
                    ▼                         ▼                       ▼
              公平性處理                 三級分流決策              SHAP 與規則萃取
                    └─────────────────────────┼───────────────────────┘
                                              ▼
                              管理意涵、上線監控、報告與 demo
```

### 6.2 資料切分

- **重現（RQ1）**：完全依原論文——前 6 個月（`month` 0–5）訓練、後 2 個月（6–7）測試。
- **其他實驗**：`month` 0–4 訓練、5 驗證（調參與選門檻）、6–7 測試；選定設定後以 0–5 重新訓練，再於 6–7 評估。測試集只用一次。

### 6.3 模型陣容（3 類 10 個）

| 類別 | 模型 | 套件 |
|---|---|---|
| 傳統機器學習 | 邏輯迴歸、隨機森林 | scikit-learn |
| | XGBoost、LightGBM、CatBoost | 各官方套件 |
| 從頭訓練的深度學習 | MLP | PyTorch／scikit-learn |
| | TabM（ICLR 2025）、RealMLP（NeurIPS 2024） | pytabkit |
| 預訓練基礎模型 | TabPFN-3.5（2026/09） | tabpfn |
| | TabICLv2（ICML 2026） | tabicl |
| 公平性專用（RQ3） | FairGBM（ICLR 2023） | fairgbm |

所有模型包成統一的 scikit-learn 介面（`fit` / `predict_proba`），評估程式只寫一次。

### 6.4 比較設計（參考 TabArena）

- **預設參數組**：10 個模型皆用預設參數。
- **調參組**：傳統機器學習與深度學習使用相同預算（Optuna，每個模型 30 次試驗，在驗證月選最佳）；深度學習若全量太慢，改在 10 萬筆子樣本上調參並註明。基礎模型不調參。
- **學習曲線**：訓練資料量 1 千、5 千、1 萬、5 萬、10 萬、全量（分層抽樣，小資料量重複 3 次）；基礎模型跑到運算資源允許為止。
- **運算環境**：Kaggle Notebook（GPU）；記錄硬體與執行時間，作為「代價」的比較依據。

### 6.5 評估指標

| 指標 | 定義／說明 |
|---|---|
| **TPR@5%FPR**（主要） | 在誤判率小於 5% 的最大門檻下的偵測率（同原論文） |
| ROC-AUC、PR-AUC | 整體排序能力；PR-AUC 對不平衡資料更敏感 |
| Brier score | 預測機率的校準程度 |
| **FPR 比**（公平性主要） | min(FPR_年輕, FPR_年長) / max(FPR_年輕, FPR_年長)；≥ 0.8 視為可接受（參考值） |
| TPR 比 | 兩組偵測率的比值（equal opportunity） |
| 期望損失 | 見 6.7 節 |
| 執行時間 | 訓練與預測秒數、是否需要 GPU |

**統計嚴謹度**：測試集 bootstrap（1,000 次）計算 95% 信賴區間；模型間差異以配對 bootstrap 檢定。

**兩種門檻分開使用**：模型比較一律用 5% 誤判率門檻（才能和原論文對照）；決策章節才用 6.7 節的最佳門檻。

### 6.6 公平性分析

1. **量測**：年齡分組依官方程式碼採 `customer_age >= 50`；另以論文正文的 `> 50` 計算一次，量化差異。
2. **切點敏感度**：年齡切在 40、50、60 歲，各自計算 FPR 比，檢驗結論是否改變。
3. **處理方法**：
   - 後處理：依年齡組設定不同門檻（必做）。
   - 訓練中處理：FairGBM（應做）——只在訓練時使用年齡，上線預測不需要，爭議較小。
4. **取捨分析**：畫出公平–效能 Pareto 前緣；第五章討論「依年齡設不同門檻」可能被視為差別對待的爭議。

### 6.7 三級分流決策模型（本研究提出）

依模型分數 *s* 與兩個門檻 *t₁ < t₂* 分成三級：

| 等級 | 條件 | 處置 | 對應台灣制度 |
|---|---|---|---|
| 一 | *s* < *t₁* | 正常開戶 | 第一、二類數位存款帳戶 |
| 二 | *t₁* ≤ *s* < *t₂* | 先開限額帳戶＋加強驗證（視訊或電信認證） | 第三類（轉帳限額較低） |
| 三 | *s* ≥ *t₂* | 人工審核或臨櫃 | 臨櫃開戶 |

**期望損失**

```
Cost(t₁, t₂) =  Σ 詐騙∈第一級  L_full
              + Σ 詐騙∈第二級  (1 − v) · L_limited
              + Σ 詐騙∈第三級  (1 − r) · L_full
              + Σ 正常∈第二級  C_friction
              + Σ 正常∈第三級  C_attrition
              + N_第二級 · C_verify + N_第三級 · C_review
```

| 參數 | 意義 | 預設（假設，做敏感度分析） |
|---|---|---|
| L_full | 詐騙帳戶在完整功能下的預期損失 | 以第一類帳戶每月非約定轉帳上限 20 萬為基準 |
| L_limited | 詐騙帳戶在限額下的預期損失 | 以第三類每月上限 5 萬為基準（≈ L_full × 1/4） |
| v | 加強驗證抓出詐騙的機率 | 0.5（敏感度 0.3–0.8） |
| r | 人工審核抓出詐騙的機率 | 0.9（可改用 FiFAR 估計） |
| C_verify、C_review | 加強驗證、人工審核的單件成本 | 依假設設定 |
| C_friction、C_attrition | 好客戶被加強驗證、被轉人工時的體驗成本與流失成本 | 依假設設定 |

> 限額數字以三信商銀條款為例，各銀行不同，報告中標明為情境模擬。

**限制條件**

- 人力：每月第三級件數 ≤ 審核人力上限 *K*。
- 公平：被分到第二級或第三級的正常申請者，兩個年齡組的比例比值 ≥ 0.8。

**求解**：在驗證月以網格搜尋 (*t₁*, *t₂*)，在測試月評估；與「固定 5% 誤判率」的二元決策比較。敏感度分析：損失與成本比 1:10、1:50、1:100，以及不同的 *K*、*v*、*r*。

**學理基礎**：三支決策（three-way decision；Yao, 2010）。本研究的新意在於對應台灣數位帳戶制度，並加入公平與人力限制。

### 6.8 知識發現

- 對最佳 GBDT 計算 TreeSHAP：全域重要性（beeswarm）、依賴圖、個案解釋（waterfall）。
- 以深度 3–4 的決策樹模仿最佳模型，萃取人看得懂的規則，並回報與原模型的一致率（fidelity）。
- 製作「BAF 特徵 ↔ 台灣銀行可取得的對應資料」對照表。

### 6.9 小銀行困境（RQ6，延伸）

- 把訓練資料切給 5 家模擬銀行：規模不同（例如 40%、25%、15%、12%、8%），客群以 Dirichlet 分配製造差異（non-IID）。
- 比較：(a) 各銀行自己訓練 GBDT、(b) 各銀行用基礎模型、(c) 聯合學習（FedAvg，MLP）、(d) 資料集中訓練（上限參考）。
- 重點看最小銀行的 TPR@5%FPR 與 FPR 比，對照 2026/08 八家銀行聯合學習實例。

### 6.10 上線監控（支撐章節）

- 特徵漂移：訓練期 vs 測試各月的 PSI（Population Stability Index）。
- 每月追蹤效能與公平性指標；利用 Variant IV、V 觀察模式改變時的衰退。
- 撰寫 Model Card（用途、資料、指標、公平性、限制）。
- 法規對照表：AI 基本法七原則、金管會 AI 指引 ↔ 本專案的實驗與產出。

## 7. 實驗矩陣與優先順序

| 編號 | 實驗 | RQ | 優先 | 主要產出 |
|---|---|---|---|---|
| E0 | EDA、台灣開放資料趨勢 | 背景 | 必做 | 圖：詐騙率分月、年齡分布；台灣趨勢圖 |
| E1 | 重現：Base 100 組 LightGBM；變體各 20 組 | RQ1 | 必做（變體：應做） | 表 C、效能–公平散佈圖（重現原論文 Figure 1） |
| E2 | 10 個模型預設參數組 | RQ2 | 必做 | 模型總表、ROC／PR 曲線 |
| E3 | 調參組 | RQ2 | 必做 | 預設 vs 調參比較表 |
| E4 | 學習曲線 | RQ2、RQ6 | 必做 | 學習曲線圖、時間成本圖 |
| E5 | 公平性量測＋切點敏感度 | RQ1、RQ3 | 必做 | 各組誤判率圖、切點敏感度表 |
| E6 | 公平性處理（門檻、FairGBM） | RQ3 | 必做／應做 | Pareto 前緣圖 |
| E7 | 三級分流決策 | RQ4 | 必做 | 期望損失曲面圖、策略比較表、敏感度分析 |
| E8 | SHAP＋規則萃取 | RQ5 | 必做 | SHAP 圖、規則樹、風控規則表 |
| E9 | 上線監控、Model Card | 支撐 | 必做 | PSI 表、分月追蹤圖 |
| E10 | 小銀行聯合學習 | RQ6 | 應做（11/15 決定） | 各銀行比較圖 |
| E11 | Streamlit 風控門檻模擬器 | 展示 | 應做 | 公開 demo |
| — | LLM 審核助理、圖分析、深偽 e-KYC、即時串流 | — | 不做 | 列入未來研究 |

## 8. 報告大綱

報告以中文撰寫，Quarto 產出 PDF；封面、摘要採中英雙語。

| 章 | 內容 | 主要圖表 |
|---|---|---|
| 封面、摘要 | 中英文題目、摘要、關鍵字 | — |
| 一、緒論 | 1.1 背景與動機、1.2 研究目的、1.3 研究問題、**1.4 研究貢獻**、1.5 範圍與限制、1.6 研究流程 | 表 A（研究貢獻）、研究流程圖、台灣詐騙趨勢圖 |
| 二、文獻探討 | 2.1 詐騙偵測、2.2 不平衡與成本敏感學習、2.3 表格模型（GBDT → 深度學習 → 基礎模型）、2.4 演算法公平性、2.5 可解釋 AI 與三支決策、**2.6 研究缺口** | 表 B（相關研究比較） |
| 三、研究方法 | 3.1 資料、3.2 EDA、3.3 前處理與切分、3.4 模型、3.5 指標、3.6 三級分流模型、3.7 實驗設計與假說 | 欄位表、模型表、實驗總覽表 |
| 四、實驗結果 | 4.1 重現、4.2 三代模型、4.3 學習曲線、4.4 公平性、4.5 三級分流決策、4.6 知識發現、4.7 小銀行困境（延伸） | **表 C（原論文 vs 本研究）**、其餘依實驗矩陣 |
| 五、管理意涵 | 分級審核流程、風控規則、公平性稽核、上線監控、法規對照 | 法規對照表、BAF↔台灣資料對照表 |
| 六、結論 | 結論、限制、未來研究 | RQ → 結果 → 貢獻 總表 |
| 參考文獻 | APA 第 7 版 | — |
| 附錄 | 超參數、完整結果、程式碼連結、Model Card、生成式 AI 使用說明 | — |

**比較表的使用原則**：表 A、B、C 加上結論總表共 4 張，各放一處，不重複；表 B 只列真正讀過的文獻。

## 9. 專案架構與程式品質

```
catch-fraud-not-customers/
├── README.md              # 作品集首頁：問題、主要發現、代表圖、demo 連結、如何重現
├── pyproject.toml         # 套件版本、ruff、pytest 設定
├── Makefile               # make data / train / figures / report / app / all
├── configs/               # 所有實驗設定（參數不寫死在程式裡）
│   ├── base.yaml          # 路徑、隨機種子、資料切分
│   ├── models/            # 各模型超參數與搜尋空間
│   └── costs.yaml         # 三級分流的成本假設
├── data/                  # 不進 git（授權限制），由腳本下載
│   ├── raw/
│   └── processed/
├── src/frauddetect/       # 核心程式碼，可 import、可測試
│   ├── data/              # 下載、讀取、欄位驗證、時間切分
│   ├── models/            # 統一介面：10 個模型 + FairGBM
│   ├── evaluation/        # metrics、fairness、cost、stats（bootstrap）
│   ├── decision/          # 三級分流決策模型
│   ├── explain/           # SHAP、規則萃取
│   ├── federated/         # 小銀行聯合學習模擬
│   ├── monitoring/        # PSI、分月追蹤
│   ├── viz/               # 統一圖表風格
│   └── cli.py             # python -m frauddetect run --config ...
├── notebooks/             # 只做探索與敘事，邏輯呼叫 src/
├── results/               # 每次實驗的指標、預測、設定快照
├── reports/paper/         # Quarto 原始檔、references.bib、figures/、tables/
├── app/streamlit_app.py   # 風控門檻模擬器（只用彙總結果，不含原始資料）
├── tests/                 # pytest 單元測試
├── docs/                  # 本計畫書、Model Card
└── .github/workflows/ci.yml   # ruff + pytest
```

**程式品質原則**

1. 運算邏輯放 `src/`，notebook 只負責說故事。
2. 設定驅動：換模型、換資料變體只改 YAML。
3. 所有模型同一介面，評估程式只寫一次。
4. 可重現：固定隨機種子、鎖定套件版本、每筆結果存設定快照。
5. 圖表全由程式產生並存入 `reports/paper/figures/`，報告直接引用，數字不會貼錯。
6. 型別註記、docstring、ruff 檢查格式；pytest 驗證 TPR@5%FPR、FPR 比、期望損失等核心函式；GitHub Actions 自動執行。

**報告工具**：Quarto（Markdown 撰寫、圖表自動編號與交互引用、BibTeX + APA CSL、中文 PDF 使用 CJK 字型）。

## 10. 時程與檢查點

| 週 | 日期 | 程式 | 報告 |
|---|---|---|---|
| W1 | 10/24–11/1 | 計畫書、repo 架構與 CI、下載資料、E0 | 讀核心論文（BAF、TabPFN、FairGBM）；第一章 |
| W2 | 11/2–11/8 | E1 重現、E2 預設參數組 | 第二章 |
| W3 | 11/9–11/15 | E3 調參組、E4 學習曲線、E5 公平性量測 | 第三章 |
| 🚦 | **11/15** | **檢查點：決定 E10（RQ6）去留** | |
| W4 | 11/16–11/22 | E6、E7、E8、E9、（E10） | 第四章 |
| W5 | 11/23–11/30 | E11 demo、README | 第五、六章、摘要、校對；**11/28 前上傳** |

## 11. 風險與對策

| 風險 | 對策 |
|---|---|
| 範圍太大 | 優先順序分級；11/15 檢查點；變體實驗縮減 |
| 重現的 600 次 LightGBM 訓練耗時（樹數最高 1 萬棵） | Base 跑滿 100 組，變體各 20 組；資料存 Parquet |
| 深度學習與基礎模型跑全量吃 GPU | Kaggle Notebook；基礎模型以學習曲線為主；記錄硬體 |
| TabPFN-3.5 首次使用需在 Prior Labs 接受授權 | 事先申請；CPU 只能跑 5,000 筆，需用 GPU |
| FairGBM 與新版 Python 相容性 | 先行測試；不行則改用 Fairlearn 的方法 |
| 原論文只有圖、沒有精確數字 | 重現目標設為「趨勢一致＋關鍵發現（FPR 比約 0.3）」；差異如實記錄 |
| 成本參數是假設 | 敏感度分析，結論不依賴單一數字 |
| 台灣開放資料無法自動下載 | 改為手動下載並記錄下載日期 |

## 12. 授權、研究倫理與分工

**授權**

- BAF：CC BY-NC-ND 4.0 → 資料不上傳；demo 只用彙總結果（分數分布、曲線）。
- BAF 官方 notebook 檔頭標示版權限制 → 程式碼獨立重寫，只參考論文公開的設定（如超參數範圍）。
- TabPFN-3.5 權重：非商業授權（課堂與作品集可用，README 註明）。TabICLv2、pytabkit：授權寬鬆。

**研究倫理**

- BAF 為合成資料，不含可識別個人資訊。
- 台灣制度對應屬情境模擬，不宣稱可直接套用於真實銀行。
- 附錄撰寫「生成式 AI 使用說明」。

**分工**

- 程式、圖表、文獻整理：由 Claude 協助。
- 報告的分析與討論：作者以自己的話撰寫或改寫，並負責最終判斷。

## 13. 初步參考文獻

> **選文原則**：每一節都同時有「奠基文獻」與「近年文獻（2023–2026）」。
> 奠基文獻不能省——用到 XGBoost、SHAP 等方法時，學術慣例要引用原始論文；近年文獻用來證明研究問題仍是現在進行式。
> 目標：近年文獻佔一半以上。標 🆕 者為 2023–2026 年。
> 第二章撰寫時補齊完整書目資訊（卷期、頁碼、DOI），以 APA 第 7 版排版；標「待確認」者須查證出處後才能引用。

### 13.1 資料與基準

- 🆕 Alves, J. V., Leitão, D., Jesus, S., Sampaio, M. O. P., Saleiro, P., Figueiredo, M. A. T., & Bizarro, P. (2023). FiFAR: A fraud detection dataset for learning to defer. *ICAIF 2023 Workshop on Synthetic Data for AI in Finance*. arXiv:2312.13218.
- 🆕 Erickson, N., et al. (2025). TabArena: A living benchmark for machine learning on tabular data. *NeurIPS 2025 Datasets and Benchmarks Track*. arXiv:2506.16791.
- Jesus, S., Pombal, J., Alves, D., Cruz, A. F., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation. *NeurIPS 2022 Datasets and Benchmarks Track*.

### 13.2 詐騙與人頭帳戶偵測

- 🆕 Chen, Y., et al. (2025). Year-over-year developments in financial fraud detection via deep learning: A systematic literature review. arXiv:2502.00201.（回顧 2019–2024 年 57 篇研究）
- 🆕 Jeri-Alvarado, et al. (2025). Financial fraud detection in the banking sector using machine learning: An exhaustive systematic review. *Computación y Sistemas*.（待確認作者全名）
- 🆕 Zhang, et al. (2026). Detection, attribution, narration: An end-to-end pipeline for explainable money mule identification. arXiv:2607.17586.（LightGBM + TreeSHAP + LLM 說明，與本研究架構最接近）
- Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science, 17*(3), 235–255.（奠基）
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit card fraud detection: A realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning Systems, 29*(8), 3784–3797.

### 13.3 表格資料模型：GBDT → 深度學習 → 基礎模型

- 🆕 Jäger, B., et al. (2026). TabPFN-3.5: Technical report. arXiv:2609.17895.
- 🆕 Qu, J., et al. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. *ICML 2026*. arXiv:2602.11139.
- 🆕 Grinsztajn, L., et al. (2025). TabPFN-2.5: Advancing the state of the art in tabular foundation models. arXiv:2511.08667.
- 🆕 Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature*. https://doi.org/10.1038/s41586-024-08328-6
- 🆕 Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2025). TabM: Advancing tabular deep learning with parameter-efficient ensembling. *ICLR 2025*. arXiv:2410.24210.
- 🆕 Holzmüller, D., Grinsztajn, L., & Steinwart, I. (2024). Better by default: Strong pre-tuned MLPs and boosted trees on tabular data. *NeurIPS 2024*. arXiv:2407.04491.
- 🆕 van Breugel, B., & van der Schaar, M. (2024). Position: Why tabular foundation models should be a research priority. *ICML 2024*, PMLR 235, 48976–48993.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.（奠基）
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.（奠基）
- Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS 2018*.（奠基）

### 13.4 演算法公平性

- 🆕 Liu, Khalil, & Goel (2026). Causal pre-training under the fairness lens: An empirical study of TabPFN. *The Web Conference 2026*. arXiv:2601.17912.
- 🆕 Cruz, A. F., Belém, C., Bravo, J., Saleiro, P., & Bizarro, P. (2023). FairGBM: Gradient boosting with fairness constraints. *ICLR 2023*.
- 🆕 Hurlin, C., Pérignon, C., & Saurin, S. (2024). The fairness of credit scoring models. arXiv:2205.10200.（待確認正式刊登期刊）
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 2016*.（奠基）
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data, 5*(2), 153–163.（奠基）

### 13.5 決策、成本與可解釋

- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI 2001*.（奠基）
- Yao, Y. (2010). Three-way decisions with probabilistic rough sets. *Information Sciences, 180*(3), 341–353.（奠基，三支決策）
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.（奠基，SHAP）
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*, 56–67.

### 13.6 聯合學習

- 🆕 Roth, H. R., et al. (2026). Privacy-preserving federated fraud detection in payment transactions with NVIDIA FLARE. arXiv:2603.13617.（JPMorgan、RBC、BNY 等合著；模擬多家機構，FedAvg 明顯優於各自訓練）
- 🆕 Alhasawi, Almtrf, & Asad (2025). FedFraud: Trust-aware federated fraud detection under non-IID data. *Security and Privacy*.（待確認完整題名）
- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS 2017*.（奠基，FedAvg）

### 13.7 工具

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.

### 13.8 台灣法規、產業與背景（皆為 2024–2026）

- 🆕 人工智慧基本法（2026 年 1 月 14 日公布施行）。
- 🆕 金融監督管理委員會（2024）。金融業運用人工智慧（AI）指引。
- 🆕 內政部警政署 165 打詐儀表板；政府資料開放平臺「警政署電信詐欺犯罪資料」。
- 🆕 新聞：2025 自然人憑證開戶事件（公視）、2026/08 八銀行 AI 聯合學習（中央社）、銀行 AI 誤鎖爭議（iThome）。
- 🆕 國際實務：印度央行 MuleHunter.ai（AI 偵測人頭帳戶）。

**統計**：學術文獻 32 篇，其中 2023–2026 年 17 篇（約 53%）；另加台灣法規與產業資料 5 項（皆為 2024–2026）。
