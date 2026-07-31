# Kaggle Health Condition Classifier

> A LightGBM pipeline for a 3-class health-risk classification competition (Kaggle Playground Series S6E7) — public leaderboard score **0.94960**. Key engineering ideas: treating missingness as signal (MNAR), context-relative deviation features, and post-hoc probability recalibration for class imbalance. See below for the full Chinese write-up.

## 專案簡介

這是一個Kaggle Playground Series（`playground-series-s6e7`）健康狀況三分類競賽的專案，其將受測者分類為 `at-risk` / `fit` / `unhealthy`。Public leaderboard最佳成績為 **0.94960**。
這個repo只是從原始的所有實驗中整理出來的乾淨版本，只保留最終有效的pipeline、模型與腳本，實驗過程中的探索性測試僅包含部分內容。

## 專案結構

```
Kaggle_Health_Portfolio/
├── config/
│   └── best_lgb_params.json     # Optuna 5-fold CV 搜尋出的最佳 LightGBM 超參數
├── src/
│   ├── __init__.py
│   ├── features.py              # 特徵工程 + GroupSleepDiffTransformer
│   └── pipeline.py              # ColumnTransformer 前處理 Pipeline
├── notebooks/
│   ├── model_experiments.ipynb  # 最佳模型的完整實驗紀錄
│   ├── model_XGB_experiments.ipynb # XGB BaseLine model 
│   └── model_XGB_experiments02.ipynb # XGB特徵工程實驗
├── models/
│   └── health_lgb_mod3.pkl      # 訓練好的模型（由 main_train.py 產出）
├── main_train.py                # 一鍵訓練腳本
├── main_inference.py            # 一鍵推論腳本
├── requirements.txt
└── README.md
```

`data/` 目錄不包含在 repo 中（見下方「取得資料」）。

## 建模重點
1. **模型選擇**:本專案最初採用 XGBoost，但經過多次實驗後發現LightGBM不僅模型效能較好，也能大幅縮短訓練與預測時間，因此改採LightGBM。推測原因在於：LightGBM預設以直方圖（histogram-based）演算法搭配leaf-wise（best-first，依最大增益分裂）生長樹，相較XGBoost預設的level-wise（depth-wise）生長方式，在本專案約69萬筆的訓練資下運算更快、能更快收斂。在葉節點數固定的前提下，leaf-wise已被證實能達到比level-wise更低的訓練損失，雖在資料過小的情況容易過擬和（見文末參考文獻 Shi, 2007；LightGBM 官方文件）。

> 需要說明的是，「leaf-wise 更容易捕捉到 `stress_level`／`physical_activity_level` 等高度集中特徵的交互作用、因此準確度較好」這個推論，是筆者根據 leaf-wise演算法設計（優先分裂當下損失下降最多的節點，理論上能把更多分裂預算集中在高資訊量特徵路徑上）所做的個人推測，並非文獻中已證實的結論，未來可透過固定超參數、僅切換 `grow_policy`（XGBoost `lossguide` vs. `depthwise`）做對照實驗來驗證。

2. **缺失值訊號**：特徵重要性分析發現，`sleep_duration`、`stress_level`／`physical_activity_level` 是否為缺失值，其重要性甚至超越BMI、心率等生理指標本身 —— 代表受測者「有沒有填寫」這件事本身就帶有風險行為訊號，而非隨機遺漏。因此在 `engineered_features()` 中明確地把缺失計數與交互情境編碼成特徵。

3. **相對偏離度（`GroupSleepDiffTransformer`）**：單看睡眠時數的絕對值無法反映真實生理負擔 —— 同樣睡6小時，在「高壓 + 久坐」狀況下可能是嚴重不足，在「低壓 + 規律運動」下卻可能相對充裕。此特徵計算觀察值「壓力 x 活動」與相同群體平均值的偏離度（此為相對脆弱程度，非絕對值）。

> **不同特徵的實驗結果**：只對 `sleep_duration` 做偏離度轉換是最佳設定（LB 0.94960）；額外把 `bmi`／`step_count`／`exercise_duration` 也做同樣轉換，LB分數反而略降至 0.94937（雖在本地測試，整體f1-score是有略微提升），因此預設保留單欄位版本。

4. **類別不平衡處理**：訓練時對 sample weight 使用 `sqrt(balanced_weight)`（而非原始預設的倒數），避免權重過度放大稀有類別導致雜訊被過度學習。實驗對照：改用完整 balanced weight（純倒數、未開根號）時，準確率與召回率皆較差，整體 F1-score 也不如 `sqrt(balanced_weight)` 版本（實驗過程可參考 `notebooks/model_XGB_experiments.ipynb`）。此作法與過去文獻的發現一致（Bakirarar & Elhan, 2023，見文末參考文獻）。

5. **決策邊界調整**：訓練時的 `sqrt(balanced_weight)` 改變的是樹的分裂與葉節點估計（模型學到的 P(y|x) 本身）；推論時的 `1/sqrt(prior)` 則是對輸出機率做一個與樣本特徵無關、每個類別固定倍率的線性縮放。兩者作用層次不同，疊加起來會放大同一個方向的效果——這也是為什麼組合起來比單獨用任何一個效果都更明顯。

> 這個組合能大幅提升 LB 分數，關鍵在於這個競賽的評分指標看起來是偏向recall導向（尤其重視少數類別的recall），而不是log loss。也就是說，這個做法本質上是**刻意犧牲多數類別的precision，去換取少數類別的recall**，是針對評分指標刻意做的決策邊界調整，不是機率校準——`predict_proba` 乘完權重後也不再是有校準意義的真實機率。若指標換成log loss，這個做法很可能會讓表現變差，需要另外處理。

## 模型表現

以下為訓練時使用 `sqrt(balanced_weight)` 加權後的分類報告（`classification_report`），從 `train.csv` 切出 80% 訓練 / 20% 測試集（`train_test_split`, `stratify=y`），共 690,088 筆。類別代碼：`0 = at-risk`、`1 = fit`、`2 = unhealthy`。此處為模型原始 `predict()` 輸出，**尚未套用**推論階段的 `1/sqrt(prior)` 機率校正。

**訓練集（552,070 筆）**

| 類別 | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 (at-risk) | 0.9940 | 0.9728 | 0.9833 | 474,049 |
| 1 (fit) | 0.8480 | 0.9644 | 0.9025 | 31,842 |
| 2 (unhealthy) | 0.8563 | 0.9620 | 0.9060 | 46,179 |
| **accuracy** | | | **0.9715** | 552,070 |
| macro avg | 0.8994 | 0.9664 | 0.9306 | 552,070 |
| weighted avg | 0.9740 | 0.9715 | 0.9722 | 552,070 |

**測試集（138,018 筆，held-out，模型未見過）**

| 類別 | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 (at-risk) | 0.9866 | 0.9652 | 0.9758 | 118,512 |
| 1 (fit) | 0.8146 | 0.9173 | 0.8629 | 7,961 |
| 2 (unhealthy) | 0.8076 | 0.9172 | 0.8589 | 11,545 |
| **accuracy** | | | **0.9584** | 138,018 |
| macro avg | 0.8696 | 0.9332 | 0.8992 | 138,018 |
| weighted avg | 0.9617 | 0.9584 | 0.9595 | 138,018 |

測試集的 macro recall（0.9332）明顯高於 macro precision（0.8696），正好印證前面第 4、5 點提到的設計取捨：`sqrt(balanced_weight)` 訓練讓少數類別（`fit`、`unhealthy`）的 recall 都拉到 0.91 以上，代價是這兩個類別的 precision 相對較低（誤將多數類別的樣本判成少數類別的情況變多）；訓練集與測試集的數字落差不大，代表模型沒有明顯過擬合。

## 環境安裝

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## 取得資料

請至 Kaggle Playground Series S6E7 競賽頁面下載後，放入專案根目錄的 `data/` 資料夾：

```
data/
├── train.csv
└── test.csv
```

## 使用方式

```bash
# 1. 訓練模型
python main_train.py

# 2. 推論
python main_inference.py
```

## Notebooks

- `notebooks/model_XGB_experiments.ipynb`：最初以 XGBoost 建立 baseline 的實驗過程，藉此找出 sqrt 加權策略與關鍵特徵，作為後續建模方向的依據。
- `notebooks/model_XGB_experiments02.ipynb`：特徵工程實驗之一。此分析原本額外納入「缺失模式」（`sleep_duration`、`stress_level`、`physical_activity_level` 缺失型態組合）類別變數，但後續發現其貢獻度較低，因此予以剔除——保留各變數各自的缺失狀態即可涵蓋大部分訊息，不需要額外建立一個綜合的缺失模式欄位。
- `notebooks/model_experiments.ipynb`：產出最終最佳模型（LightGBM）的完整流程，包含資料處理定義、Optuna 5-fold CV超參數搜尋（20 trials，以 multi-class log loss 為目標）、最終模型訓練與存檔。

## 參考文獻

Bakirarar, B., & Elhan, A. H. (2023). Class weighting technique to deal with imbalanced class problem in machine learning: Methodological research. *Türkiye Klinikleri Biyoistatistik*, 15(1), 19–29. https://doi.org/10.5336/biostatic.2022-93961

Shi, H. (2007). *Best-first decision tree learning* [Master's thesis, The University of Waikato]. Cited in: LightGBM documentation, [Leaf-wise (Best-first) Tree Growth](https://lightgbm.readthedocs.io/en/latest/Features.html).
