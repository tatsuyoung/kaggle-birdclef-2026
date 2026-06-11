# BirdCLEF 2026

## コンペ概要

BirdCLEF 2026 は、野鳥の鳴き声から種を識別する音声分類コンペティションです。

2026 年版では、通常の学習データに加えて Soundscape (SS) データ が提供されました。SS データは複数の鳥の鳴き声や環境音が混在する長時間の録音であり、実環境に近い状況での識別性能が求められる点が特徴です。そのため、ノイズや重複した鳴き声に対して頑健なモデル設計や学習戦略が重要となりました。

コンペページ: https://www.kaggle.com/competitions/birdclef-2026

---

## 結果

- 最終順位: 2899 / 4091

順位自体は高くありませんでしたが、公開 Notebook をそのまま利用するのではなく、自分なりのアプローチを多く取り入れて試行錯誤した思い入れのあるコンペです。

---

## 実装内容

### 1. Fold 作成
Notebook: bird-2026-fold-save.ipynb

- filename を Group ID として使用し、同一音源が Train / Validation に分割されないように設定
- MultilabelStratifiedKFold により 5 Fold を作成
- マルチラベルの分布を保ちながら学習データを分割

---

### 2. 音声データ前処理
Notebook:
- bird-2026-fold-audio-save-1.ipynb
- bird-2026-fold-audio-save-ss.ipynb

- 音声データを NumPy 配列として保存し、高速に読み込める形式へ変換
- float32 に統一してメモリ効率を改善
- 通常データと Soundscape (SS) データの両方を前処理

---

### 3. EfficientNet + LSE Head
Notebook: bird-2026-pseudo2-training-fold-1.ipynb

- Backbone: EfficientNet-B0
- Mel Spectrogram を入力特徴量として使用
- TOP_DB = 60
- Log-Sum-Exp (LSE) Head を採用し、時間方向のフレーム単位の予測をクリップ単位の予測へ集約

---

### 4. GeM + SED モデル
Notebook: bird-2026-pseudo2-gemf-b0-training-fold-1.ipynb

- Backbone: tf_efficientnet_b0.ns_jft_in1k
- TOP_DB = 80
- Sound Event Detection (SED) Head を導入
- Global Average Pooling の代わりに GeM (Generalized Mean) Pooling を採用

#### 共通
- Pseudo Label を利用した学習を実施
- Perch v2 Embeddings を教師とした知識蒸留 (Knowledge Distillation) を実施

---

## モデル保存の工夫

モデル保存時には ROC AUC のみではなく、以下の独自スコアを利用しました。

text score_total = 0.7 * roc_auc_rank + 0.3 * roc_auc 

CV と Leaderboard の乖離を小さくする目的で Rank ベースの評価と ROC AUC を組み合わせました。

一方で、振り返ると本コンペの評価指標は ROC AUC であったため、最終的には ROC AUC のみを基準にモデル選択した方が良かったと考えています。

---

## 使用した主な技術

- EfficientNet-B0
- tf_efficientnet_b0.ns_jft_in1k
- Mel Spectrogram
- Log-Sum-Exp (LSE) Head
- Sound Event Detection (SED) Head
- GeM (Generalized Mean) Pooling
- Perch v2 Embeddings を用いた Knowledge Distillation
- MultilabelStratifiedKFold
- Pseudo Label
- Soundscape (SS) データの活用

---

## 振り返り

このコンペでは、公開解法をそのまま再現するのではなく、自分で仮説を立てて検証を繰り返しました。

特に GeM Pooling、SED Head、Perch v2 を用いた知識蒸留、Pseudo Label の活用、独自のモデル選択指標 などを試行し、音声分類モデルの設計や評価方法について多くの知見を得ることができました。
