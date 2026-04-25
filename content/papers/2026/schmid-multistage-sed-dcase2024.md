---
title: "Multi-Iteration Multi-Stage Fine-Tuning of Transformers for Sound Event Detection with Heterogeneous Datasets"
authors: "Florian Schmid, Paul Primus, Tobias Morocutti, Jonathan Greif, Gerhard Widmer"
venue: "DCASE 2024 Workshop / arXiv:2407.12997"
year: 2024
url: "https://arxiv.org/abs/2407.12997"
code: "https://github.com/CPJKU/cpjku_dcase24"
read_date: 2026-04-26
status: read
tags:
  - audio
  - sed
  - sound-event-detection
  - pseudo-label
  - semi-supervised
---

## TL;DR

> DCASE 2024 Challenge Task 4 (Sound Event Detection) の **1 位解法**。3 つの大規模事前学習 Audio Spectrogram Transformer (ATST, fPaSST, BEATs) を **2 段階学習 + 反復疑似ラベル蒸留** で fine-tune し、単一モデルで DESED 上 **PSDS1 = 0.692** (SOTA) を達成。BirdCLEF のような弱ラベル音声分類にも応用可能な汎用的フレームワーク。

## 背景・問題設定

- Sound Event Detection (SED) は音声中のイベントの **発生区間と種類** を同時に検出するタスク。単なる分類 (clip-level) より難しい。
- DCASE Task 4 は **異種混合データ** を扱う: DESED (強ラベル + 弱ラベル + ラベルなし) + MAESTRO (マルチクラス strong-label)。
- 事前学習 Transformer は AudioSet で clip-level 学習されており、**frame-level SED に直接適用できない**。CRNN 等の context network が必要。
- 従来は Transformer を frozen で使い CRNN だけ学習 → 性能の天井が低い。

## 手法

### 2 段階学習 (Two-Stage Training)

**Stage 1 (Frozen Transformer)**:
- 大規模事前学習 Transformer (ATST / fPaSST / BEATs) を frozen。
- CRNN (context network) のみを学習。
- 目的: CRNN が Transformer 特徴を frame-level 処理に適応。

**Stage 2 (Joint Fine-tuning)**:
- Transformer + CRNN を同時に fine-tune。
- **強い自己教師あり損失** (heavily weighted SSL losses) を併用し、Transformer の catastrophic forgetting を防止。

### 反復疑似ラベル蒸留 (Multi-Iteration Pseudo-Label Distillation)

1. Iteration 1 完了後、3 つの fine-tuned Transformer のアンサンブルで **全学習データに強い疑似ラベル** を生成 (frame-level logits の平均)。
2. Iteration 2 で Stage 1 から再学習。**疑似ラベルとの BCE 蒸留損失** を追加。
   - MSE より BCE が優れることを実験で確認。
   - 疑似ラベルは soft target (hard ではない)。
3. Iteration 2, Stage 2 で再度 joint fine-tuning。

### cSEBBs 後処理

- Class-Specific Event Bounding Boxes: イベント境界の検出を改善する後処理手法。
- Median filter よりも大幅に改善 (ATST: 0.617 → **0.692** PSDS1)。

## 実験

### PSDS1 スコアの推移 (DESED Public Eval)

| Stage | ATST | fPaSST | BEATs | Ensemble |
|---|---|---|---|---|
| I1.S1 (frozen) | 0.493 | 0.502 | 0.509 | — |
| I1.S2 (fine-tune) | 0.520 | 0.514 | 0.539 | 0.569 |
| I2.S1 (+pseudo) | 0.536 | 0.526 | 0.537 | — |
| I2.S2 (+pseudo, fine-tune) | 0.548 | 0.539 | 0.557 | — |

### 最終結果 (cSEBBs 後処理後)

| Model | PSDS1 |
|---|---|
| ATST + cSEBBs | **0.692** (SOTA) |
| ATST + median filter | 0.617 |
| fPaSST + cSEBBs | 0.601 |
| BEATs + cSEBBs | 0.622 |
| Previous SOTA | 0.686 |

### mpAUC (MAESTRO)

| Model | I2.S2 |
|---|---|
| ATST | 0.750 |
| fPaSST | 0.719 |
| BEATs | 0.729 |

### Ablation (ATST I2.S1 基準)

| 変更 | mpAUC | PSDS1 |
|---|---|---|
| Baseline | 0.741 | 0.536 |
| DESED only | 0.724 (-0.017) | — |
| MAESTRO only | — | 0.531 (-0.005) |
| Hard pseudo-labels | 0.706 (-0.035) | 0.538 |
| Pseudo all classes | 0.717 (-0.024) | 0.534 |

**Key insight**: Soft pseudo-labels >>> hard pseudo-labels (mpAUC -3.5%)。

### 学習ハイパーパラメータ

| Parameter | Stage 1 | Stage 2 (I2) |
|---|---|---|
| CNN LR | 1e-3 | 1e-5 ~ 5e-5 |
| RNN LR | 1e-3 | 1e-4 ~ 5e-4 |
| Transformer LR | — (frozen) | 1e-4 |
| Weight decay | 1e-2 | 1e-3 |

### バッチ構成 (5 データソース)

| Source | Stage 1 | Stage 2 |
|---|---|---|
| MAESTRO strong | 12 | 56 |
| DESED real strong | 10 | 40 |
| DESED synthetic strong | 10 | 40 |
| DESED weakly annotated | 20 | 72 |
| DESED unlabeled | 20 | 72 |

## 強み

- **体系的で再現性が高い**: 2 段階 × 2 反復の構造が明確。コードも公開。
- **Soft pseudo-label の効果が定量的に示されている**: hard vs soft で mpAUC -3.5% の差。
- **ATST の圧倒的優位**: BEATs, fPaSST を個別に上回り、cSEBBs 後処理で更に +0.075 PSDS1。
- **異種混合データの活用**: 弱ラベル + 強ラベル + ラベルなしの 3 種を統合的に活用する実用的フレームワーク。

## 弱み・未解決の問い

- **cSEBBs 後処理への依存度が高い**: 0.617 → 0.692 の +0.075 は後処理による。モデル単体の性能はそこまで突出していない。
- **Iteration 3 以降の効果は未検証**: BirdCLEF 1 位 (4 反復) と比較して反復数が少ない。
- **DESED 特化**: 10 クラスの環境音 SED データセットで検証。234 種の生物音響への適用は直接確認されていない。
- **計算コスト**: 3 モデル × 2 段階 × 2 反復 = 12 回の学習。

## 関連研究とのつながり

- 前身: ATST-Frame (Chen et al., 2022) — frame-level embedding 生成用の SSL モデル。
- 同時期: MAT-SED (Cai et al., Interspeech 2024) — 純粋 Transformer ベース SED。PSDS1 = 0.587 / PSDS2 = 0.896。
- 応用: [[papers/2026/birdclef2025-noisy-student-1st-place]] — BirdCLEF 2025 1 位も同じ反復疑似ラベル戦略。

## 自分の研究・実装への示唆

**BirdCLEF+ 2026** への示唆:

1. **Soft pseudo-label の採用**
   - 現在の疑似ラベル戦略が hard threshold ベースなら、soft target (logits 平均) に切り替えるだけで改善が見込める。
   - 本論文で hard → soft で mpAUC +3.5% の差を実証。
2. **2 段階学習の適用**
   - Perch (frozen) + 小規模 CRNN → joint fine-tune の 2 段階が有効。Stage 1 で CRNN が安定してから Perch を触る。
   - NFNet でも同様: frozen backbone + SED ヘッド → joint fine-tune。
3. **反復蒸留の導入**
   - NFNet + Perch のアンサンブルで soundscape に疑似ラベル → 各モデルを蒸留損失で再学習 → 再度アンサンブル。
   - BirdCLEF 1 位 (4 反復) + 本論文 (2 反復) の知見を合わせて 2〜3 反復を実施。
4. **ATST の検討**
   - Perch / NFNet の代替として ATST を frozen 特徴量に使う選択肢。frame-level embedding が SED に適している。
   - ただし CPU 推論速度が懸念。モデルサイズ確認が必要。
5. **cSEBBs 相当の後処理**
   - 現在の threshold-only 推論に対し、イベント境界の後処理を加えることで soundscape の偽陽性を削減できる可能性。
