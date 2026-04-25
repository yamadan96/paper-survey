---
title: "Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation (BirdCLEF+ 2025, 2nd Place)"
authors: "Volodymyr Sydorskyy et al."
venue: "Kaggle BirdCLEF+ 2025 Competition (2nd Place) / GitHub"
year: 2025
url: "https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place"
code: "https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place"
read_date: 2026-04-26
status: read
tags:
  - audio
  - bird-sound
  - pseudo-label
  - semi-supervised
  - inference-efficiency
  - onnx
---

## TL;DR

> BirdCLEF+ 2025 の 2 位解法 (public 0.925 / private **0.928**)。**eca_nfnet_l0 + EfficientNetV2-S** の 2 モデルアンサンブルを FocalBCE + 半教師あり蒸留で学習し、**ONNX → fp16 → OpenVINO** 変換で CPU 推論を高速化。ドメインシフト対策として Xeno-Canto + iNaturalist + CSA からの大規模事前学習と soundscape 疑似ラベル蒸留を組み合わせた。

## 背景・問題設定

- BirdCLEF+ 2025: 234 種、CPU のみ 90 分制約。
- ドメインシフト: 学習データ (クリーン単一種録音) → テスト (soundscape, 多種混在, ノイズ)。
- 推論効率: GPU 不可のため、モデルサイズと推論速度の最適化が必須。

## 手法

### モデルアーキテクチャ

| Model | Parameters | Pre-training |
|---|---|---|
| eca_nfnet_l0 | ~24M | ImageNet → Xeno-Canto + iNaturalist + CSA |
| tf_efficientnetv2_s_in21k | ~21M | ImageNet-21k → Xeno-Canto + iNaturalist + CSA |

- **入力**: 5 秒クリップ → mel-spectrogram
- **損失**: FocalBCELoss (クラス不均衡対策)
- **最適化**: AdamW (1e-4) / RAdam (1e-3), cosine annealing (min LR 1e-6)
- **エポック**: 50
- **バッチサイズ**: 64

### 大規模事前学習

- Xeno-Canto, iNaturalist, CSA (広域種分類データ) で事前学習。
- "smaller" と "bigger" の 2 段階事前学習データセットを使い分け。

### 半教師あり蒸留 (Pseudo-Label Distillation on Soundscapes)

- 教師モデルのアンサンブルで soundscape に疑似ラベルを生成。
- フィルタリング閾値:
  - F2 probability threshold: 0.5
  - Model confidence threshold: 0.1
  - Minimum instances: 4
  - Minimum probability: 0.4
- 疑似ラベル付き soundscape + ラベル付きデータで再学習。

### クラスバランシング

- **SqrtBalancing**: クラス頻度の平方根で重み付け (不均衡を緩和しつつ過補正を防ぐ)。
- **EqualBalancing**: 均等サンプリング (レア種を過剰にサンプル)。
- 設定ごとに使い分け。

### 推論最適化パイプライン

```
PyTorch → ONNX → fp16 quantization → OpenVINO
```

- CPU 推論で大幅高速化。90 分制約をクリア。
- 5-fold cross-validation のトップ 5 fold の予測を平均。

## 実験

- **Public AUC**: 0.925
- **Private AUC**: 0.928
- 1 位 (0.930) との差はわずか 0.002。

### 1 位との差分

| Aspect | 1st (Babych) | 2nd (Sydorskyy) |
|---|---|---|
| Iteration count | 4+ rounds | 1-2 rounds |
| Model diversity | 6+ architectures | 2 architectures |
| Power Scaling | Yes | No |
| Inference format | ONNX | ONNX → OpenVINO |
| Score | 0.930 | 0.928 |

## 強み

- **再現性が高い**: コード全公開。学習スクリプト、設定ファイル、推論パイプラインすべて含む。
- **OpenVINO 変換**: ONNX だけでなく OpenVINO まで変換することで CPU 推論をさらに最適化。BirdCLEF 2026 の制約下で実用的。
- **FocalBCE**: 通常の BCE よりクラス不均衡に強い。alpha/gamma の調整で minority class の学習を強化。
- **シンプルなアンサンブル**: 2 モデルだけで 0.928 は驚異的。モデル数を増やさずに高性能。

## 弱み・未解決の問い

- **疑似ラベルの反復回数が少ない**: 1 位が 4 反復で +0.032 改善しているのに対し、1-2 反復で止めている。
- **Power Scaling 未使用**: 1 位の key technique が欠如。
- **SED モデル不使用**: clip-level 分類のみ。frame-level の SED は試みていない。
- **Augmentation が BasicAug のみ**: MixUp, SpecAugment 等の詳細が不明。

## 関連研究とのつながり

- 同著者: VSydorskyy/BirdCLEF_2023_1st_place (2023 年 1 位)。同じチームが継続的に上位。
- 1 位: [[papers/2026/birdclef2025-noisy-student-1st-place]] — 反復疑似ラベル + Power Scaling。
- 5 位: BirdCLEF-2025-5th-place-solution (myso1987) — 別のアプローチ。

## 自分の研究・実装への示唆

**BirdCLEF+ 2026** への直接的アクション:

1. **eca_nfnet_l0 の採用**
   - 現在の NFNet 設定を eca_nfnet_l0 (Efficient Channel Attention + NFNet) に変更。2 位で実証済み。
   - `timm` ライブラリで `timm.create_model('eca_nfnet_l0', pretrained=True)` ですぐ利用可能。
2. **ONNX → OpenVINO パイプライン**
   - 現在は ONNX のみだが、OpenVINO 変換で CPU 推論がさらに高速化。
   - `from openvino.tools import mo; mo.convert_model(onnx_path)` で変換。
   - ただし Kaggle 環境に OpenVINO がインストールされているか要確認。
3. **FocalBCE の導入**
   - 現在の損失関数が BCE なら FocalBCE に切り替え。$\gamma = 2.0, \alpha = 0.25$ が定番。
   - BirdCLEF の極端なクラス不均衡 (サンプル 5 件 vs 1000 件) に対して有効。
4. **SqrtBalancing**
   - クラスサンプリング重みを $\sqrt{N_c}$ に比例させる。完全均等 vs 頻度比例の中間。実装 1 行。
5. **疑似ラベルフィルタリング閾値の参考値**
   - F2 prob = 0.5, confidence = 0.1, min instances = 4, min prob = 0.4 は具体的で再現可能。
   - まずこの値で試し、validation で調整。
