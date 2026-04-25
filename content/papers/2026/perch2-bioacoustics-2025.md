---
title: "Perch 2.0: The Bittern Lesson for Bioacoustics"
authors: "Tom Denton, Lucile Dupin, Bart van Merriënboer, et al. (Google DeepMind & Google Research)"
venue: "arXiv:2508.04665"
year: 2025
url: "https://arxiv.org/abs/2508.04665"
code: ""
read_date: 2026-04-26
status: read
tags:
  - audio
  - bird-sound
  - representation-learning
  - semi-supervised
  - pseudo-label
---

## TL;DR

> Google DeepMind が開発した生物音響汎用モデル Perch の第 2 世代。鳥類専用から **14,795 クラス (14,597 種 + 198 環境音)** のマルチ分類群モデルへ拡張。**自己蒸留 + プロトタイプ学習 + ソース予測** の 3 目的で EfficientNet-B3 を学習し、BirdSet AUROC = **0.908**, BEANS Accuracy = **0.838** で SOTA。海洋生物の少数ショット転移でも専門モデルを凌駕。

## 背景・問題設定

- Perch 1.0 (EfficientNet-B1, 7.8M params) は鳥類専用で、他の分類群 (両生類、哺乳類、昆虫) への汎化が不十分。
- 生物音響分野では「種分類が最良の事前学習タスク」という仮説があるが、マルチ分類群スケールで検証されていなかった。
- ラベル品質の問題: Xeno-Canto, iNaturalist はユーザー投稿データでノイズが多い。

## 手法

### アーキテクチャ

- **バックボーン**: EfficientNet-B3 (12M parameters)
- **入力**: 32 kHz モノ → 5 秒セグメント (160,000 samples) → Log mel-spectrogram (128 bins, 500 frames, 10ms hop, 20ms window, 60 Hz〜16 kHz)
- **空間埋め込み**: (5, 3, 1536) → **平均埋め込み**: 1536 次元

### 3 つの学習目的

1. **線形分類器** (14,795 クラス): Cross-entropy + softmax。標準的な種分類。
2. **プロトタイプ学習分類器**: 各クラスに 4 つのプロトタイプベクトルを学習。**自己蒸留**: プロトタイプ予測を soft target として線形分類器を訓練。
3. **ソース予測ヘッド**: 低ランク射影 (rank 512) で各サンプルを「自身のクラス」として分類。instance-level の表現学習。

### 学習スケジュール

- **Phase 1** (300k steps): 線形分類器 + ソース予測 + プロトタイプ
- **Phase 2** (400k steps): Phase 1 + **自己蒸留** (プロトタイプ → 線形分類器への蒸留を追加)

### Generalized Mixup

- 複数の音声窓を混合して合成信号を生成。
- 混合数: $N \sim \text{BetaBin}(n, \alpha, \beta) + 1$
- 重み: 対称 Dirichlet 分布
- Multi-hot ターゲット: 音量に関わらず全種を認識。

## 実験

### 学習データ

| Source | Records | Notes |
|---|---|---|
| Xeno-Canto | 896,255 | Birds: 860k, Amphibia: 2.3k, Insects: 32k, Mammals: 1.3k |
| iNaturalist | 571,698 | Birds: 480k, Amphibia: 51k, Insects: 31k, Mammals: 9k |
| Tierstimmenarchiv | 33,859 | Birds: 27k, Others: 7k |
| FSD50K | 40,966 | General sound events |
| **Total** | **1,542,778** | **14,795 classes** |

### 主要結果

| Model | BirdSet AUROC | BirdSet cmAP | BEANS Acc |
|---|---|---|---|
| Perch 1.0 (B1) | 0.839 | — | 0.809 |
| Audio ProtoPNet-5 | 0.896 | — | — |
| BirdMAE-L | 0.886 | — | — |
| **Perch 2.0 (B3)** | **0.908** | **0.431** | **0.838** |

### 海洋転移学習 (Few-shot, k=16)

| Dataset | ROC-AUC |
|---|---|
| DCLDE 2026 species | 0.977 |
| NOAA PIPAN | 0.924 |
| ReefSet | 0.981 |

海洋訓練データはほぼゼロにも関わらず専門モデルを凌駕。「種分類は生物音響の頑健な事前学習タスク」を実証。

## 強み

- **スケール**: 1.5M 録音、15k クラスという前例のないスケール。これだけで他モデルを圧倒。
- **自己蒸留の有効性**: プロトタイプ → 線形分類器への蒸留で、単なる分類よりも表現が豊かに。
- **Generalized Mixup**: 従来の 2 サンプル mixup を N サンプルに拡張。multi-label 設定に自然に適合。
- **Zero-shot marine**: 海洋データなしで海洋 SOTA は、事前学習の汎化力を強力に示す。

## 弱み・未解決の問い

- **EfficientNet-B3 のアーキテクチャ選択**: ViT 系 (BirdMAE, Audio-MAE) との直接比較が不足。CNN vs Transformer の議論が残る。
- **計算コスト**: 700k steps の学習は大規模 GPU が必要。再現性に懸念。
- **Fine-tuning 性能の欠如**: BirdSet での fine-tuning 結果が提示されていない。frozen embedding のみの評価では不公平。
- **BirdCLEF 競技環境での検証なし**: soundscape ドメインシフト下での実力は未検証。

## 関連研究とのつながり

- 前身: Perch 1.0 (Ghani et al., 2023) — EfficientNet-B1, 鳥類専用。
- 同時期: [[papers/2026/birdmae-2025]] (Bird-MAE) — SSL + MAE アプローチ。BirdSet AUROC 0.886 で Perch 2.0 に次ぐ。
- 応用: BirdCLEF+ 2025/2026 でのベースラインモデルとして広く使用。

## 自分の研究・実装への示唆

**BirdCLEF+ 2026** への直接的なアクション:

1. **Perch 2.0 への移行を検討**
   - 現在使用中の Perch (v1) から v2 への移行で BirdSet AUROC +6.9%。soundscape でも同等以上の改善が期待。
   - ただし公開モデルの有無を確認する必要あり。
2. **自己蒸留の導入**
   - NFNet の学習に自己蒸留を追加する低コスト改善。プロトタイプ分類器を補助ヘッドとして付けるだけ。
3. **Generalized Mixup**
   - 現在の mixup を N サンプル版に拡張。multi-label soundscape データに自然に適合。
   - `BetaBinomial(n, alpha, beta)` + Dirichlet 重みの実装は数十行。
4. **ソース予測**
   - instance-level 対照学習の代替として面白い。各サンプルを「自身のクラス」として分類するアイデアは SimCLR/BYOL より実装が簡単。
5. **14,795 クラスの embedding を活用**
   - Perch 2.0 embedding を frozen feature として使い、小規模な SED ヘッドを soundscape データで fine-tune するハイブリッドが有望。
