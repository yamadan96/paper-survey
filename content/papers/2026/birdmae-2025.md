---
title: "Can Masked Autoencoders Also Listen to Birds?"
authors: "Lukas Rauch, Rene Heinrich, Ilyass Moummad, Alexis Joly, Bernhard Sick, Christoph Scholz"
venue: "TMLR (Transactions on Machine Learning Research) 2025"
year: 2025
url: "https://arxiv.org/abs/2504.12880"
code: "https://github.com/DBD-research-group/Bird-MAE"
read_date: 2026-04-26
status: read
tags:
  - audio
  - bird-sound
  - representation-learning
  - semi-supervised
  - sed
---

## TL;DR

> AudioSet で事前学習した汎用 Audio-MAE を鳥類音声ドメインに特化させた **Bird-MAE**。BirdSet (1.6M 録音, 9,735 種) で自己教師あり学習し、**全 8 つの下流タスクで SOTA** を達成。さらに、**プロトタイプ探索 (Prototypical Probing)** という frozen 表現の活用法を提案し、linear probing を MAP で最大 +37% 改善、fine-tuning との差をわずか 3% に縮小。

## 背景・問題設定

- Audio-MAE (AudioSet 事前学習) は汎用音声タスクに強いが、**鳥類音声の微細な種間差異・高い種内変動** を捉えきれない。
- 汎用モデルをドメイン特化するには「データだけ変えれば良い」のか、「パイプライン全体の適応」が必要なのかが未解明。
- BirdSet が提供する大規模ベンチマーク (8 下流タスク、21〜132 種) で初めて体系的に検証。

## 手法

### アーキテクチャ

- **ViT-Base/16, ViT-Large/16, ViT-Huge/16** の 3 サイズ
- **Mask ratio**: 0.75 (Audio-MAE の 0.8 から変更)
- **Decoder**: ViT (Audio-MAE の Swin から変更)
- **入力**: 32 kHz → 128 mel bins × 512 frames のスペクトログラム

### 事前学習

- **データ**: XCL-1.6M (Xeno-Canto から 1.6M の curated 録音。元の 3.4M からキュレーション)
- **9,735 鳥種**をカバー
- **エポック**: 150 (飽和確認済み)
- **バッチサイズ**: 1024
- **Mixup**: $\alpha = 0.3$
- **学習率**: 0.0002

### プロトタイプ探索 (Prototypical Probing)

- 各クラス $c$ に $J$ 個のクラス固有プロトタイプベクトルを学習パラメータとして保持。
- 空間パッチ特徴とプロトタイプ間の類似度を max-pooling で計算。
- 非負重み制約付きの最終線形層。
- **frozen 表現のまま** fine-tuning に近い性能を実現。

### ドメイン適応型 fine-tuning augmentation

- Time shifting, background noise mixing, gain adjustment
- Multi-label mixup, no-call mixing
- Time/frequency masking
- これらで **約 17% の MAP 改善**。

## 実験

### Fine-tuning MAP (%) — 主要結果

| Model | POW | HSN | PER | NES | UHH | NBP | SSW | SNE |
|---|---|---|---|---|---|---|---|---|
| Perch (B1) | 41.1 | 41.8 | 18.8 | 39.1 | 27.8 | 63.6 | 28.1 | 29.5 |
| BirdNext (ConvNext) | — | — | 19.6 | — | — | 62.2 | — | — |
| Audio-MAE-B | 21.3 | — | — | — | — | 67.0 | — | — |
| **Bird-MAE-L** | **55.3** | **55.3** | **34.6** | **41.5** | **30.2** | **71.7** | **40.8** | **33.8** |
| Bird-MAE-H | 54.1 | 54.8 | 33.3 | 39.3 | 29.8 | 69.4 | 41.3 | 32.2 |

Bird-MAE-L が全タスクで最良。Huge は Large を下回る (過学習の兆候)。

### Frozen 表現の評価 (Linear vs Prototypical Probing)

| Model | Probing | HSN MAP |
|---|---|---|
| Audio-MAE-B | Linear | 8.8 |
| Audio-MAE-B | Proto | 19.4 |
| Bird-MAE-L | Linear | 12.4 |
| **Bird-MAE-L** | **Proto** | **49.0** |

Prototypical probing は linear probing を **+37pp** 改善。Fine-tuning (55.3) との差はわずか **3.3%**。

### データキュレーションの効果

- XCL-3.4M (未キュレーション): 52.2% MAP
- **XCL-1.6M (キュレーション済み)**: 55.3% MAP
- 小さくてクリーンなデータ > 大きくてノイジーなデータ。

### Few-shot 評価

- 10-shot prototypical probing で MAP ~55%。full-data fine-tuning の ~70% に対して少ないデータで高い性能。

## 強み

- **Prototypical Probing は非常に実用的**: frozen 表現のまま fine-tuning 並みの性能。計算コスト激減。
- **データキュレーションの重要性を実証**: 50% に削減しても MAP +3%。品質 > 量。
- **ドメイン特化 augmentation の定量的効果**: +17% MAP は再現すべき知見。
- **公開コード・モデル**: HuggingFace に `Bird-MAE-Large` が公開済み。

## 弱み・未解決の問い

- **BirdSet のみでの評価**: BEANS や marine タスクなど他ベンチマークでの検証がない (Perch 2.0 との直接比較が不十分)。
- **ViT-Huge の過学習**: 1.6M データでも Huge は飽和。さらなるスケールでどうなるかは不明。
- **Soundscape 環境での評価なし**: BirdCLEF 的な multi-label soundscape 設定での性能は未検証。
- **推論コスト**: ViT-L は EfficientNet より重く、Kaggle の CPU 推論制約下では不利。

## 関連研究とのつながり

- 前身: Audio-MAE (Huang et al., 2022) — AudioSet 事前学習の汎用 MAE。Bird-MAE はこのドメイン特化版。
- 同時期: [[papers/2026/perch2-bioacoustics-2025]] — Perch 2.0。BirdSet AUROC 0.908 vs Bird-MAE-L 0.886 (AUROC ベース)。
- 後続応用: BirdCLEF+ 2025 の上位解法では frozen Bird-MAE 特徴 + SED ヘッドの組み合わせが試みられている。

## 自分の研究・実装への示唆

**BirdCLEF+ 2026** への示唆:

1. **Prototypical Probing の即座活用**
   - Perch embedding に prototypical probing を適用すれば、frozen のまま性能改善が見込める。
   - 実装: クラスごとに 4 プロトタイプ + max-pool + non-negative 線形層。数十行で実装可能。
2. **データキュレーションの優先**
   - 学習データに低品質録音が混在している場合、50% 削減しても性能が上がる可能性。
   - Silero-VAD や energy-based filtering でクリーンサブセットを構築する価値あり。
3. **ドメイン特化 augmentation**
   - Background noise mixing + no-call mixing + time/freq masking を NFNet 学習に導入。+17% は期待しすぎだが、数 % の改善は堅い。
4. **Bird-MAE-L を frozen 特徴量として活用**
   - HuggingFace からダウンロード可能。Perch + Bird-MAE-L の 2 系統 embedding アンサンブルが面白い。
   - ただし ViT-L の推論速度が CPU 制約に収まるか要確認。
5. **Few-shot 補強**
   - レア種 (サンプル < 10) には prototypical probing が linear probe より遥かに強い。レア種戦略として採用可能。
