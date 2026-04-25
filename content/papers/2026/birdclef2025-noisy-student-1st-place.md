---
title: "1st Place Solution: Multi-Iterative Noisy Student Is All You Need (BirdCLEF+ 2025)"
authors: "Nikita Babych"
venue: "Kaggle BirdCLEF+ 2025 Competition (1st Place Write-up)"
year: 2025
url: "https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n"
code: ""
read_date: 2026-04-26
status: read
tags:
  - audio
  - bird-sound
  - pseudo-label
  - semi-supervised
  - sed
  - inference-efficiency
---

## TL;DR

> BirdCLEF+ 2025 (鳥・両生類・哺乳類・昆虫の 234 種音声識別) の 1 位解法。**Multi-Iterative Noisy Student** (反復的疑似ラベル + 自己蒸留) と **Power Scaling** を組み合わせ、EfficientNet ファミリのアンサンブルで private AUC = 0.930 を達成。ラベルなし soundscape データをフル活用する半教師あり学習が支配的勝因。

## 背景・問題設定

- BirdCLEF+ 2025 はコロンビア中部マグダレナ渓谷の soundscape から 234 種を識別するコンペ。ラベル付きデータ (Xeno-Canto 録音) と大量のラベルなし soundscape が提供される。
- 推論制約: **700 本の 1 分音声を CPU のみ 90 分以内** に処理する必要があり、軽量モデルが必須。
- ドメインシフト: Xeno-Canto (クリーン単一種録音) → soundscape (ノイズ・多種混在) の大きなギャップ。
- クラス不均衡: 出現頻度が 30 件未満のレア種が多数存在。

## 手法

### Multi-Iterative Noisy Student Training

1. **ラウンド 0 (教師モデル)**: ラベル付きデータのみで 10 個の EfficientNet を学習。
2. **ラウンド 1 (疑似ラベル I)**: 教師アンサンブルでラベルなし soundscape に疑似ラベルを付与。ラベル付き 50% + 疑似ラベル 50% で再学習。MixUp + StochasticDepth でノイズ注入。スコア: 0.872 → **0.898**。
3. **ラウンド 2〜5 (疑似ラベル II, 4 反復)**: Power Scaling を導入し、疑似ラベルの確信度をべき乗変換で先鋭化。反復ごとに教師モデルを更新。スコア: 0.898 → **0.930**。
4. **自己蒸留**: 2 回の自己蒸留パスで soft target を利用。

### Power Scaling

- 疑似ラベルの確率分布にべき乗変換 (power transform) を適用し、高確信度の予測を強化してノイジーな低確信度を抑制。
- 通常の hard threshold (0.5) よりも多くのサンプルを活用でき、情報損失が少ない。

### アンサンブル構成

| Model | Count |
|---|---|
| EfficientNet-V2-S | 4 |
| EfficientNet-V2-B3 | 3 |
| EfficientNet-B3-NS | 4 |
| EfficientNet-B0-NS (両生類・昆虫特化) | 2 |
| EfficientNet-L0 | 1 |
| RegNetY-016 | 1 |
| RegNetY-008 | 1 |

### 追加データ

- Xeno Archive から +5,489 の鳥録音を追加。
- +17,197 の昆虫・両生類録音で "other" クラスを強化。

## 実験

- **Baseline (ラベル付きのみ)**: ~0.87 AUC
- **+ 疑似ラベル I + MixUp + StochDepth**: 0.898
- **+ Power Scaling + 4 rounds pseudo-label II**: **0.930** (private)
- 1 位 (0.930) と 2 位 (0.928) の差はわずか 0.002。トップ 5 はすべて疑似ラベル + アンサンブル。

## 強み

- **コンセプトのシンプルさ**: 核心は「反復的疑似ラベル + Power Scaling」の 2 つだけ。特殊なアーキテクチャは不要。
- **スケーラビリティ**: 反復回数を増やすほどスコアが向上する傾向。4 回で収束。
- **CPU 推論の現実解**: EfficientNet ファミリは ONNX/OpenVINO 変換が容易で、CPU 90 分制約に適合。
- **ドメイン適応**: soundscape 自体を疑似ラベル対象にすることで、テストドメインのデータで直接学習。

## 弱み・未解決の問い

- **疑似ラベルの確認バイアス**: 教師モデルが誤分類する種 (レア種) は反復で悪化する可能性。
- **計算コスト**: 10 モデル × 4〜5 反復 = 40〜50 回の学習ループ。GPU リソースが潤沢でないと再現困難。
- **Power Scaling のハイパーパラメータ**: べき指数の選択基準が不明。探索コストが隠れている。
- **SED モデルとの比較欠如**: 上位陣の一部は SED モデルブレンド (Quantile-Mix) で 0.893 を達成しており、CNN-only アプローチの限界は未検証。

## 関連研究とのつながり

- 前身: Noisy Student (Xie et al., 2020, EfficientNet + ImageNet pseudo-labels)。音声分野への直接適用。
- 2 位解法 (VSydorskyy): eca_nfnet_l0 + EfficientNetV2-S, FocalBCE, ONNX → OpenVINO 変換。[[papers/2026/birdclef2025-2nd-place-sed]]
- BirdCLEF 2024: Pseudo Multi-Label Classification (arXiv:2407.06291) — 同じ疑似ラベル戦略の前年版。

## 自分の研究・実装への示唆

**BirdCLEF+ 2026** (現在 LB 0.915) への直接的アクション:

1. **反復疑似ラベルの実装が最優先**
   - 現在は Perch + NFNet の単純アンサンブルだが、soundscape 疑似ラベルを 2〜4 反復行えばスコア 0.02〜0.03 の改善が見込める。
   - Power Scaling の実装は `pred ** power / (pred ** power + (1 - pred) ** power)` の 1 行。
2. **EfficientNetV2-S の追加**
   - 現在の NFNet に加え、EfficientNetV2-S を追加してアンサンブル多様性を確保。1 位は 4 種のバックボーンを使用。
3. **CPU 推論の最適化**
   - ONNX → OpenVINO 変換で CPU 推論速度を 2〜3x 改善可能。2 位解法がこれを実証。
4. **レア種への対策**
   - 疑似ラベルの confidence threshold を種ごとに調整するか、レア種のみ Xeno Archive から追加データを注入。
5. **実装順序**: (a) 疑似ラベル 1 反復 → (b) Power Scaling 追加 → (c) EfficientNetV2 追加 → (d) 反復回数増加。
