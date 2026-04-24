---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby"
venue: "ICLR 2021 / arXiv:2010.11929"
year: 2020
url: "https://arxiv.org/abs/2010.11929"
code: "https://github.com/google-research/vision_transformer"
read_date: 2026-04-24
status: read
tags:
  - vit
  - foundation-models
  - classification
  - transformer
---

## TL;DR

> 画像を $16 \times 16$ のパッチに分割し、各パッチを 1 トークンとして **標準 Transformer エンコーダにそのまま突っ込む** だけで ImageNet を解く研究。CNN 特有の帰納バイアス（局所性・translation 等変性）を排し、**「十分なプレトレデータ（JFT-300M）があれば inductive bias より scale が勝つ」** ことを実証。現代の Vision Transformer（ViT）系研究（DINO, DINOv2, MAE, Swin, CLIP）の出発点。

## 背景・問題設定

- 2020 年時点で画像認識は **CNN（ResNet, EfficientNet, BiT）が支配的**。Attention を部分的に使うハイブリッド手法（Non-local Nets, BoTNet）はあったが、**Transformer を pure に使う試み**は小規模で失敗してきた。
- NLP 側では Transformer が BERT / GPT で完全勝利しており、「なぜ画像側では効かないのか？」が open question。
- 本論文の問い: 「**CNN 特有のバイアスを一切入れず、純 Transformer に画像を食わせたらどうなるか？ そして、データ量を十分に増やしたときに何が起こるか？**」

## 手法

### パイプライン

画像 $x \in \mathbb{R}^{H \times W \times C}$ を処理する流れ:

1. **パッチ分割**: $P \times P$ パッチに分割（典型的に $P = 16$）。系列長 $N = HW / P^2$。
2. **Patch embedding**: 各パッチをフラット化して線形射影 → $D$ 次元ベクトル。
3. **[CLS] token を先頭に追加**（BERT と同じ）。
4. **学習可能な 1D 位置埋め込み**を足す（2D sinusoidal でも大差なし、と報告）。
5. **標準 Transformer encoder**（Multi-Head Self-Attention + MLP + LayerNorm, Pre-LN）を $L$ 層。
6. 最終層の [CLS] に **MLP ヘッド** を繋げて分類。

### モデルバリアント

| Model | Layers | Hidden $D$ | Heads | Params |
|-------|--------|-----------|-------|--------|
| ViT-Base (B/16) | 12 | 768 | 12 | 86M |
| ViT-Large (L/16) | 24 | 1024 | 16 | 307M |
| ViT-Huge (H/14) | 32 | 1280 | 16 | 632M |

`/16` や `/14` はパッチサイズ。小さいほど系列長が増え計算コスト増、精度向上。

### 設計判断

- **Inductive bias を極限まで削る**: 局所性も translation 等変性も入れない。位置情報は埋め込みだけ。
- これが「データが少ないと CNN に負ける」原因であり、**「データを増やせば自ら学習する」** 前提。

## 実験

- **プレトレーニングデータ量による挙動**:
  - **ImageNet-1k** (1.3M 画像) のみ: ResNet に負ける。
  - **ImageNet-21k** (14M): ResNet 同等。
  - **JFT-300M** (Google 内部データセット、3 億画像): **ResNet (BiT) を上回る**。
- **下流タスク** (ImageNet, CIFAR-10/100, VTAB-19 tasks):
  - ViT-H/14 (JFT 事前学習) が ImageNet top-1 **88.55%** で当時 SOTA。
  - VTAB で BiT を上回る。
- **計算効率**: 同じ精度なら ResNet より **FLOPs ベースで約 2〜4× 効率**（特に大きなモデル）。
- **Attention 可視化**: 浅い層は局所的、深い層はグローバルな注目パターンを学習する。
- **Ablation**:
  - 位置埋め込み: 1D 学習可能で十分。2D sinusoidal や relative でも大差なし。
  - パッチサイズ: 小さい（$14$）ほど精度高いが計算重い。
  - [CLS] token vs global average pooling: ほぼ同等。論文は [CLS] を採用。

## 強み

- **シンプル**: BERT そのもの。画像処理の前処理が「パッチ化 + 線形射影」だけ。アーキテクチャ特有の工夫が皆無で、実装と再現が極めて楽。
- **Scaling hypothesis の視覚版での実証**: データ量 → モデル表現力の単調増加を明確に示した。以降の基盤モデル研究の動機付け。
- **NLP との統一**: Transformer 一本で画像・テキスト・マルチモーダルを扱える道を開いた（CLIP, Flamingo, LLaVA へ繋がる）。
- **後続のプラットフォーム化**: DINO / DINOv2 / MAE / CLIP / SAM などが全てこのアーキテクチャの上に乗っている。

## 弱み・未解決の問い

- **データ飢餓**: ImageNet-1k スケールでは CNN に劣る。**帰納バイアスの欠如を埋める十分なデータが必要**。この弱点は DeiT / MAE / DINO が埋めに行く。
- **高解像度が高コスト**: 系列長が $O(HW/P^2)$、Attention は $O(N^2)$。大きな画像で爆発。Swin / ViT-H/14 with windowing などで緩和。
- **位置埋め込みの解像度非汎化**: 学習時と異なる解像度で推論するには interpolation が必要で、性能が落ちる。
- **[[papers/2024/dinov2-registers|Attention artifact]]** — 後に判明したが、学習が進むと低情報パッチに高ノルムトークンが出現する。ViT 設計の副作用。
- **2D 構造の情報を活用していない**: 自然画像は局所相関・スケール階層を持つが、pure ViT は全部自力学習。Swin 系が階層構造を再導入する動機になった。
- **少数データドメイン（医療、衛星、災害）への直接適用は難しい**。プレトレと LoRA/DoRA の組み合わせが実用解になる（現代的な使い方）。

## 関連研究とのつながり

- 系譜上の前身:
  - BERT (Devlin et al., 2019) — アーキテクチャそのもの。[CLS] も継承。
  - iGPT (Chen et al., 2020) — ピクセル単位の Transformer。ViT のパッチ化で計算効率を得た。
  - Non-local Neural Networks (Wang et al., 2018) — attention を CNN に混ぜる試み。ViT はこれを「混ぜる」ではなく「全部 attention」で解決。
- 後続:
  - **DeiT (Touvron et al., 2021)** — ImageNet-1k のみで ViT を訓練可能にする蒸留手法。データ飢餓の緩和。
  - **[[papers/2021/mae]]** — マスク画像モデリングで self-supervised 事前学習。ViT のデータ依存を self-supervised で解消。
  - **Swin Transformer (Liu et al., 2021)** — 階層構造と window attention。高解像度対応。
  - **DINO (Caron et al., 2021) → [[papers/2023/dinov2|DINOv2]]** — self-distillation ベースの SSL。現在の視覚基盤モデルの主流。
  - **CLIP (Radford et al., 2021)** — ViT + contrastive text-image pretraining。VLM の原点。
  - **[[papers/2024/dinov2-registers|Vision Transformers Need Registers]]** — ViT の attention artefact を発見・解決。本論文の欠点の一つを埋めた。
- 競合・対抗:
  - ConvNeXt (Liu et al., 2022) — 「ResNet を ViT 並みに磨けば勝てる」という反証。**FIT2025 の比較対象**。
  - EfficientNet / BiT — CNN スケーリングの代表。ViT に取って代わられた。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **FIT2025 の ResNet vs ConvNeXt vs ViT 比較を理論的に位置付けられる**
   - ViT の弱点は「データ飢餓」と「帰納バイアスの欠如」。被災建物データセットは小規模 → 一見 CNN 有利。
   - **しかし ViT + LoRA が勝てるのは、プレトレ（ImageNet-21k や DINOv2）が帰納バイアスの欠如を補うから**。これを原理から言えると、学会発表の説得力が上がる。
2. **LoRA / DoRA が ViT で特によく効く理由**
   - ViT の MLP + Attention 重みは**高次元の線形射影の集合**。LoRA の低ランク仮説（intrinsic rank）と自然に噛み合う。
   - ConvNeXt のような CNN は畳み込みカーネル（局所バイアスが組み込み済み）で、**LoRA 的な低ランク更新とは設計哲学が違う**。FIT2025 で ViT+LoRA が強かったのは構造的必然。
3. **パッチサイズの選択は解像度タスクで重要**
   - 災害画像は建物損傷の細部が重要 → `/14` や `/8` の小さなパッチが有利な可能性。
   - ただし系列長が 2〜4 倍になるので VRAM・学習時間と相談。**[[papers/2023/qlora|QLoRA]] + 小パッチ**で VRAM を捻出する手がある。
4. **位置埋め込みの解像度問題**
   - 高解像衛星画像 ($2048 \times 2048$) を使う場合、事前学習時（$224 \times 224$）の位置埋め込みを**interpolate** して使う必要あり。
   - 単純な bilinear で落ちる場合があるので、**evaluate 時に精度が再現しないと疑う箇所の一つ**。
5. **Pure ViT → 改良系への移行ロードマップ**
   - 現在: ViT + LoRA（FIT2025）
   - 次: [[papers/2023/dinov2|DINOv2]]（self-supervised ViT） + LoRA → 少数データへの耐性向上
   - 次々: [[papers/2024/dinov2-registers|DINOv2+Registers]] + [[papers/2024/dora|DoRA]] + MTL → dense task 性能・解釈性・パラメータ効率の全面改善
   - **各ステップで ViT 原著のどの制約を緩和しているか**を明示すれば、学位論文の段階的貢献として綺麗。
6. **発表・論文での位置づけ**
   - 「ViT は帰納バイアスを捨てた分、データとプレトレで補う設計」という一文は、ResNet/ConvNeXt 比較の前提として冒頭に置く価値あり。査読者が前提を共有しやすい。

→ 次に読む:
- DeiT (Touvron et al., 2021) — 小規模データで ViT を訓練する蒸留手法。
- [[papers/2021/mae]] — self-supervised pretraining で ViT のデータ飢餓を解消（読了）。
- ConvNeXt (Liu et al., 2022) — CNN を ViT に近づけた反証。FIT2025 比較対象の本尊。
