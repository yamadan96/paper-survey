---
title: "Masked Autoencoders Are Scalable Vision Learners"
authors: "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick"
venue: "CVPR 2022 (Oral) / arXiv:2111.06377"
year: 2021
url: "https://arxiv.org/abs/2111.06377"
code: "https://github.com/facebookresearch/mae"
read_date: 2026-04-24
status: read
tags:
  - ssl
  - vit
  - foundation-models
  - masked-image-modeling
---

## TL;DR

> **画像パッチの 75% をマスクして可視 25% から元画像を復元**する、BERT の画像版とも言える自己教師あり手法。**非対称 encoder-decoder 設計**（encoder は可視パッチだけ処理、decoder は小さく軽量）により、学習が 3〜4 倍高速化。ViT-Huge で ImageNet-1k fine-tuning **87.8%**（当時 SOTA）。DINO 系とは異なる「reconstruction」系 SSL の代表格で、[[papers/2023/dinov2|DINOv2]] の iBOT 目的関数とも近い発想。

## 背景・問題設定

- BERT が NLP で完全勝利した裏には「**入力の一部をマスクして予測**」という単純な self-supervised 目的があった。
- 画像側で同じことを試みる先行研究（iGPT, BEiT）はあったが、**BERT ほどシンプルに効く手法が確立していなかった**。
- 2020 年前後の vision SSL は contrastive（SimCLR, MoCo, DINO）が主流で、**データ拡張と負例設計**の工夫勝負だった。
- 本論文の問い: 「**画像版 BERT を素直に組んだら効かないのはなぜか？ そして、何を変えれば効くか？**」

## 手法

### 非対称 Encoder-Decoder

入力画像 $x$ をパッチ列 $\{x_1, ..., x_N\}$ に分割し、以下の処理を行う:

1. **ランダムマスキング**: パッチの 75% を除去。可視 25% のみ残す。
2. **Encoder**（ViT-Base/Large/Huge）: 可視パッチのみを処理。**マスクされたパッチを見ない**のが肝。
3. **Decoder**（小さな Transformer, 8 層 × 512 次元）: 可視パッチの表現 + 学習可能なマスクトークンを入力に、元画像の **ピクセル値を予測**。
4. **損失**: 予測ピクセルと元ピクセルの MSE。マスクされたパッチにのみ loss を適用。

推論時（下流タスク）には **encoder のみ使用**。Decoder は捨てる。

### 3 つのキー設計

1. **高マスク比率（75%）**: NLP は 15% で十分だが、画像は冗長性が高く **75% マスクしないと「簡単すぎて意味ある表現が育たない」**。
2. **非対称設計**: encoder は 25% しか見ないため**メモリ・計算が 4 倍効率化**。大きな ViT-Huge を現実的に学習可能にする。
3. **ピクセル再構成ターゲット + patch ごとの正規化**: 予測ターゲットを**パッチ内平均・分散で正規化**することで、高周波成分の学習を促進し精度向上。

### 学習のイメージ

- 可視パッチだけで画像全体を想像できるほどの**高レベル意味表現**を encoder が獲得する、という設計。
- 復元結果は視覚的にもっともらしいが、**細部は合っていない**ことが多い。重要なのは「特徴量」であり「再構成品質」ではない。

## 実験

- **モデル**: ViT-Base / Large / Huge。
- **プレトレーニング**: ImageNet-1k のみ（**JFT 不要**）で 800 エポック。
- **下流タスク（fine-tuning）**:
  - **ImageNet-1k**: ViT-Large で 85.9%、**ViT-Huge で 87.8%**（当時 SOTA）。supervised pretraining を上回る。
  - COCO 物体検出、ADE20k セグメンテーションでも改善。
- **Linear probing**: fine-tuning より明らかに弱い（DINO や MoCo-v3 に劣る）。**MAE の強みは fine-tuning 前提**。
- **モデルサイズ scaling**: supervised は ViT-Large で飽和するが、**MAE は Huge まで伸び続ける**。
- **Ablation**:
  - マスク比率: 75% が最適。40%・60%・85% は劣化。
  - Decoder 深さ: 8 層で頭打ち。軽量で OK。
  - パッチ正規化: +1% 前後の改善。
  - Decoder への位置埋め込み: 必須。
- **学習速度**: 非対称設計により **同じ精度に 3× 速く到達**。

## 強み

- **シンプル**: 「マスクしてピクセル復元」という 1 文で説明可能。実装も短い。
- **スケーラブル**: ViT-Huge まで単調に伸び、基盤モデルの SSL レシピとして有力。
- **ImageNet-1k のみで SOTA** — JFT-300M 級の大規模データを使わない点が実用的。
- **非対称設計の計算効率**が重要な貢献。以降の大規模 SSL はこの設計を踏襲。
- **Fine-tuning 前提ならば contrastive 系を上回る**タスクが多い。

## 弱み・未解決の問い

- **Linear probe が弱い**: 凍結特徴の線形分類では DINO / MoCo-v3 に劣る。**features をそのまま kNN や retrieval に使うタスクには不向き**。[[papers/2023/dinov2|DINOv2]] の iBOT + DINO ハイブリッドの方が有利。
- **Dense prediction（セグメ・深度）での改善が控えめ**: contrastive / iBOT 系に比べ patch token の質が物足りない場合がある。
- **ピクセル予測の意味**: 細部を無視した reconstruction は「意味」と「詳細」の中間を学習している。ターゲットの妥当性は理論的に曖昧。
- **Text alignment がない**: CLIP のようなゼロショット能力は獲得できない。VLM 化には別途 fine-tuning が必要。
- **Augmentation 軽視**: MAE は minimal augmentation で動くが、contrastive SSL と比べて invariance 設計が弱い。ドメインシフト耐性で劣る可能性。

## 関連研究とのつながり

- 系譜上の前身:
  - [[papers/2020/vit]] — encoder の本体。
  - BERT (Devlin et al., 2019) — masked language modeling の直接の範型。
  - BEiT (Bao et al., 2021) — discrete token を予測する MIM。pixel ではなく token。
  - iGPT (Chen et al., 2020) — autoregressive pixel generation。
- 同時期・発展:
  - SimMIM (Xie et al., 2021) — 同時期の masked image modeling。shallow decoder + pixel loss で MAE と類似の結論。
  - iBOT (Zhou et al., 2022) — MIM + self-distillation。**[[papers/2023/dinov2|DINOv2]] が採用**した統合版。
- 後続:
  - ConvNeXt V2 (Woo et al., 2023) — MAE アイデアを CNN に sparse convolution で移植。
  - VideoMAE — 動画への拡張。時空間マスキング。
  - Audio-MAE — 音声への拡張。
  - MAE-ST — 時空間 MAE。ビデオ異常検知にも応用される。
- Contrastive 系との対比:
  - DINO (Caron et al., 2021) — self-distillation + view augmentation。MAE と設計哲学が対極。
  - [[papers/2023/dinov2|DINOv2]] — DINO + iBOT ハイブリッド。MAE 的な MIM と contrastive を統合。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **MAE vs DINOv2 の選択基準を言語化できる**
   - Fine-tuning 前提 → MAE も強い選択肢。
   - 凍結特徴＋線形プローブ／kNN → DINOv2 優位。
   - Dense prediction（セグメ） → DINOv2 優位。
   - **現在の被災画像分類（classification fine-tuning）では両者競合する**。比較実験の価値あり。
2. **MAE は LoRA / DoRA と相性が良い**
   - MAE の特徴は「fine-tuning で花開く」設計。**凍結 backbone + LoRA** で下流適応するのは MAE の思想と自然に噛み合う。
   - [[papers/2021/lora|LoRA]] / [[papers/2024/dora|DoRA]] の intrinsic rank 仮説が成り立つのは MAE も同じ。FIT2025 の ViT ベース結果は MAE でも再現しうる。
3. **75% マスキングを「データ拡張」として転用する発想**
   - 被災画像は**部分的な瓦礫や遮蔽**が多い。学習時に **70〜80% の空間マスクをかける**ことで、モデルの遮蔽耐性を高められる可能性。
   - MAE pretraining を使わずとも、**マスキング比率を高めたデータ拡張**として別 backbone に導入する spike 価値あり。
4. **MAE を補助タスクとして MTL に組み込む**
   - IEICE2026 の MTL 設計で、**損傷度分類 + MAE 再構成**を補助タスクにする案が考えられる。
   - 利点: 小データで正則化効果が出る可能性。
   - 欠点: decoder の追加計算コスト。
   - 価値: 「災害画像は部分欠落が多い」という現場特性と MAE の仮定が一致している点で研究的に面白い。
5. **衛星画像・高解像度タスクでは MAE が有利な場面がある**
   - 衛星 / 航空画像は **空間スケールが大きく冗長** → MAE の高マスク比率仮定と噛み合う。
   - SatMAE（Cong et al., 2022）など衛星画像特化の MAE 変種も出ており、参照価値あり。
6. **実装着手の最短ルート**
   - `facebookresearch/mae` 公式リポジトリに ViT-B/L/H の事前学習済み重みあり。
   - Hugging Face にも `facebook/vit-mae-base` 等で統合済み。`transformers` ライブラリでロード可能。
   - FIT2025 の ViT backbone を MAE 版 ViT に差し替えて **LoRA / DoRA で再評価**する比較実験が 1 週間で組める。
7. **研究ストーリーへの位置づけ**
   - 「MAE（再構成系 SSL）と DINOv2（対照系 SSL + MIM）の比較」という軸は、**SSL の選択肢を網羅的に議論**できる強力な構造。
   - FIT2025 の次段階として「どの SSL を選ぶか」という問いを立て、その上で DINOv2 を採用する理由を述べる形に組むと論文が書きやすい。

→ 次に読む:
- iBOT (Zhou et al., 2022) — MAE と DINO を橋渡しする [[papers/2023/dinov2|DINOv2]] の構成要素。
- SimMIM — MAE の同時期・姉妹研究。決定的な違いを押さえる。
- ConvNeXt V2 — MAE を CNN に移植した例。CNN vs ViT の SSL 比較として。
- SatMAE — 衛星画像特化。災害画像と直接相性が良い。
