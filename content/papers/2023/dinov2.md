---
title: "DINOv2: Learning Robust Visual Features without Supervision"
authors: "Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski"
venue: "arXiv:2304.07193 / TMLR 2024"
year: 2023
url: "https://arxiv.org/abs/2304.07193"
code: "https://github.com/facebookresearch/dinov2"
read_date: 2026-04-24
status: read
tags:
  - vlm
  - ssl
  - foundation-models
  - vit
---

## TL;DR

> 画像の自己教師あり学習（Self-Supervised Learning, SSL）を真面目にスケールさせ、**凍結したまま下流タスクで CLIP 系を上回る汎用視覚特徴**を獲得した研究。データキュレーション（LVD-142M）＋ DINO/iBOT ハイブリッド目的関数＋学習安定化（KoLeo、Sinkhorn-Knopp、FSDP）が肝。

## 背景・問題設定

- SSL は BERT など NLP では「基盤モデル化」した一方、画像側は CLIP のような **弱教師あり（text-image ペア）** が事実上のデファクトだった。
- 画像の純粋な SSL（DINO, iBOT, MAE）は優れた特徴を出すが、**スケーリング時の不安定性** と **未キュレーションデータの取り込み** が課題で、CLIP を広範ベンチマークで上回る例が乏しかった。
- 本論文の問いは明確: 「SSL を適切に設計＆スケールすれば、**凍結特徴のまま** CLIP/OpenCLIP を超えられるか？」

## 手法

コアアイデア: **DINO（image-level）＋ iBOT（patch-level）のハイブリッド目的関数** を、自動キュレーションした大規模データで ViT-g/14 に対して学習し、小さな ViT に蒸留する。

### データキュレーション: LVD-142M
- 未キュレーション 1.2B 画像から、キュレーション済みソース（ImageNet, Google Landmarks 等）と近傍検索でフィルタ、重複削減した **1.42 億画像** を構築。
- 「データを増やす」ではなく「**質を維持しつつ多様性を増やす**」方向の設計。

### 目的関数
- $\mathcal{L} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + \lambda \mathcal{L}_{\text{KoLeo}}$
- **DINO**: student-teacher の CLS token に対する cross-entropy。
- **iBOT**: マスク画像モデリング。パッチレベルの token 予測を追加。
- **KoLeo regularizer**: 特徴空間での **近傍距離の対数を最大化** し、表現の一様分布を促す。ここが地味に効く。
- Softmax 正規化に **Sinkhorn-Knopp centering** を使い collapse を防止。

### 学習効率
- **FSDP**（Fully Sharded Data Parallel）、Flash Attention、fp16 な shard を活用。
- ViT-g/14（1.1B params）→ ViT-S/B/L/g へ **蒸留**。推論時は用途に合わせて軽量版を選ぶ。

## 実験

- **評価は「凍結特徴に線形プローブ or kNN」** が基本。ファインチューニングしないのがポイント。
- ImageNet-1k linear: **86.5%**（ViT-g/14）。OpenCLIP-G を上回る。
- 下流タスクが幅広く強い:
  - セグメンテーション（ADE20k）、深度推定（NYUv2）、インスタンス検索、動画理解、細粒度分類。
  - **dense prediction（セグメ・深度）で特に SSL の利が出る** → text-image ペア学習が苦手な領域。
- Ablation:
  - LVD-142M vs ImageNet-22k: キュレーションが精度に効く。
  - KoLeo の有無: nearest-neighbor 検索系が明確に改善。
  - ViT-g → 蒸留 ViT-S でも teacher を上回る場合がある。

## 強み

- 「凍結 backbone をそのまま使える」という **エンジニアリング上の圧倒的な利点**。下流で LoRA や線形 head を足すだけで済む。
- 蒸留済みチェックポイント（ViT-S/14, B/14, L/14, g/14）を公開しており、再現性・実用性が極めて高い。
- dense 特徴（patch token）の質が良く、セグメ・深度・対応探索などにそのまま流用できる。

## 弱み・未解決の問い

- **学習レシピが複雑**: DINO+iBOT+KoLeo+Sinkhorn+FSDP+teacher EMA… どれがどれだけ効いているか、読み手がフル再現するのはつらい。
- **キュレーションパイプラインがブラックボックス気味**: LVD-142M の再構築は外部研究者には事実上困難。
- テキスト対応（multi-modal retrieval 等）を捨てている。CLIP 系との使い分けが必要。
- 解像度制約（518px 程度まで）：衛星・災害画像のような高解像 × 微細構造タスクでは window 分割などの工夫が要る。

## 関連研究とのつながり

- 直接の前身:
  - [[papers/2020/vit]] — アーキテクチャの土台。DINOv2 は ViT-S/B/L/g を backbone に使う。
  - DINO (Caron et al., 2021) — student-teacher / self-distillation の original。
  - iBOT (Zhou et al., 2022) — masked image modeling を DINO に統合。
- 同時期・比較対象:
  - CLIP / OpenCLIP — 弱教師あり基盤モデル。dense タスクでは劣勢。
  - MAE (He et al., 2022) — reconstruction 型 SSL。
- 後続:
  - [[papers/2024/dinov2-registers]] — attention の artefact を register token で解消。dense task と解釈性が改善。
  - [[papers/2024/dora]] — DINOv2 backbone と組むと低 rank で有利。DoRA のほうがタスクによっては LoRA を上回る。
  - SAM 2, Depth Anything 系は DINOv2 を backbone として活用。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **DINOv2 backbone + LoRA は「少数データ × ドメインシフト」に噛み合う**
   凍結特徴が十分強いため、LoRA で低ランク適応すれば過学習を抑えつつドメイン移行できる。FIT2025 の結果と整合。
2. **Patch token の dense 性質はマルチタスク学習と相性が良い**
   IEICE2026 の MTL（損傷度 + タイプ + 重症度）で、CLS token だけでなく patch token も補助ヘッドに流すと「局所的損傷」の学習信号が拾える可能性。次の実験候補。
3. **高解像度対応はボトルネック**
   衛星・ドローン画像で 2k × 2k 入力が必要な場合、DINOv2 を sliding window で使うと patch token がエッジで崩れる。`DINOv2 + registers` 版の確認と、window overlap / mosaic cropping の比較を要検討。
4. **実装メモ**: `facebookresearch/dinov2` の `hubconf.py` から `dinov2_vits14 / vitb14 / vitl14 / vitg14` がロード可能。PEFT の LoRA と組むのは素直。GPU メモリ制約では ViT-S/14 + LoRA から始めるのが無難。

→ 次に読む:
- [[papers/2024/dinov2-registers]] — artefact 解消版（読了）。
- [[papers/2021/lora]] — PEFT 本体の再確認。DINOv2 backbone と組んで少数データ適合に使う。
- [[papers/2022/ibot]] (未作成) — iBOT 単体の挙動理解。
