---
title: "Vision Transformers Need Registers"
authors: "Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski"
venue: "ICLR 2024 (Outstanding Paper Award) / arXiv:2309.16588"
year: 2024
url: "https://arxiv.org/abs/2309.16588"
code: "https://github.com/facebookresearch/dinov2"
read_date: 2026-04-24
status: read
tags:
  - vlm
  - vit
  - interpretability
  - foundation-models
---

## TL;DR

> 大規模学習済み ViT（[[papers/2023/dinov2|DINOv2]], CLIP, DeiT-III など）の attention map に現れる「**背景パッチに集中する高ノルムトークン（artifacts）**」の原因を突き止め、解決した研究。ViT に数個の **「register token」** を追加するだけで artifact が消え、**dense prediction と object discovery が改善し、attention map も人間に解釈可能**になる。ICLR 2024 Outstanding Paper Award。

## 背景・問題設定

- DINOv2 の attention map を可視化すると、被写体ではなく**空や壁などの情報量ゼロのパッチに強い attention** が乗る現象があった。同じ挙動は CLIP / OpenCLIP / DeiT-III でも観測される。
- この「artifact token」は以下の性質を持つ:
  - **Token norm が極端に大きい**（周辺の 10〜100 倍）。
  - **局所パッチ情報を保持していない**（元の低情報パッチの内容と無関係）。
  - **グローバル情報を保持している**（classification / matching で使える）。
- Artifact は以下のタスクで実害:
  - セグメンテーション・深度推定などの **dense prediction** で精度低下。
  - **Attention 可視化の解釈性**が壊れる（なぜモデルが空を見ているのか説明できない）。
  - **Object discovery / unsupervised segmentation** で検出漏れ。
- 本論文の問い: 「この artifact は**何を補っているのか**、そして**どう消せば良いのか**？」

## 手法

### 仮説: 「スクラッチパッド仮説」

- 十分に学習された大規模 ViT は、**グローバル情報（分類に必要な要約）を格納する場所**を必要とする。
- 既存 ViT では **CLS トークン 1 個しかない** ため、モデルが勝手に**「冗長な背景パッチ」を追加ストレージとして流用**し、そこに情報を押し込む。
- これが高ノルム artifact の正体。モデルは壊れているのではなく、**意図せずに register を自作している**。

### 解決策: Register Tokens

- 入力の patch token 列に、**$N$ 個の学習可能な `[reg]` token を追加**。`[CLS]` と同じ扱いだが、最終層で使わず捨てる。
- $N = 4$ で十分効果が出る。`N = 8, 16` でも大差なし。
- 学習全体（DINO + iBOT + KoLeo の full pipeline）を register 込みで再学習する必要あり（既存の DINOv2 重みに後付けは困難）。

### 効果の観測

- **背景パッチの token norm が正常化**。Histogram で artifact ピークが消える。
- Register token 自身が artifact を**吸収**し、高ノルムを引き受ける。
- Attention map が被写体に素直に乗るようになり、**人間が見て納得できる可視化**になる。

## 実験

- **評価**:
  - Linear probe (ImageNet-1k): わずかに改善（大差なし）。
  - **Dense tasks**: ADE20k セグメンテーション +0.5 mIoU、NYU depth 改善。
  - **Object discovery (LOST)**: 大幅改善。Artifact が消えて attention が clean になった直接効果。
  - Nearest neighbor retrieval: 改善。
- **Ablation**:
  - $N = 0$: 元の DINOv2（artifact 顕著）。
  - $N = 1$: 改善するが不十分。
  - $N \geq 4$: 安定して artifact 消失。
- **計算コスト**: seq length が +4 程度増えるだけ。実質無視できる。
- **公開チェックポイント**: `dinov2_vits14_reg4`, `dinov2_vitb14_reg4`, `dinov2_vitl14_reg4`, `dinov2_vitg14_reg4` が利用可能。

## 強み

- **観察 → 仮説 → 最小介入で解決**という研究の型が綺麗。ICLR Outstanding 納得の論文。
- 「なぜ attention が変な場所を見るのか」という**長年の経験的疑問に説明**を与えた。
- 実装が極めて軽い（token を 4 個足すだけ）。既存 ViT 再学習に組み込むコストが低い。
- **Dense prediction と解釈性の両方**を同時に改善できる稀な改良。
- Object discovery / open-vocabulary segmentation など、**patch token の質が本質的に効くタスク**での改善が大きい。

## 弱み・未解決の問い

- **既存 DINOv2 重みに後付けできない**。Register 込みで再学習が必須。公開チェックポイントに頼ることになる。
- **なぜ $N = 4$ か**は経験則。理論的な下限・上限は示されていない。
- **小規模モデル（DeiT-S クラス）では効果が小さい**。artifact は十分大きなモデルで顕在化する現象のため。
- Supervised ViT（DeiT-III）でも効くが、**self-supervised より効果が弱い**傾向。
- Register が吸収しきれない残存 artifact が稀に出る事例がある（論文の付録）。

## 関連研究とのつながり

- 直接の前身:
  - [[papers/2023/dinov2]] — artifact が最も強く観察された対象。本論文は DINOv2 v1 の実用改良版の位置付け。
  - CLIP (Radford et al., 2021), DeiT-III — artifact を共有する対象として分析。
- 同時期・関連:
  - Attention sink (Xiao et al., 2023, LLM 側) — LLM で同じく「高ノルム無意味トークン」が出現する現象。横の並びで面白い。
  - Dense Prediction Transformers (Ranftl et al., 2021) — patch token の質が効くタスクの代表。
- 応用:
  - SAM 2 の backbone、Depth Anything v2 の DINOv2 利用など、**dense 特徴を使う下流で register 版が標準化** しつつある。
  - VLM（LLaVA 系）の visual encoder としても register 版が採用事例あり。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **衛星・航空・ドローン画像は artifact が出やすい**
   - 被災地の画像は空・海・単色の瓦礫面など**情報量の低い広い領域**を多く含む。典型的な artifact 発生条件。
   - 現行の DINOv2（v1）を使っている場合、**register 版へ差し替えるだけで patch token が改善する可能性が高い**。
2. **IEICE2026 の MTL 拡張（patch token を補助ヘッドに流す）に必須**
   - 先の DINOv2 ノートで提案した「CLS だけでなく patch token も MTL 補助ヘッドに流す」設計は、**patch token が artifact で汚れていると効かない**。
   - Register 版に差し替えれば、patch-level の「タイプ」「重症度」補助ヘッドが安定して学習できる見込み。**IEICE2026 の次の実験候補として最優先**。
3. **可視化の説得力が跳ね上がる**
   - 学会発表・論文で「モデルは損傷部分を正しく見ているか」を示す attention map 可視化は、register 版にしないと**レビュアに「なぜ空を見ているのか」と突っ込まれる**リスクがある。
   - Register 版 DINOv2 なら attention map が被写体に素直に乗るため、**発表映えと説得力が上がる**。
4. **実装着手の最短ルート**
   - `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg4')` に切り替えるだけ。LoRA / DoRA / QLoRA のコードはそのまま使える。
   - checkpoint は v1 と互換でなく、**LoRA adapter は register 版で取り直す**必要がある。
5. **高解像度入力への影響**
   - 被災画像は 2k×2k などの高解像で扱うことも多い。Sliding window で処理すると各 window 端で artifact が発生する問題が緩和される見込み。
   - ただし window 間の register token は独立なので、**window 重なり部での特徴一貫性**は別途確認が必要（実測で比較）。
6. **研究ストーリーへの組み込み方**
   - 「DINOv2 + LoRA（FIT2025） → DINOv2+Registers + DoRA + MTL（次）」という**連続的な改良ストーリー**が自然に組める。
   - それぞれのステップで何が効いたかを分離して測れば、学位論文・学会論文の段階的貢献として綺麗にまとまる。

→ 次に読む:
- iBOT (Zhou et al., 2022) — [[papers/2023/dinov2]] の内部で使われている patch-level 目的関数。register と patch token の関係を深く理解するのに必要。
- Attention Is All You Need (Vaswani et al., 2017) の再読 — register は結局 [CLS] 拡張なので、原点に戻ると理解が深まる。
- Attention Sink (Xiao et al., 2023) — LLM 側の類似現象。現象の普遍性を押さえる意味で。
