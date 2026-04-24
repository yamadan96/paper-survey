---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen"
venue: "ICLR 2022 / arXiv:2106.09685"
year: 2021
url: "https://arxiv.org/abs/2106.09685"
code: "https://github.com/microsoft/LoRA"
read_date: 2026-04-24
status: read
tags:
  - peft
  - llm
  - fine-tuning
  - adapter
---

## TL;DR

> 事前学習済み重みを凍結したまま、各層の重み更新 $\Delta W$ を**低ランク分解 $\Delta W = BA$** で近似する PEFT 手法。学習パラメータを 10000 倍削減しつつ、推論時は $W + BA$ を統合できるため**レイテンシ増加ゼロ**。GPT-3 175B の full fine-tuning と同等以上の性能を、VRAM とストレージの桁違いな削減で実現した。

## 背景・問題設定

- GPT-3 級の大規模モデルを個別タスクに適応させるとき、**全パラメータ fine-tuning（FT）** は非現実的:
  - 学習時 VRAM が爆発する（optimizer state を含め base と同サイズ以上）。
  - タスクごとに base と同サイズの checkpoint を保存する必要がある。
- 既存 PEFT には欠点があった:
  - **Adapter** (Houlsby 2019): 追加モジュールが**推論レイテンシを増やす**。
  - **Prefix Tuning / Prompt Tuning**: 入力系列の一部を「学習可能トークン」に取られ、**有効コンテキスト長が縮む**。最適化が不安定。
- 本論文の問い: 「**推論レイテンシもコンテキスト長も犠牲にせず**、full FT 級の性能を出す PEFT はどう設計すべきか？」

## 手法

### 中核アイデア: 重み更新の低ランク分解

事前学習重み $W_0 \in \mathbb{R}^{d \times k}$ に対し、更新を

$$
W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)
$$

と書く。$W_0$ は凍結、$A$ と $B$ のみ学習する。順伝播は

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

スケーリング係数 $\alpha/r$ を掛けることで、rank $r$ を変えても学習率を再調整せずに済む。

### 初期化と推論時マージ

- $A$ はガウス分布、$B$ は**ゼロ初期化**。これにより学習開始時 $\Delta W = 0$ が保証され、学習初期に base から逸脱しない。
- 推論時は $W' = W_0 + BA$ を**事前にマージ**できる。new architecture ではなくただの線形層なので、**レイテンシは完全に元と同じ**。

### 仮説: Intrinsic Rank Hypothesis

「downstream タスクへの適応で必要な重み更新は、その真の次元よりはるかに低い **intrinsic rank** を持つ」という仮説。実験的には $r = 1$〜$8$ で full FT と匹敵することを示し、この仮説を裏付ける。

### 適用対象

- Transformer の **attention の $W_q, W_v$** に入れるのが最もコスパが良い。$W_k$ や FFN への適用は効果が限定的。
- ただし Vision / Diffusion / VLM の最近の実践では、FFN や projection head まで広げるのが一般的になっている。

## 実験

- **モデル**: RoBERTa, DeBERTa, GPT-2, GPT-3 175B。
- **タスク**: GLUE, E2E NLG, WikiSQL, SAMSum 等。
- **学習パラメータ数**: GPT-3 175B で full FT の **1/10000**。
- **性能**:
  - GLUE で full FT と同等。
  - GPT-3 で GPT-3 Adapter / Prefix Tuning / BitFit を上回り、full FT と同等以上のタスクが多数。
- **VRAM / ストレージ**: GPT-3 175B の fine-tuning で VRAM を約 3 倍削減、checkpoint は 350GB → 35MB。
- **Ablation**:
  - $r = 1$ でも多くのタスクで競争力がある。
  - Attention の $W_q, W_v$ の組み合わせが最良。
  - Random projection（学習しない $BA$）はダメ — 学習が効いている。

## 強み

- **推論レイテンシ増加がゼロ**というのが単純に強い。production 投入の障壁が一段下がる。
- **タスクごとに数 MB の adapter を切り替え**れば同じ base でマルチタスクサービング可能。マルチテナントに自然。
- 実装が極めて小さい。公式リポジトリも短く、追試が楽。
- 後続研究の土台として理想的 — QLoRA, DoRA, LoRA+, VeRA, LongLoRA など派生が大量に生まれた。

## 弱み・未解決の問い

- **rank $r$ とどの層に入れるかは依然ヒューリスティック**。AdaLoRA などが動的配分を提案しているが、決定版はない。
- Base model と**ドメインが大きく離れる**タスクでは full FT に劣るケースがある（医療画像、衛星画像など）。
- 重みの低ランク更新で十分、という仮説は**全タスクで成り立つとは限らない**。プレトレでのカバレッジに依存。
- Bias と LayerNorm は触らないので、それらが重要なタスクでは BitFit 併用などが必要。

## 関連研究とのつながり

- 系譜上の前身:
  - [[papers/2019/adapters]] (未作成) Houlsby et al. — Adapter layers。LoRA の問題意識の起点。
  - Prefix Tuning (Li & Liang, 2021) — 入力側に学習可能 prefix を挿入。
  - BitFit (Zaken et al., 2021) — bias のみ学習。ミニマル PEFT。
- 同時期・発展:
  - [[papers/2023/qlora]] — 4bit 量子化 base + LoRA。単一 GPU で 65B FT。
  - [[papers/2024/dora]] — 重みを magnitude × direction に分解して LoRA より表現力を拡張。
  - VeRA, LoRA+, LongLoRA など。
- Vision / 基盤モデル応用:
  - [[papers/2023/dinov2]] — DINOv2 backbone + LoRA で凍結特徴に低ランク適応するのが近年の定石。
  - Stable Diffusion の LoRA — UNet の cross-attention に LoRA を刺し、少量画像でキャラ／スタイル学習。
  - SAM / SAM 2 のドメイン特化でも LoRA が多用される。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **なぜ FIT2025 で LoRA が効いたかを言語化できる**
   - ImageNet 事前学習と被災建物画像は**ドメインが離れている**が、intrinsic rank 仮説の範囲内に収まるレベルの「視覚特徴の低ランク補正」で足りた、と解釈できる。学会発表では「LoRA が効いた」だけでなく、**この仮説に沿って効いた** と位置付けると議論が強くなる。
2. **MTL×LoRA の設計選択肢**
   - IEICE2026 のマルチタスク学習（損傷度 / タイプ / 重症度）で **タスクごとに別 LoRA を切り替える** 設計が素直。共有 backbone + task-specific LoRA head は Multi-LoRA serving の考え方と同じ。
   - 代替案: 単一 LoRA を共有し、task ヘッドのみ分岐。パラメータ効率は高いが、タスク間干渉が起きやすい。
   - 比較実験として spike する価値あり。
3. **rank 選定の実験プロトコル**
   - DINOv2 + LoRA で $r \in \{1, 2, 4, 8, 16, 32\}$ を走らせ、Macro F1 と VRAM のパレートを見る。FIT2025 段階では rank を固定していたと思うので、ここが次の原著論文の伸びしろ。
4. **どの層に入れるかの ablation**
   - 標準レシピ（attention の $Q, V$）と、FFN まで広げた設定、patch embedding まで触る設定の比較。衛星・災害画像では空間特徴が重要なので FFN 拡張が効く可能性がある。
5. **QLoRA / DoRA への拡張**
   - VRAM 制約がきつい環境（研究室の共有 GPU など）では QLoRA で 4bit 化した DINOv2 + LoRA を検証する価値がある。DoRA は rank を絞れる場合の性能向上が報告されており、rank 不足時の保険として良さそう。

→ 次に読む:
- [[papers/2023/qlora]] — 量子化との組み合わせ。研究室 GPU 制約で必須級（読了）。
- [[papers/2024/dora]] — LoRA の表現力拡張。rank を絞りたい場面で（読了）。
- [[papers/2019/adapters]] — Adapter 本体。LoRA との設計思想比較。
