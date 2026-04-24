---
title: "DoRA: Weight-Decomposed Low-Rank Adaptation"
authors: "Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen"
venue: "ICML 2024 (Oral) / arXiv:2402.09353"
year: 2024
url: "https://arxiv.org/abs/2402.09353"
code: "https://github.com/NVlabs/DoRA"
read_date: 2026-04-24
status: read
tags:
  - peft
  - llm
  - vlm
  - fine-tuning
  - adapter
---

## TL;DR

> 事前学習済み重み $W$ を **「方向（matrix）× 大きさ（vector）」に分解**し、方向は LoRA で低ランク更新、大きさは直接学習する手法。追加パラメータは LoRA に対して約 0.01% とほぼ無料で、**低ランク ($r = 4, 8$) で特に効く**。LLaVA などの VLM や LLaMA 系 LLM で一貫して LoRA を上回り、ICML 2024 oral。

## 背景・問題設定

- [[papers/2021/lora]] は PEFT のデファクトになったが、**full fine-tuning (FT) に対して常に劣る余地**が残る。特に rank が小さい領域では顕著。
- なぜ LoRA は FT より劣るのか？ これが本論文の出発点。
- 著者らは FT と LoRA の学習ダイナミクスを「**方向変化** と **大きさ変化**」に分けて可視化し、

  - **Full FT**: 方向と大きさの変化が **負の相関** → 独立に自由に動く。
  - **LoRA**: 方向と大きさの変化が **正の相関** → 結合して動く。更新自由度が実質低い。

  という差を発見。LoRA は「表現力不足」ではなく「**表現の結合による自由度不足**」が効いている、という仮説。
- 本論文の問い: 「**LoRA と同じパラメータ予算で、FT のような decoupled な更新パターンを再現できるか？**」

## 手法

### 重み分解: magnitude × direction

事前学習重み $W_0 \in \mathbb{R}^{d \times k}$ を

$$
W_0 = m \cdot \frac{V_0}{\|V_0\|_c}, \quad m \in \mathbb{R}^{1 \times k},\ \ V_0 \in \mathbb{R}^{d \times k}
$$

と分解する。$\|V\|_c$ は **列ごとの L2 norm**、$m$ は各列のスケール（1×k ベクトル）。初期値は $V_0 = W_0$、$m = \|W_0\|_c$。

### 更新式

- **方向 $V$** は [[papers/2021/lora]] と同じ低ランク更新 $V = V_0 + BA$。$B, A$ のみ学習。
- **大きさ $m$** は**直接学習**。1×k という極小ベクトルなので追加コスト実質ゼロ。

最終的に順伝播は

$$
h = \Big(m \cdot \frac{V_0 + BA}{\|V_0 + BA\|_c}\Big) x
$$

### 実装上のポイント

- 列正規化 $\|V_0 + BA\|_c$ の再計算が毎 step 入るため、LoRA 比で **約 1.2〜1.4 倍の学習時間**。推論時はマージして単一線形層に戻せるので、**推論レイテンシ増加はゼロ**。
- Hugging Face PEFT では `LoraConfig(use_dora=True)` で切り替え可能。既存コードベースへの差分は 1 行。
- QLoRA と直交的に組み合わせられる（**QDoRA**）。4bit base に DoRA を刺す実装が公開されている。

## 実験

- **タスクと base model**:
  - LLaMA-7B / 13B + Commonsense Reasoning（8 tasks 平均）
  - LLaVA-1.5-7B + Visual Instruction Tuning
  - VL-BART + Image / Video-Text Understanding
  - LLaMA2-7B + MT-Bench
- **主結果**:
  - 全セッティングで **DoRA > LoRA**。Commonsense で平均 +3.7 点、VL タスクで平均 +0.9 点。
  - **Rank が小さいほど差が広がる**: $r=4$ で DoRA は $r=16$ の LoRA と同等性能。
  - QLoRA（4bit base）と組んだ QDoRA も QLoRA 単体を上回る。
- **学習コスト**: LoRA 比で 1.2〜1.4×。**推論コストは同一**。
- **Ablation**:
  - 方向だけ学習（magnitude 固定）: LoRA と同等。
  - 大きさだけ学習（direction 固定）: 大きく劣化。
  - **両方学習する設計が本質**。単独ではどちらも不十分。

## 強み

- **Plug and play**: `use_dora=True` の 1 フラグで LoRA → DoRA へ移行可。既存 LoRA レシピの資産を活かせる。
- **Rank 制約が強い場面で特に効く** — VRAM や adapter ストレージが厳しい運用で実利。
- **分析先行で設計**: magnitude/direction decoupling という観察から手法を導いているため、納得感のある貢献。
- LLM / VLM 両方で検証されている。画像タスクにも効く見通しがある。

## 弱み・未解決の問い

- **学習時間 1.2〜1.4 倍**は無視できないコスト。LoRA からの乗り換えは「rank を下げられる」分で元を取る設計が前提。
- **分析は観察的**: 「decoupled 更新 = 良い」は実験的に妥当だが、**因果的な説明は与えていない**。
- **Column-wise normalization は選択の一つ**。Row-wise や block-wise の方が良い層もあり得る（論文では触れていない）。
- Vision / ViT 単独での大規模検証は少なめ。VLM 経由の評価が中心。
- Magnitude $m$ のオーバーパラメータ化（`1×k` vs `1×1` スカラー）の設計余地が残る。

## 関連研究とのつながり

- 直接の前身:
  - [[papers/2021/lora]] — 低ランク adapter 本体。DoRA は LoRA の拡張として定式化される。
  - Weight Normalization (Salimans & Kingma, 2016) — 重みを方向と大きさに分解するアイデアそのもの。DoRA の分解形式の出どころ。
- 同時期・発展:
  - [[papers/2023/qlora]] — 4bit base + LoRA。DoRA と組んで **QDoRA**。
  - LoRA+ (Hayou et al., 2024) — $A, B$ に異なる学習率を設定。DoRA と直交的に併用可能。
  - VeRA (Kopiczko et al., 2024) — ランダム射影を共有して学習スカラーのみを持つ超パラメータ効率 PEFT。別方向の改良。
  - rsLoRA (Kalajdzievski, 2023) — scaling factor を $\alpha/\sqrt{r}$ に変更。rank を上げやすくする。
- Vision 応用:
  - [[papers/2023/dinov2]] — DINOv2 backbone + DoRA は低 rank 制約下で LoRA より有利な見通し。
  - SAM、Stable Diffusion の adapter 学習でも DoRA 採用例が増えている。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **低 rank で効くという性質は研究室 GPU 運用と嚙み合う**
   - 学習時メモリは rank で決まる。**DoRA なら $r=4$ で LoRA $r=16$ 相当** の性能 → VRAM 余剰を batch size やデータ拡張に回せる。
   - IEICE2026 の MTL でタスク数が増えると adapter が積み上がるため、単一 adapter を軽くできる意味は大きい。
2. **magnitude/direction decoupling 分析を自分の実験に輸入**
   - FIT2025 で「LoRA が効いたが full FT には届かなかった」という結果があるなら、**DoRA で追試 → decouple パターンが出れば論文の主張が補強**される。
   - 可視化: fine-tuning 前後の各線形層の $\|\Delta W\|_c$ と $\cos(W_0, W_0 + \Delta W)$ 散布図。論文 Fig 2 と同じ作り方で再現できる。
3. **QDoRA を現実解として検証**
   - DINOv2-ViT-g/14 + QDoRA を A6000 / A100 40GB クラスで走らせ、LoRA / QLoRA / DoRA / QDoRA の 4 群比較。Macro F1 × VRAM × 学習時間の 3 軸パレートで示せると発表映えする。
4. **MTL における magnitude 共有 vs タスク別**
   - 興味深い設計選択: **direction（LoRA の $B, A$）はタスク共有、magnitude $m$ のみタスク別** にする、またはその逆。
   - タスク間の「どの層が重要か」の差を magnitude に吸収させる、という直感的な解釈が可能。
   - これは現行手法に無い未踏領域で、**IEICE2026 以降の独自貢献の候補**。spike 1 週間で試す価値あり。
5. **実装着手の最短ルート**
   - `peft.LoraConfig(use_dora=True)` + `bitsandbytes.load_in_4bit=True` で QDoRA が組める。**既存 LoRA コードの差分は 1 行**。
   - 学習時間増（1.2〜1.4×）は想定して schedule を組む。
6. **落とし穴: column-wise 正規化の数値安定性**
   - $\|V_0 + BA\|_c$ が極端に小さい列が出ると学習不安定。事前学習が浅いまま ViT を使うと発生しやすい。DINOv2 なら問題は出にくいはずだが、ログに `NaN / Inf` が混じったら疑う。

→ 次に読む:
- LoftQ — QLoRA の量子化誤差を LoRA 初期化で相殺。QDoRA と組むと相補的。
- LoRA+ — 学習率の非対称性。DoRA と直交的に併用可能。
- [[papers/2024/dinov2-registers]] — DINOv2 attention artefact 解消。高解像度タスクでの実装課題（読了）。
