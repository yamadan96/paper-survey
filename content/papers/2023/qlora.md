---
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
authors: "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer"
venue: "NeurIPS 2023 / arXiv:2305.14314"
year: 2023
url: "https://arxiv.org/abs/2305.14314"
code: "https://github.com/artidoro/qlora"
read_date: 2026-04-24
status: read
tags:
  - peft
  - quantization
  - llm
  - fine-tuning
---

## TL;DR

> LoRA の base model を **4bit 量子化して凍結**することで、**65B LLM を単一 48GB GPU で fine-tuning** 可能にした研究。新提案は 3 つ: (1) 正規分布に最適な **NF4**（NormalFloat-4bit）データ型、(2) 量子化定数までもう 1 段量子化する **Double Quantization**、(3) CPU へ退避する **Paged Optimizers**。Guanaco 65B が Vicuna benchmark で ChatGPT の 99.3% に到達、24 時間／1 GPU で FT 完了。

## 背景・問題設定

- [[papers/2021/lora]] は PEFT の決定打だったが、**base model 自体は BF16 / FP16 で VRAM に載せる必要**があった。
  - 例: LLaMA 65B を FP16 で載せるだけで **130GB VRAM** が必要。研究室スケールでは事実上アクセス不可。
- 量子化 + fine-tuning の組み合わせは先行研究（GPTQ, LLM.int8()）があったが、**量子化 base + PEFT をまともに動かし、かつ full FT と同等性能を達成する** レシピは確立していなかった。
- 本論文の問い: 「**4bit 量子化した base に LoRA を刺しても、full 16bit FT と同じ精度が出せるか？**」そしてそれを可能にするエンジニアリングを詰める。

## 手法

3 つの技術要素の組み合わせ。

### 1. NF4: 4-bit NormalFloat

- 事前学習済み重みは**おおむね正規分布**に従う、という観察から出発。
- 標準正規分布 $N(0,1)$ の **quantile（等確率分位点）を 4bit の 16 値に割り当てる**情報理論的に最適な dtype。
- block-wise quantization（ブロック単位で scale を持つ）で outlier に対処。
- 保存は 4bit、**計算時は BF16 に dequantize** して matmul。勾配は BF16 で流す。

### 2. Double Quantization

- 1 段目: 重みを 4bit + block-wise scale（FP32）で量子化。
- 2 段目: **この FP32 scale 自身をさらに 8bit 量子化**する。
- 平均で約 **0.37 bits/param 節約**（65B で 3GB）。効果は地味だが無料。

### 3. Paged Optimizers

- NVIDIA Unified Memory を使い、**optimizer state（Adam の $m, v$）を必要に応じて GPU↔CPU でページング**。
- Gradient checkpointing 時のメモリスパイクで OOM する問題を緩和。`cudaMallocManaged` で透過的に扱える。

### 組み合わせ

- 4bit 凍結 base + BF16 の LoRA adapter を **全線形層**に挿入（attention だけでなく FFN まで）。
- 実装は bitsandbytes の `Linear4bit` + Hugging Face PEFT。学習ループは通常の LoRA と同じ。

## 実験

- **モデル**: LLaMA 7B / 13B / 33B / 65B、T5。
- **データ**: OASST1, FLAN v2, Alpaca, Self-Instruct 他計 8 種。最終モデル **Guanaco** は OASST1 のみで学習。
- **主結果**:
  - **Guanaco 65B が Vicuna benchmark で ChatGPT の 99.3%**。GPT-4 判定で full 16bit FT と互角。
  - Guanaco 33B も ChatGPT 超え（97.8% → 99.3% 付近）。
  - 65B を単一 A100 80GB で 24 時間学習。48GB でも動作。
- **量子化による劣化はほぼゼロ**:
  - NF4 vs FP4（均等分位）: NF4 が一貫して優位。
  - Double Quantization: 精度劣化なく VRAM 節約。
  - 全線形層に LoRA を入れる > attention のみに入れる（タスク次第）。
- **評価の注意点**: Vicuna benchmark は GPT-4 judge で、ベンチマークハッキングしやすい。論文内で MMLU / Commonsense QA なども併記。

## 強み

- **VRAM 制約の常識を変えた** — 13B / 33B / 65B の FT が「研究室 GPU 1 枚」のスケールに入った。
- **NF4 は情報理論的な設計**で、他の 4bit 量子化（FP4, INT4）と比べ自明に強い場面が多い。
- Hugging Face PEFT + bitsandbytes に統合済みで、**実装コストが実質ゼロ**。`load_in_4bit=True` と `LoraConfig` を渡すだけ。
- Paged Optimizer は QLoRA 特有というより、**Gradient checkpointing + 大バッチ**のあらゆる学習に応用できる副産物。

## 弱み・未解決の問い

- **NF4 の「正規分布仮定」は layer によって破綻**する。attention の $W_q, W_v$ や LayerNorm 後の射影など、分布が偏る層で劣化する余地。後続 LoftQ / AWQ 等が対処。
- **推論時のマージが素直にできない** — 4bit base に LoRA を足すと、数値精度の都合でマージ後再量子化する必要がある。商用 serving では別途設計が要る。
- **Block-wise quantization の block size** はハイパラであり、感度がある。
- LLM 中心の評価。**Vision / ViT への一般化**は論文では触れていない（後続研究・実装で確認されてきた段階）。
- Guanaco の評価が **GPT-4 judge 中心**で、ベンチマークとしての信頼性に注意が要る。MMLU 等の客観指標も併読必須。

## 関連研究とのつながり

- 直接の前身:
  - [[papers/2021/lora]] — ベースとなる PEFT。QLoRA は「LoRA を 4bit 凍結 base に対して動かす」ことが本質。
  - LLM.int8() (Dettmers et al., 2022) — 8bit 推論の基礎。同じ著者。
  - GPTQ (Frantar et al., 2022) — post-training 4bit 量子化。推論向け。
- 同時期・発展:
  - LoftQ (2023) — 量子化誤差を LoRA の初期化で相殺する。
  - QA-LoRA (2023) — 推論時マージまで考えた設計。
  - DoRA (2024) / LoRA+ — LoRA 側の改良。QLoRA と直交して組める。
  - HQQ — 較正データ不要の高速量子化。
- Vision 応用:
  - [[papers/2023/dinov2]] — DINOv2-ViT-L/g + QLoRA で研究室 GPU 上での domain adaptation が現実的になる。
  - SAM、Stable Diffusion の LoRA 学習でも 4bit 化が一般化。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **DINOv2-ViT-g/14 + QLoRA が現実解になる**
   - FIT2025 は ViT ベース、IEICE2026 は DINOv2 ベース。**g/14（1.1B）以上を研究室 GPU で触れる**ようになるのは大きい。
   - ただし ViT は LLM に比べ小さいため、VRAM 節約の旨みは 65B ほどではない。**「g/14 を 24GB GPU で回したい」** という具体的状況で真価が出る。
2. **NF4 の正規分布仮定が ViT で成り立つかを検証**
   - DINOv2 は self-supervised でプレトレされており、attention の重み分布が LLM と異なる可能性がある。`AbsMax 量子化誤差` を layer ごとに可視化する実験を組むと論文化できる余地あり。
3. **MTL ヘッドは BF16 で保持する設計**
   - IEICE2026 の multi-task head（損傷度・タイプ・重症度）は**量子化対象外**にして通常精度で学習するのが素直。4bit 化するのは DINOv2 backbone だけ。
4. **Paged Optimizer は副産物として即効性がある**
   - QLoRA を使わなくても、Paged Optimizer だけ単独で導入すれば「gradient checkpointing で OOM → バッチ縮小」の現在の運用が改善する。今日から入れて良い。
5. **評価プロトコルに注意**
   - Guanaco のように `GPT-4 judge` に寄せず、**Macro F1 / クラス別 Recall** を主指標にするのは現状維持で OK。量子化前後の **「最悪クラス性能の劣化」** を必ず見る。平均だけ見ると落とし穴。
6. **推論 serving の設計は別問題**
   - 本番デプロイを視野に入れるなら、`4bit base + LoRA` のまま serving するのか、FP16 にマージして serving するのかを早めに決める。後者は `merge_and_unload()` 後に再量子化する手順が必要。

→ 次に読む:
- [[papers/2024/dora]] — LoRA 側の改良。QLoRA と併用可能。
- LoftQ — 量子化誤差の LoRA 初期化補正。QLoRA の弱点を直接突く。
- AWQ — 活性化統計ベースの量子化。推論 serving 視点で。
