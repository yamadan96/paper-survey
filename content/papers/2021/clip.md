---
title: "Learning Transferable Visual Models From Natural Language Supervision (CLIP)"
authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever"
venue: "ICML 2021 / arXiv:2103.00020"
year: 2021
url: "https://arxiv.org/abs/2103.00020"
code: "https://github.com/openai/CLIP"
read_date: 2026-04-24
status: read
tags:
  - vlm
  - clip
  - contrastive
  - foundation-models
  - zero-shot
---

## TL;DR

> Web から収集した **4 億の image-text ペア (WIT)** を使い、画像 encoder とテキスト encoder を**対照学習で共有埋め込み空間に写像**する研究。下流タスクで **zero-shot 分類**（「a photo of a {class}」というプロンプトとの類似度で判定）を可能にし、**ImageNet zero-shot 76.2%** で supervised ResNet-50 と同等。ドメインシフト耐性も強い。現代の VLM（LLaVA, BLIP, Stable Diffusion の text encoder 等）の土台。

## 背景・問題設定

- 既存の画像認識モデルは **固定クラス集合に対する supervised 学習**（ImageNet 1000 クラス等）で訓練され、新しいクラスには再学習が必要。
- 一方 NLP 側では GPT / T5 が「タスクをテキストで記述すれば zero-shot で解ける」方向に進んでいた。
- 先行研究（VirTex, ConVIRT）は「画像と自然言語」の対応学習を試みていたが、**データ規模が小さすぎて性能が伸び切っていなかった**。
- 本論文の問い: 「**Web スケールの image-text ペア**と**純粋な対照学習**で、**固定クラスに縛られない汎用視覚モデル**を作れるか？」

## 手法

### Contrastive Language-Image Pretraining

バッチサイズ $N$ の image-text ペア $\{(I_i, T_i)\}_{i=1}^{N}$ に対し:

1. 画像を image encoder $f_I$（ResNet または [[papers/2020/vit|ViT]]）で埋め込み。
2. テキストを text encoder $f_T$（Transformer, 63M params for largest）で埋め込み。
3. 両方を同じ次元の embedding 空間に射影し、$\ell_2$ 正規化。
4. $N \times N$ の類似度行列 $S_{ij} = f_I(I_i)^\top f_T(T_j) / \tau$ を構築。
5. **対角成分を正例、それ以外を負例**として対称 cross-entropy 損失（InfoNCE の対称版）。

$$
\mathcal{L} = \frac{1}{2}\big(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\big)
$$

### Zero-shot 推論

クラス $\{c_1, \ldots, c_K\}$ を分類したいとき:

1. 各クラスを **プロンプト**にする: `"a photo of a {c_k}"`。
2. 各プロンプトを text encoder で埋め込み → $K$ 個の「クラス埋め込み」。
3. 画像を image encoder で埋め込み、**最も類似度が高いクラスを選択**。

「学習時にそのクラスを一度も見ていない」状態で分類できる、これが zero-shot の本体。

### スケール

- **データ**: Web から 4 億の image-text ペア (WIT, Web Image-Text)。公開はされていない。
- **モデル**: ResNet-50/101, ViT-B/32, ViT-B/16, ViT-L/14 など 9 種類。
- **計算**: 最大で V100×592 カード日相当。OpenAI 以外が再現困難だったため、後に **OpenCLIP / LAION-400M** が再現版を公開。
- **バッチサイズ**: 32,768（対照学習は負例が多いほど強い）。

## 実験

- **Zero-shot 分類**（30 データセット平均）:
  - **ImageNet**: 76.2%（supervised ResNet-50 と同等）。
  - CIFAR-10/100, STL-10 などで強い。
  - 細粒度（Flowers-102, Stanford Cars, Aircraft）や反直観的タスク（MNIST 88%）で苦戦。
- **Robustness to distribution shift**:
  - ImageNet-V2 / Sketch / A / R での精度低下が supervised モデルより大幅に小さい。**一般化性能の質的差**。
- **Linear probe**（凍結特徴に線形層）: 多くのデータセットで supervised pretraining を上回る。
- **Few-shot 学習**: 16 shot で zero-shot と同等、それ以上で supervised を超える。
- **Scale 実験**: データ量とモデルサイズに対して単調に性能向上。**飽和の兆しは見られない**。

## 強み

- **Zero-shot の実用レベル化**: 「クラスをテキストで書けば分類できる」という UI は運用面で革命的。タスクごとの再学習コストがゼロ。
- **Distribution shift への耐性**: supervised より「自然な」特徴を獲得している証左。
- **汎用プラットフォーム化**: 下流での VLM（LLaVA, BLIP-2）、text-to-image（Stable Diffusion, DALL-E）、open-vocabulary 検出／セグメンテーション（OWL-ViT, CLIPSeg）、retrieval までほぼ全て CLIP embedding が基礎に。
- **Prompt engineering という新しいパラダイム**: 「a photo of a {class}」→ 「a photo of a {class}, a type of pet」などテンプレート工夫で精度が変わる。

## 弱み・未解決の問い

- **細粒度分類が弱い**: 鳥種・車種・航空機種など、視覚的差が微小なタスクで大幅に劣る。
- **空間推論・カウントが壊滅的**: 「左側の犬」「3 匹の猫」など位置・数は苦手。global 特徴なので dense 情報が捨てられている。
- **Dense prediction は本質的に不可**: セグメ・検出には patch token を追加の工夫で取り出す必要（CLIPSeg, MaskCLIP）。
- **Text encoder が固定で弱い**: downstream でテキスト側を fine-tune しにくい設計。後続の BLIP / LLaVA は text 側を LLM に差し替える方向。
- **データバイアス・有害性**: Web 由来のため差別的・有害な関連付けを学習する。NSFW / stereotypes のリスク。
- **[[papers/2024/dinov2-registers|Attention artifact]]**: CLIP の ViT も同じ artifact を持つ。register が解決。
- **再現性**: WIT が非公開で、OpenCLIP / LAION で再現するまで community reproducibility に時間が掛かった。

## 関連研究とのつながり

- 系譜上の前身:
  - [[papers/2020/vit]] — image encoder 候補の一つ（ViT-B/16, ViT-L/14）。
  - VirTex (Desai & Johnson, 2020) — small-scale image-text contrastive。CLIP の直接の前身。
  - ConVIRT (Zhang et al., 2020) — 医療画像での image-text 対照。
  - SimCLR (Chen et al., 2020) — 対照学習の方法論的ベース。
- 同時期・再現:
  - ALIGN (Jia et al., 2021) — Google 版 CLIP。ノイジーな 1.8B ペアで学習。
  - OpenCLIP / LAION — 再現可能版。研究コミュニティのデファクトに。
- 改良:
  - **SigLIP (Zhai et al., 2023)** — softmax を sigmoid に置き換え、小バッチで強い。
  - EVA-CLIP — 大規模化・蒸留。
  - Florence / Florence-2 — Microsoft の汎用 VLM。
  - MetaCLIP — Meta の再現・強化版。
- 応用・下流:
  - [[papers/2023/dinov2|DINOv2]] — CLIP と比較対象。SSL vs text-supervision の軸を明確化。
  - LLaVA / BLIP-2 — CLIP を visual encoder として LLM に接続。現代の VLM の基本構成。
  - Stable Diffusion / DALL-E 2 — text encoder 部分に CLIP 採用。
  - OWL-ViT / CLIPSeg / MaskCLIP — open-vocabulary 検出／セグメ。
- 対照:
  - [[papers/2021/mae]] — reconstruction 系 SSL。対照系の CLIP と設計哲学が対極。

## 自分の研究・実装への示唆

現在の **被災建物画像 多クラス損傷度分類（FIT2025 / IEICE2026）** との接続点:

1. **Zero-shot ベースラインとして CLIP を必ず引く**
   - 被災建物分類で CLIP を zero-shot 適用し、プロンプトを `"a photo of a {damage_level} building"` 形式で並べれば、**「追加学習なしの下限性能」**が得られる。
   - これを FIT2025 / IEICE2026 の supervised 結果と並べると、**「domain-specific fine-tuning の価値」を定量化**できる。論文の説得力が増す。
2. **CLIP embedding を retrieval として使う運用**
   - 大量の被災画像が流れてくる災害対応シナリオで、**「過去の類似損傷事例を retrieval」**するユースケースは CLIP が得意。
   - 分類ではなく「支援システム」として CLIP を組み込む提案は、現場ニーズとも噛み合う。研究の社会実装軸として強い。
3. **テキストで MTL 補助タスクを記述する発想**
   - IEICE2026 の MTL（損傷度 + タイプ + 重症度）の各タスクを **テキストプロンプトで記述し CLIP 類似度を補助損失にする**設計が考えられる。
   - 例: 画像と `"collapsed building"` の類似度を subtask 1、`"partially damaged roof"` を subtask 2。**task head を持たずにテキストだけで MTL を構築**できる。
   - 新規性あり。少なくとも spike して挙動を見る価値。
4. **ただし CLIP は細粒度・空間推論が弱い**という制約を明示
   - 損傷度の微妙な差（軽微 vs 中程度）や**空間的局所性**（建物の一部のみ損傷）は CLIP 単独では掴めない可能性が高い。
   - **CLIP は zero-shot・retrieval の枠で使い、精密な分類は [[papers/2023/dinov2|DINOv2]] + [[papers/2024/dora|DoRA]] + MTL** という役割分担が現実的。
5. **OpenCLIP / SigLIP の現代的な選択**
   - OpenAI 版 CLIP は学習データが非公開で再現性に難がある。研究用途では **OpenCLIP（LAION-2B 学習）** や **SigLIP（Google）** を使うのが現代的。
   - Hugging Face から `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` や `google/siglip-large-patch16-384` がロード可能。FIT2025 の比較実験に即入れられる。
6. **CLIP+LoRA / CLIP+DoRA による domain adaptation**
   - WIT には災害画像が少ない前提で、**CLIP を LoRA / DoRA で災害ドメインに軽く domain adapt** するのは有望。
   - [[papers/2021/lora|LoRA]] は CLIP の image encoder（ViT）に素直に刺さる。text encoder 側にも刺すと prompt ロバストネスが改善する可能性。
7. **将来の研究ストーリーへの布石**
   - 現在の「災害画像分類」→「災害対応支援 VLM」への発展路が CLIP を経由して開ける。
   - 「画像だけでなくテキストも扱う災害 AI」という構図は**論文・プロジェクトとしてスケールアップしやすい**。就活・博士進学時の differentiation にも効く。

→ 次に読む:
- SigLIP (Zhai et al., 2023) — CLIP の改良、小バッチで効く実用版。
- OpenCLIP / LAION — 再現可能な実装。研究現場の標準。
- BLIP-2 (Li et al., 2023) — CLIP + LLM の接続方法の代表例。
- LLaVA (Liu et al., 2023) — instruction-tuned VLM。災害対応支援の先例。
