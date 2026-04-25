---
title: "Deep Neural Network Benchmarks for Selective Classification"
authors: "Andrea Pugnana, Lorenzo Perini, Jesse Davis, Salvatore Ruggieri"
venue: "Journal of Data-centric Machine Learning Research (DMLR), Vol 1, 2024"
year: 2024
url: "https://arxiv.org/abs/2401.12708"
code: "https://github.com/andrepugni/ESC"
read_date: 2026-04-26
status: read
tags:
  - selective-classification
  - cost-sensitive
---

## TL;DR

> DNN ベースの Selective Classification (SC) 手法 **18 種類** を **44 データセット** (画像 + テーブル、2 値 + 多値) で体系的にベンチマーク。「万能の勝者は存在しない」ことを実証し、手法選択はタスク・目的依存であると結論。OOD 棄却の脆弱性を浮き彫りにした実用的ガイドライン論文。

## 背景・問題設定

- **Selective Classification (SC)** = モデルが確信度の低い入力を棄却 (reject/abstain) し、残りの予測精度を高める枠組み。
- 医療、自動運転、災害対応など**誤分類コストが高い**応用で重要。
- 既存の評価は **少数データセット・少数手法・狭い評価基準** に限定されており、公正な比較が困難だった。
- 本論文の問い: 「18 手法を統一条件で比較したとき、どの手法が、どの条件で優れるか？」

## 手法

### 18 手法の分類

**Learn-to-Abstain (棄却クラス学習型)**:
- DG (Deep Gamblers): 予測に "abstain" クラスを追加し、不確実な入力を棄却クラスに分類。
- SAT, SAT+EM: Self-Adaptive Training + Entropy Minimization。

**Learn-to-Select (選択関数同時学習型)**:
- SELNET: 分類器と confidence function を同時学習し、target coverage に最適化。
- SELNET+EM: SelNet + Entropy Minimization。

**Score-based (スコア閾値型)**:
- SR (Softmax Response): max softmax 確率を confidence とする最もシンプルな手法。
- SAT+SR, SELNET+SR: 上記手法 + SR の組み合わせ。
- ENS (Ensemble): アンサンブルのエントロピーまたは平均 SR。
- ENS+SR: アンサンブル + SR (最強候補)。
- CONFIDNET: 追加ネットワークで confidence を推定。
- REG: 回帰ベースの confidence 推定。
- SELE, SCROSS: 交差検証ベースの confidence。
- PLUGINAUC, AUCROSS: AUC 最適化ベースの手法。

### 評価基準

1. **Selective Error Rate** (主指標): 棄却後の残りの予測の誤り率。
2. **Empirical Coverage**: 実際に予測を行ったサンプルの割合 (target coverage にどれだけ近いか)。
3. **Rejected Class Distribution**: 棄却されたサンプルのクラス分布 (少数クラスが偏って棄却されないか)。
4. **OOD Performance**: 分布外データに対する棄却能力。

### 評価設定

- **7 つの target coverage**: 70%, 75%, 80%, 85%, 90%, 95%, 99%
- **44 データセット**: 画像 (CIFAR-10/100, SVHN 等) + テーブルデータ (UCI 等)
- **2 値分類 + 多値分類**

## 実験

### 主要発見

1. **万能の勝者は存在しない**: 全データセット・全 coverage で一貫して最良の手法はない。
2. **ENS+SR (アンサンブル + Softmax Response) が最も安定**: selective error rate で最も高頻度に上位にランク。
3. **SR (Softmax Response) が驚くほど強い**: 最もシンプルな手法が複雑な Learn-to-Abstain/Select 手法と同等以上の場合が多い。
4. **Coverage キャリブレーション**: ほとんどの手法で target coverage に近い empirical coverage を実現。キャリブレーション手法は有効。
5. **少数クラスの偏り棄却**: 多くの手法が少数クラスのサンプルを優先的に棄却する傾向。**cost-sensitive 応用では要注意**。
6. **OOD 棄却の失敗**: 既存 SC 手法は **OOD サンプルの棄却に失敗** する場合が多い。SC と OOD 検出は別問題。

### 定量的ハイライト

- ImageNet top-5: 2% error を 99.9% 信頼度で保証可能、coverage ~60%。
- 低 coverage (70%) では Learn-to-Select 手法が有利な傾向。
- 高 coverage (95%+) では Score-based (SR, ENS+SR) が安定。

## 強み

- **規模**: 18 手法 × 44 データセット × 7 coverage は前例のない規模。
- **実装公開**: GitHub に全手法の統一実装。再現性が高い。
- **実用的ガイドライン**: 「何を使うべきか」の明確な指針。
- **OOD の落とし穴の指摘**: SC = OOD 検出ではないことの明確化。

## 弱み・未解決の問い

- **Cost-sensitive 設定の欠如**: 全クラスの誤分類コストを均一と仮定。cost matrix 付きの評価がない。
- **大規模モデル (ViT, DINOv2) での検証なし**: ResNet 系の CNN 中心。Foundation model 時代の SC は未検証。
- **推論時計算コストの比較なし**: ENS は学習・推論ともにコストが高いが、その trade-off は定量化されていない。
- **時系列・音声データなし**: 画像 + テーブルのみ。音声 SC への適用は自明でない。

## 関連研究とのつながり

- 前身: SelectiveNet (Geifman & El-Yaniv, 2019) — SELNET の原型。coverage 制約付き学習。
- 理論基盤: Selective Classification for Deep Neural Networks (Geifman & El-Yaniv, NeurIPS 2017) — SR の理論的正当化。
- 同時期: Calibrated Selective Classification (Fisch et al., TMLR 2022) — [[papers/2026/fisch-calibrated-sc-2022]] 校正済み SC。
- 応用: Selective Conformal Risk Control (Xu et al., 2024) — conformal prediction + SC の統合。

## 自分の研究・実装への示唆

**hisaichi (災害建物損傷度分類)** への直接的アクション:

1. **SR (Softmax Response) をまず実装**
   - 最もシンプルかつ安定。既に実装済みの selective classification (ECE=0.1147, $\theta=0.70$ で Coverage 69.1%/Acc 76.9%) の confidence function が SR ベースなら、これは正しい選択。
   - 追加コストゼロで baseline として使える。
2. **ENS+SR の採用検討**
   - 10-seed の DINOv2+LoRA モデルが既にあるので、アンサンブル平均 SR が最も自然な拡張。
   - 各 seed のモデルで prediction → 平均 softmax → max を confidence に。実装数行。
3. **少数クラスの偏り棄却に注意**
   - 本論文の知見: SC は少数クラスを優先棄却する。災害分類の E3/T3 (重度損傷) はサンプル少ないため、重要なクラスが棄却される危険。
   - **Cost-sensitive selective classification** (現在の IWAIT 論文テーマ) はこの問題への直接的解答。棄却閾値をクラスごとに調整。
4. **OOD の別途対策**
   - SC だけでは OOD (非災害画像、未知の損傷タイプ) を棄却できない。
   - DINOv2 embedding の Mahalanobis 距離等の OOD 検出を SC と組み合わせる 2 段階アプローチが必要。
5. **IWAIT 論文への引用**
   - 本論文は SC のサーベイ兼ベンチマークとして引用必須。「既存 SC は cost-sensitive 設定を欠く → 本論文で提案」のストーリーに直接使える。

**BirdCLEF+ 2026** への示唆:

- 234 種の分類で低確信度予測を棄却し、高確信度のみ提出する戦略。
- ただし BirdCLEF は全 row_id に予測が必要なので、棄却 = prior probability 予測にフォールバック。
