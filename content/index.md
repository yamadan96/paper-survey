---
title: Paper Survey
---

論文サーベイの個人ノート。LLM / VLM / MLOps を中心に、気になった論文を読んだ順に溜めていく場所。

## ここから読む

- [[papers/|論文ノート一覧]]（年別）
- [[topics/|トピック別まとめ]]（テーマごとの横断整理）
- [[templates/paper_template|論文ノート用テンプレート]]

## ディレクトリ構成

- `papers/YYYY/<slug>.md` — 1 論文 1 ノート。要約＋批判的読解。
- `topics/<topic>.md` — 分野横断のテーマページ。関連ノートを wikilink で束ねる。
- `templates/paper_template.md` — 新規ノート作成時にコピーする雛形。

## 運用ルール

- 論文間リンクは `[[papers/2024/example]]` 形式。Quartz が自動でグラフ化する。
- Frontmatter の `tags:` に `llm` / `vlm` / `mlops` / `rl` などを付けてフィルタ用途に使う。
- `status:` に `reading` / `read` / `skim` のいずれかを書き、積読と読了を区別する。

## このサイトについて

- [Quartz v4](https://quartz.jzhao.xyz/) で生成、GitHub Pages に自動デプロイ。
- ソース: [github.com/yamadan96/paper-survey](https://github.com/yamadan96/paper-survey)
- 著者: [yamadan96](https://yamadan96.github.io/)
