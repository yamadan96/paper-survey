# paper-survey

論文サーベイの個人ノート。[Quartz v4](https://quartz.jzhao.xyz/) で生成して GitHub Pages に公開している。

**公開サイト**: https://yamadan96.github.io/paper-survey/

## ディレクトリ構成

```
content/
├── index.md                      # ランディングページ
├── papers/
│   └── YYYY/<slug>.md            # 1 論文 1 ノート
├── topics/                       # 分野横断のテーマページ
└── templates/paper_template.md   # 新規ノート用テンプレート
```

## ローカルでプレビュー

```bash
npm install              # 初回のみ
npx quartz build --serve # http://localhost:8080
```

## 論文ノートを追加する

```bash
cp content/templates/paper_template.md content/papers/2026/<slug>.md
$EDITOR content/papers/2026/<slug>.md
git add content/papers/2026/<slug>.md
git commit -m "add: <論文タイトル>"
git push
```

`main` への push で `.github/workflows/deploy.yml` が走り、GitHub Pages にデプロイされる。

## ノート記述ルール

- Wikilink: `[[papers/2026/slug]]`、`[[topics/llm-inference-efficiency]]`
- `tags:` には `llm` / `vlm` / `mlops` / `rl` などを英語スラッグで。
- `status:` は `reading` | `read` | `skim` のいずれか。
- スラッグと frontmatter キーは英語（URL・プラグイン都合）。本文は日本語で OK。

## ライセンス

ノート本文: CC BY 4.0。Quartz 本体: MIT（`LICENSE.txt`）。
