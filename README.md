# paper-survey

Personal paper survey notes, published with [Quartz v4](https://quartz.jzhao.xyz/).

**Live site:** https://yamadan96.github.io/paper-survey/

## Layout

```
content/
├── index.md                      # landing page
├── papers/
│   └── YYYY/<slug>.md            # one note per paper
├── topics/                       # cross-paper theme pages
└── templates/paper_template.md   # copy this to start a new paper note
```

## Local development

```bash
npm install           # first time only
npx quartz build --serve
```

Serves at `http://localhost:8080`.

## Adding a paper

```bash
cp content/templates/paper_template.md content/papers/2026/my-slug.md
$EDITOR content/papers/2026/my-slug.md
git add content/papers/2026/my-slug.md
git commit -m "add: <paper title>"
git push
```

Pushing to `main` triggers the GitHub Actions workflow in `.github/workflows/deploy.yml`, which rebuilds the site and deploys to GitHub Pages.

## Conventions

- Wikilinks: `[[papers/2026/slug]]`, `[[topics/llm-inference-efficiency]]`
- Tags in frontmatter: `tags: [llm, vlm, mlops, rl, ...]`
- Status in frontmatter: `status: reading | read | skim`

## License

Notes: CC BY 4.0. Quartz itself: MIT (see `LICENSE.txt`).
