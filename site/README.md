# HYPERGAMY promo site

Static Astro site for the HYPERGAMY calculator. It is a marketing/evidence page, not the
calculator: every number on it is copied from the repository's trained/measured output, and
the site deliberately makes no claim the Python tool does not support.

## Run it

```bash
cd site
npm install
npm run dev      # http://localhost:4321/h-AI-pergamy/
npm run build    # static output in site/dist/
npm run preview
```

Astro is configured for the GitHub Pages project URL:

- `site`: `https://thijnperd.github.io`
- `base`: `/h-AI-pergamy`

Internal links and assets must use `import.meta.env.BASE_URL`; a hard-coded `/foo` will 404
under the project path.

## Deploy

`.github/workflows/deploy-pages.yml` builds this directory with `withastro/action@v6` and
deploys the artifact with `actions/deploy-pages@v5`. Pages must be set to **GitHub Actions**
as its source; the workflow runs on pushes to `main` that touch `site/`.

## Content rules

1. A number must be trained, measured, or labelled a research prior — the same rule as the
   root `AGENTS.md`.
2. The partner section is measured ACS context, never a lifetime forecast.
3. Do not add an invented screenshot, quote, user count, or logo.
4. The hero map is generated from `hypergamy_map.matrices()` by
   `python tools/export_site_map.py`; do not hand-edit `src/data/map.json`.
5. OKCupid has no date-outcome column; use it only for descriptive priors, never as a
   training label.

## Icons

Icons come from [Lucide](https://lucide.dev), ISC licensed, via `@lucide/astro`. The
calculator uses the same icon set as inline SVG, with attribution in its footer.
