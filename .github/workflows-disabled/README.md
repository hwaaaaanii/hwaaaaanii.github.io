# Disabled workflows

These came with the Chirpy theme fork and are only useful for theme development
(commit message lint, theme CI, CodeQL, release, stale bot). GitHub only runs
files in `.github/workflows/`, so moving them here turns them off.
To turn one back on, move it back into `.github/workflows/`.
The site itself is deployed by `.github/workflows/jekyll.yml`.
