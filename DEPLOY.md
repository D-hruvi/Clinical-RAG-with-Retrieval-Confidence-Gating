# Deploying to Render

This repo includes `render.yaml`, so Render can stand up both services from a
single Blueprint. There's no Docker step involved on Render's side — it uses
native Python builds, which is why `requirements-api.txt` and
`requirements-ui.txt` are split (keeps the UI build fast and cheap).

## Before you deploy

0. **Repo layout matters.** `render.yaml` must sit at your GitHub repo's
   root. If you push this folder as a subfolder of an existing repo instead
   of its own repo root, either move the files up or add `rootDir:
   medrag-llamaindex` to both services in `render.yaml`.
1. **Commit real guideline PDFs** to `data/guidelines/`. They are *not*
   gitignored on purpose — Render's build step reads from the repo, not from
   your local machine. Use public-domain or redistribution-licensed sources
   (WHO, CDC, NIH, NICE, ICMR guidelines are typically fine — check each
   source's terms before committing it to a public repo).
2. Push this repo to GitHub.

## Deploy

1. Render dashboard → **New → Blueprint** → connect the GitHub repo.
2. Render reads `render.yaml` and shows two services: `medrag-api`,
   `medrag-ui`. Click through.
3. You'll be prompted once for `OPENAI_API_KEY`. The UI's `API_URL` is set to
   `https://medrag-api.onrender.com` — **check that name is actually free in
   your account** (Render will tell you during setup if it collided and
   picked something else). If it did, update `API_URL` on the `medrag-ui`
   service in the dashboard after both services exist — no code change or
   redeploy needed, it's read at request time.
4. Click **Apply**. The API service's build step installs dependencies *and*
   builds the vector index in one go — expect the first build to take a few
   minutes (embedding calls to OpenAI happen here, so it also costs a small
   amount of API credit per deploy).
5. Once both services show "Live," open the `medrag-ui` service's URL. That's
   your demo link for your resume/portfolio.

## What to actually expect on Render's free tier

- Free web services **spin down after inactivity** and take 30-60s to wake up
  on the next request — the first query after idle time will be slow. This is
  normal Render free-tier behavior, not a bug in this project; mention it if
  you demo it live so it doesn't look broken.
- The index rebuilds **on every deploy** (that's the trade-off for not paying
  for a persistent disk or a Qdrant Cloud instance). Fine for a portfolio
  project with a small, static guideline set; not how you'd do this in
  production with a large or frequently-updated corpus.
- Rerunning `python -m src.evaluation.run_eval` against the deployed system
  isn't automatic — that's a local step against your local index (or you can
  SSH/shell into the Render instance from the dashboard and run it there).

## If a build fails

- "No documents to index" → you didn't commit any PDFs and the PubMed query
  in `render.yaml`'s `buildCommand` returned nothing usable. Add PDFs or
  change the query term.
- Build timeout on the free plan → your PDF set is too large to embed within
  Render's free build time limit. Trim the guideline set or upgrade the plan.
