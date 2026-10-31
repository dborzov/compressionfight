# Deployment

## Where the game lives

- **GitHub Pages (live):** https://www.borzov.ca/compressionfight/ — serves the repo root of `main`
  (legacy Pages build, `.nojekyll` present). Every push to `main` is a deploy; there is nothing to build.
- **Why that URL:** the user site repo `dborzov/dborzov.github.io` has the custom domain `www.borzov.ca`, and
  GitHub serves every project page of the account under the user site's domain. So
  `https://dborzov.github.io/compressionfight/` 301s to `https://www.borzov.ca/compressionfight/`, and the apex
  `https://borzov.ca/compressionfight/` 301s there too via the existing apex-to-www redirect on Cloudflare.
  **The goal "reachable at borzov.ca/compressionfight" is therefore already met without any Worker.**
- **Worker (fallback, not deployed):** `deploy/worker/` is a Cloudflare Worker that would serve
  `borzov.ca/compressionfight*` directly on the apex (no redirect hop) by proxying the Pages origin. Keep it
  only if the apex redirect ever goes away or a redirect-free apex URL is wanted.

## The worker

`deploy/worker/src/index.js` handles `borzov.ca/compressionfight*`:

- `/compressionfight` (no trailing slash) → 301 to `/compressionfight/`, so relative asset paths resolve;
- `/compressionfight/<rest>` → fetches `https://dborzov.github.io/compressionfight/<rest>` with the request
  method (GET/HEAD) and headers, caches at the edge for 5 minutes, and returns the response with GitHub's
  `Cache-Control` / `ETag` / `Last-Modified` headers passed through;
- anything else → 404.

`deploy/worker/wrangler.jsonc` names the worker `compressionfight-proxy`; its `routes` entry for
`borzov.ca/compressionfight*` (zone `e95e36b1ce9a27495c67b7613be83b1a`) is **commented out** so nothing on
borzov.ca changes until someone decides to go live.

### Going live (two steps)

1. In `deploy/worker/wrangler.jsonc`, uncomment the `routes` block.
2. From `deploy/worker/`, run `npx wrangler deploy`.

A Worker route only fires on traffic that passes through Cloudflare; the apex already is proxied (it answers
with `server: cloudflare` and the apex-to-www 301), so no DNS change is needed. The Worker route takes
precedence over the redirect rule for that path only.

Check: `curl -sI https://borzov.ca/compressionfight/` should return `200` with `x-proxied-from: dborzov.github.io`,
and `curl -sI https://borzov.ca/compressionfight` should return `301` with `location: .../compressionfight/`.

### Rolling back

- Remove just the route (the worker stays deployed, borzov.ca stops going through it):
  re-comment the `routes` block and run `npx wrangler deploy` again, or delete the route in the dashboard
  (Workers & Pages → compressionfight-proxy → Settings → Domains & Routes).
- Remove the worker entirely: `npx wrangler delete` from `deploy/worker/`.
- Previous versions: `npx wrangler versions list` then `npx wrangler rollback`.

Local test without touching production: `npx wrangler dev` in `deploy/worker/`, then open
`http://localhost:8787/compressionfight/`.

## Auth status (checked 2026-10-31)

| Tool | Status |
|---|---|
| `gh` | logged in as `dborzov` (ssh, scopes repo / read:org / gist / admin:public_key) |
| `npx wrangler whoami` | logged in via OAuth as tihoutrom@gmail.com, account `d666b77c4126b3b31c76ecb8a8fa098c` |
| `cf auth whoami` | logged in via OAuth as tihoutrom@gmail.com, same account |

Either `wrangler` or `cf` can deploy the worker; this README uses `wrangler` because the project has a
`wrangler.jsonc`. No deploy, DNS or route change has been made yet.
