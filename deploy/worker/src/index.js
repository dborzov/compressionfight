// compressionfight-proxy
//
// Serves https://borzov.ca/compressionfight/* from the GitHub Pages site
// https://dborzov.github.io/compressionfight/*. GitHub Pages cannot serve a
// sub-path of a custom domain, so this Worker sits on the route
// borzov.ca/compressionfight* and proxies.

const ORIGIN = 'https://dborzov.github.io';
const PREFIX = '/compressionfight';

// Hop-by-hop / origin-specific request headers we do not forward.
const DROP_REQUEST_HEADERS = ['host', 'cf-connecting-ip', 'cf-ray', 'cf-visitor', 'cf-ipcountry', 'x-forwarded-proto', 'x-real-ip'];

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Only handle the game path; anything else on this route is a 404.
    if (url.pathname !== PREFIX && !url.pathname.startsWith(PREFIX + '/')) {
      return new Response('Not found', { status: 404 });
    }

    // /compressionfight -> /compressionfight/ so relative asset paths resolve.
    if (url.pathname === PREFIX) {
      url.pathname = PREFIX + '/';
      return Response.redirect(url.toString(), 301);
    }

    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method not allowed', { status: 405, headers: { allow: 'GET, HEAD' } });
    }

    // Same path on the GitHub Pages origin (the repo is served under /compressionfight/ there too).
    const upstream = new URL(ORIGIN + url.pathname + url.search);

    const headers = new Headers(request.headers);
    for (const h of DROP_REQUEST_HEADERS) headers.delete(h);

    const upstreamResponse = await fetch(upstream.toString(), {
      method: request.method,
      headers,
      redirect: 'manual',
      cf: { cacheEverything: true, cacheTtl: 300 },
    });

    // Pass the response through, keeping GitHub's caching headers (Cache-Control, ETag, Last-Modified)
    // and rewriting any Location header that points back at the origin host.
    const respHeaders = new Headers(upstreamResponse.headers);
    const loc = respHeaders.get('location');
    if (loc && loc.startsWith(ORIGIN)) {
      respHeaders.set('location', loc.slice(ORIGIN.length));
    }
    respHeaders.set('x-proxied-from', upstream.host);

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: respHeaders,
    });
  },
};
