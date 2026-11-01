// Choose Your Codec: canvas renderers for the engine's generic Viz descriptors.
// CYC.Viz.draw(canvas, viz [, opts]) draws any descriptor into the canvas, sized to the
// canvas's CSS box, devicePixelRatio aware. Colours come from the design tokens (CSS custom
// properties --viz-* on :root) and the stage colour (--rc) resolved on the canvas element:
// paper-2 background, ink text, stage colours for marks, hairline grid at 40% alpha.
// Lines are anti-aliased; only data planes (image pixels, coefficient cells) stay crisp.

window.CYC = window.CYC || {};

CYC.Viz = (function () {
  'use strict';

  const scratch = document.createElement('canvas').getContext('2d');

  function normColor(c, fallback) {
    if (!c) return fallback;
    try {
      scratch.fillStyle = '#000';
      scratch.fillStyle = c;
      return scratch.fillStyle; // normalized '#rrggbb' or 'rgba(...)'
    } catch (e) { return fallback; }
  }

  function toRgb(color) {
    const m = /^#([0-9a-f]{6})$/i.exec(color || '');
    if (m) { const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    const r = /rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/.exec(color || '');
    return r ? [+r[1], +r[2], +r[3]] : [42, 33, 27];
  }
  function withAlpha(color, a) { const [r, g, b] = toRgb(color); return `rgba(${r},${g},${b},${a})`; }
  function mix(c1, c2, k) {
    const a = toRgb(c1), b = toRgb(c2);
    const q = Math.max(0, Math.min(1, k));
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * q)},${Math.round(a[1] + (b[1] - a[1]) * q)},${Math.round(a[2] + (b[2] - a[2]) * q)})`;
  }

  function tokens(canvas, opts) {
    const cs = getComputedStyle(canvas);
    const root = getComputedStyle(document.documentElement);
    const get = (name, fallback) => {
      const v = (cs.getPropertyValue(name) || root.getPropertyValue(name) || '').trim();
      return v || fallback;
    };
    const accent = (opts && opts.color) || get('--rc', get('--teal', '#246b66'));
    return {
      bg: normColor(get('--viz-bg', '#ebdfc8'), '#ebdfc8'),
      fg: normColor(get('--viz-fg', '#2a211b'), '#2a211b'),
      dim: normColor(get('--viz-dim', '#5a4a3d'), '#5a4a3d'),
      match: normColor(get('--viz-match', '#6f8f4e'), '#6f8f4e'),
      dict: normColor(get('--viz-dict', '#b8892e'), '#b8892e'),
      run: normColor(get('--viz-run', '#6b4a7a'), '#6b4a7a'),
      grid: normColor(get('--viz-grid', 'rgba(184,159,120,.4)'), 'rgba(184,159,120,.4)'),
      line: normColor(get('--viz-line', '#b89f78'), '#b89f78'),
      paper: normColor(get('--viz-paper', '#f3eadb'), '#f3eadb'),
      track: normColor(get('--viz-track', '#e0d2b6'), '#e0d2b6'),
      mark: normColor(get('--viz-mark', '#b05a4c'), '#b05a4c'),
      teal: normColor(get('--teal', '#246b66'), '#246b66'),
      sage: normColor(get('--sage', '#6f8f4e'), '#6f8f4e'),
      rose: normColor(get('--rose', '#b05a4c'), '#b05a4c'),
      plum: normColor(get('--plum', '#6b4a7a'), '#6b4a7a'),
      copper: normColor(get('--copper', '#9c6b3f'), '#9c6b3f'),
      gold: normColor(get('--gold', '#b8892e'), '#b8892e'),
      accent: normColor(accent, '#246b66'),
      font: get('--viz-font', '"JetBrains Mono", monospace').replace(/^["']|["']$/g, ''),
    };
  }

  function setup(canvas) {
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(32, Math.round(rect.width || canvas.clientWidth || canvas.width || 320));
    const h = Math.max(32, Math.round(rect.height || canvas.clientHeight || canvas.height || 150));
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (canvas.width !== pw) canvas.width = pw;
    if (canvas.height !== ph) canvas.height = ph;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    return { ctx, w, h, dpr };
  }

  function fmtBits(bits) {
    if (!isFinite(bits)) return '?';
    const bytes = bits / 8;
    if (bytes < 1024) return `${Math.round(bytes)} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10240 ? 2 : 1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
  function fmtBytes(bytes) { return fmtBits(bytes * 8); }

  // small text label on a translucent paper slip, so it stays legible over marks
  function label(ctx, text, x, y, color, font, align, t) {
    ctx.font = font;
    ctx.textAlign = align || 'left';
    ctx.textBaseline = 'top';
    const w = ctx.measureText(text).width;
    const bx = align === 'right' ? x - w - 4 : align === 'center' ? x - w / 2 - 4 : x - 2;
    ctx.fillStyle = withAlpha(t ? t.paper : '#f3eadb', .82);
    ctx.fillRect(bx, y - 1, w + 6, parseInt(font, 10) + 3);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    ctx.textAlign = 'left';
  }

  function hairline(ctx, x1, y1, x2, y2, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }

  function clear(ctx, w, h, t) {
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
  }

  function empty(ctx, w, h, t, msg) {
    clear(ctx, w, h, t);
    label(ctx, msg || 'no visualisation', 6, 6, t.dim, `13px ${t.font}`, 'left', t);
  }

  // ---------- text ----------
  function drawText(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const fs = 13;
    const font = `${fs}px ${t.font}`;
    ctx.font = font;
    ctx.textBaseline = 'top';
    const cw = Math.max(5, ctx.measureText('M').width);
    const lh = fs + 3;
    const pad = 6;
    const cols = Math.max(8, Math.floor((w - pad * 2) / cw));
    const rows = Math.max(1, Math.floor((h - pad * 2) / lh));
    const tintFor = (cls) => cls === 'match' ? t.match : cls === 'dict' ? t.dict : cls === 'run' ? t.run : null;

    let col = 0, row = 0, truncated = false;
    const spans = Array.isArray(viz.spans) ? viz.spans : [];
    outer:
    for (const s of spans) {
      const text = String(s.text == null ? '' : s.text).replace(/\r/g, '');
      const cls = s.cls || 'lit';
      const tint = tintFor(cls);
      for (const ch of text) {
        if (ch === '\n') { col = 0; row++; if (row >= rows) { truncated = true; break outer; } continue; }
        if (col >= cols) { col = 0; row++; if (row >= rows) { truncated = true; break outer; } }
        const x = pad + col * cw, y = pad + row * lh;
        if (tint) {
          ctx.fillStyle = withAlpha(tint, .30);
          ctx.fillRect(x, y - 1, Math.ceil(cw), lh);
        }
        ctx.fillStyle = tint ? t.fg : t.dim;
        const glyph = (ch === '\t' || ch === ' ') ? (tint ? '·' : '') : (ch.charCodeAt(0) < 32 ? '¤' : ch);
        if (glyph) ctx.fillText(glyph, x, y);
        col++;
      }
    }
    if (truncated) label(ctx, '…', w - 14, h - lh - 3, t.fg, font, 'left', t);
  }

  // ---------- waveform ----------
  function drawWaveform(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const series = Array.isArray(viz.series) ? viz.series : [];
    if (!series.length) return empty(ctx, w, h, t, 'no samples');
    const n = series.length;
    const bandH = h / n;
    const palette = [t.accent, t.sage, t.plum, t.copper, t.teal];
    series.forEach((arr, i) => {
      const top = i * bandH;
      const mid = top + bandH / 2;
      hairline(ctx, 0, Math.round(mid) + .5, w, Math.round(mid) + .5, t.grid);
      if (i > 0) hairline(ctx, 0, Math.round(top) + .5, w, Math.round(top) + .5, t.line);
      const len = arr.length;
      if (!len) return;
      const color = palette[i % palette.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      const amp = (bandH / 2) * 0.86;
      const step = Math.max(1, Math.floor(len / (w * 2)));
      for (let k = 0, px = 0; k < len; k += step, px++) {
        const v = Math.max(-1, Math.min(1, arr[k] || 0));
        const x = (k / Math.max(1, len - 1)) * (w - 1);
        const y = mid - v * amp;
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      const lbl = (viz.labels && viz.labels[i]) || `series ${i + 1}`;
      label(ctx, lbl, 6, top + 4, t.fg, `12px ${t.font}`, 'left', t);
      ctx.fillStyle = color; ctx.fillRect(2, top + 4, 2, 13);
    });
  }

  // ---------- grid (planes / rgb) ----------
  function planeToCanvas(plane) {
    const w = plane.w | 0, h = plane.h | 0;
    if (w <= 0 || h <= 0) return null;
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    const img = cx.createImageData(w, h);
    const d = img.data, v = plane.values;
    for (let i = 0, n = w * h; i < n; i++) {
      const g = Math.max(0, Math.min(255, Math.round((v && v[i] != null ? v[i] : 0) * 255)));
      d[i * 4] = g; d[i * 4 + 1] = g; d[i * 4 + 2] = g; d[i * 4 + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    return c;
  }
  function rgbToCanvas(rgb) {
    const w = rgb.w | 0, h = rgb.h | 0;
    if (w <= 0 || h <= 0 || !rgb.data) return null;
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    const img = cx.createImageData(w, h);
    const d = img.data, src = rgb.data;
    const perPx = src.length >= w * h * 4 ? 4 : 3;
    for (let i = 0, n = w * h; i < n; i++) {
      d[i * 4] = src[i * perPx]; d[i * 4 + 1] = src[i * perPx + 1]; d[i * 4 + 2] = src[i * perPx + 2]; d[i * 4 + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    return c;
  }

  // frame a tile like the portraits: cream mat + gold hairline
  function frame(ctx, t, dx, dy, dw, dh) {
    ctx.fillStyle = t.paper; ctx.fillRect(dx - 4, dy - 4, dw + 8, dh + 8);
    ctx.strokeStyle = t.gold; ctx.lineWidth = 1;
    ctx.strokeRect(dx - 4.5, dy - 4.5, dw + 9, dh + 9);
  }

  function drawGrid(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const tiles = [];
    if (viz.rgb) { const c = rgbToCanvas(viz.rgb); if (c) tiles.push({ name: viz.rgb.name || 'RGB', c }); }
    for (const p of (viz.planes || [])) { const c = planeToCanvas(p); if (c) tiles.push({ name: p.name || '', c }); }
    if (!tiles.length) return empty(ctx, w, h, t, 'no planes');
    const gap = 10, pad = 8, labelH = 18;
    const availW = (w - pad * 2 - gap * (tiles.length - 1)) / tiles.length - 8;
    const availH = h - pad * 2 - labelH - 8;
    let x = pad + 4;
    for (const tile of tiles) {
      const sx = availW / tile.c.width, sy = availH / tile.c.height;
      let s = Math.min(sx, sy);
      if (s >= 1) s = Math.floor(s);
      const dw = Math.max(1, Math.round(tile.c.width * s)), dh = Math.max(1, Math.round(tile.c.height * s));
      const dx = Math.round(x + (availW - dw) / 2), dy = Math.round(pad + labelH + 4 + (availH - dh) / 2);
      frame(ctx, t, dx, dy, dw, dh);
      ctx.imageSmoothingEnabled = s < 1;      // smooth when shrinking, crisp pixels when enlarging
      ctx.drawImage(tile.c, dx, dy, dw, dh);
      ctx.imageSmoothingEnabled = true;
      label(ctx, `${tile.name} ${tile.c.width}×${tile.c.height}`.trim(), x + availW / 2, pad, t.fg, `12px ${t.font}`, 'center', t);
      x += availW + gap + 8;
    }
  }

  // ---------- heatmap ----------
  function drawHeatmap(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const gw = viz.w | 0, gh = viz.h | 0;
    if (gw <= 0 || gh <= 0 || !viz.values) return empty(ctx, w, h, t, 'no coefficients');
    const zero = toRgb(t.track);                 // exact zero: sunken paper
    const lo = toRgb(mix(t.paper, t.accent, .25)); // small magnitude: a hint of the stage colour
    const hi = toRgb(t.accent);                  // large magnitude: full stage colour
    const top = toRgb(t.fg);                     // the very largest: towards ink
    const off = document.createElement('canvas');
    off.width = gw; off.height = gh;
    const ox = off.getContext('2d');
    const img = ox.createImageData(gw, gh);
    const d = img.data, v = viz.values;
    for (let i = 0, n = gw * gh; i < n; i++) {
      const val = v[i] || 0;
      let r, g, b;
      if (val === 0) { [r, g, b] = zero; }
      else {
        const m = Math.max(0, Math.min(1, val));
        if (m < 0.85) {
          const q = m / 0.85;
          r = lo[0] + (hi[0] - lo[0]) * q; g = lo[1] + (hi[1] - lo[1]) * q; b = lo[2] + (hi[2] - lo[2]) * q;
        } else {
          const q = (m - 0.85) / 0.15 * .7;
          r = hi[0] + (top[0] - hi[0]) * q; g = hi[1] + (top[1] - hi[1]) * q; b = hi[2] + (top[2] - hi[2]) * q;
        }
      }
      d[i * 4] = Math.round(r); d[i * 4 + 1] = Math.round(g); d[i * 4 + 2] = Math.round(b); d[i * 4 + 3] = 255;
    }
    ox.putImageData(img, 0, 0);
    const pad = 8;
    let s = Math.min((w - pad * 2) / gw, (h - pad * 2) / gh);
    if (s >= 1) s = Math.floor(s);
    const dw = Math.max(1, Math.round(gw * s)), dh = Math.max(1, Math.round(gh * s));
    const dx = Math.round((w - dw) / 2), dy = Math.round((h - dh) / 2);
    ctx.imageSmoothingEnabled = false;          // coefficient cells are data: keep them crisp
    ctx.drawImage(off, dx, dy, dw, dh);
    ctx.imageSmoothingEnabled = true;
    if (viz.blockSize && viz.blockSize > 1 && s * viz.blockSize >= 6) {
      ctx.strokeStyle = withAlpha(t.paper, .55);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let bx = 0; bx <= gw; bx += viz.blockSize) { const x = dx + Math.round(bx * s) + .5; ctx.moveTo(x, dy); ctx.lineTo(x, dy + dh); }
      for (let by = 0; by <= gh; by += viz.blockSize) { const y = dy + Math.round(by * s) + .5; ctx.moveTo(dx, y); ctx.lineTo(dx + dw, y); }
      ctx.stroke();
    }
    ctx.strokeStyle = t.line; ctx.lineWidth = 1; ctx.strokeRect(dx - .5, dy - .5, dw + 1, dh + 1);
    if (typeof viz.zeroFrac === 'number') {
      label(ctx, `${Math.round(viz.zeroFrac * 100)}% zeros`, w - 6, 5, t.fg, `12px ${t.font}`, 'right', t);
    }
    label(ctx, `${gw}×${gh}`, 6, 5, t.dim, `12px ${t.font}`, 'left', t);
  }

  // ---------- histogram ----------
  function drawHistogram(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const bins = Array.isArray(viz.bins) ? viz.bins : [];
    if (!bins.length) return empty(ctx, w, h, t, 'no symbols');
    const pad = 8, topH = 20, bottomH = (viz.labels && viz.labels.length) ? 16 : 6;
    const plotH = h - pad - topH - bottomH;
    const n = bins.length;
    const max = Math.max(1e-9, ...bins.map(b => +b || 0));
    const bw = (w - pad * 2) / n;
    const gap = bw > 4 ? 1 : 0;
    // faint horizontal guides at 1/2 and full height
    hairline(ctx, pad, topH + .5, w - pad, topH + .5, t.grid);
    hairline(ctx, pad, Math.round(topH + plotH / 2) + .5, w - pad, Math.round(topH + plotH / 2) + .5, t.grid);
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, +bins[i] || 0);
      const bh = Math.round((v / max) * plotH);
      const x = Math.round(pad + i * bw), y = topH + plotH - bh;
      ctx.fillStyle = v === 0 ? t.line : t.accent;
      ctx.fillRect(x, y, Math.max(1, Math.floor(bw) - gap), Math.max(v === 0 ? 1 : 2, bh));
    }
    hairline(ctx, pad, topH + plotH + .5, w - pad, topH + plotH + .5, t.line);
    if (viz.labels && viz.labels.length && bw >= 11 && n <= 40) {
      ctx.font = `10px ${t.font}`; ctx.fillStyle = t.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let i = 0; i < n; i++) {
        const l = String(viz.labels[i] == null ? '' : viz.labels[i]);
        ctx.fillText(l.length > 4 ? l.slice(0, 4) : l, pad + i * bw + bw / 2, topH + plotH + 4);
      }
      ctx.textAlign = 'left';
    } else if (viz.labels && viz.labels.length) {
      label(ctx, `${viz.labels[0]} … ${viz.labels[viz.labels.length - 1]}`, w / 2, topH + plotH + 3, t.dim, `10px ${t.font}`, 'center', t);
    }
    const H = typeof viz.entropyBitsPerSymbol === 'number' ? viz.entropyBitsPerSymbol : null;
    const F = typeof viz.fixedBits === 'number' ? viz.fixedBits : null;
    const txt = `${n} symbols` + (H != null ? `  H = ${H.toFixed(2)} bits/sym` : '') + (F != null ? `  fixed = ${F} bits` : '');
    label(ctx, txt, 6, 4, t.fg, `12px ${t.font}`, 'left', t);
  }

  // ---------- scatter ----------
  function drawScatter(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const pts = viz.points;
    if (!pts || pts.length < 2) return empty(ctx, w, h, t, 'no points');
    const pad = 10, size = Math.min(w, h) - pad * 2;
    const ox = (w - size) / 2, oy = (h - size) / 2;
    const toX = (x) => ox + (Math.max(-1, Math.min(1, x)) + 1) / 2 * size;
    const toY = (y) => oy + (1 - (Math.max(-1, Math.min(1, y)) + 1) / 2) * size;
    ctx.strokeStyle = t.line; ctx.lineWidth = 1;
    ctx.strokeRect(ox + .5, oy + .5, size - 1, size - 1);
    hairline(ctx, Math.round(toX(0)) + .5, oy, Math.round(toX(0)) + .5, oy + size, t.grid);
    hairline(ctx, ox, Math.round(toY(0)) + .5, ox + size, Math.round(toY(0)) + .5, t.grid);
    const cents = viz.centroids;
    const k = cents ? Math.floor(cents.length / 2) : 0;
    const asg = viz.assignments;
    const n = Math.floor(pts.length / 2);
    const palette = [t.teal, t.sage, t.rose, t.plum, t.copper, t.gold];
    const colorOf = (c) => palette[c % palette.length];
    for (let i = 0; i < n; i++) {
      const x = toX(pts[i * 2]), y = toY(pts[i * 2 + 1]);
      ctx.fillStyle = (asg && k) ? withAlpha(colorOf(asg[i] % k), .75) : withAlpha(t.fg, .7);
      ctx.beginPath(); ctx.arc(x, y, 1.6, 0, Math.PI * 2); ctx.fill();
    }
    if (cents && k) {
      for (let c = 0; c < k; c++) {
        const x = toX(cents[c * 2]), y = toY(cents[c * 2 + 1]);
        ctx.fillStyle = asg ? colorOf(c) : t.accent;
        ctx.strokeStyle = t.paper; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = t.paper; ctx.beginPath(); ctx.arc(x, y, 1.3, 0, Math.PI * 2); ctx.fill();
      }
    }
    label(ctx, `${n} pts` + (k ? `  ${k} centroids` : ''), 6, 4, t.fg, `12px ${t.font}`, 'left', t);
  }

  // ---------- bits ----------
  function drawBits(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const raw = +viz.raw || 0, bound = +viz.bound || 0, out = +viz.out || 0;
    const rows = [
      { name: 'raw', v: raw, color: t.line },
      { name: 'bound', v: bound, color: t.mark },
      { name: 'out', v: out, color: t.accent },
    ];
    const max = Math.max(1, ...rows.map(r => r.v));
    const pad = 8, labelW = 48, valW = 74;
    const barW = Math.max(10, w - pad * 2 - labelW - valW);
    const rowH = Math.min(30, (h - pad * 2 - 18) / 3);
    ctx.font = `12px ${t.font}`;
    rows.forEach((r, i) => {
      const y = pad + i * rowH;
      ctx.fillStyle = t.dim; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.fillText(r.name, pad, y + rowH / 2 - 7);
      // track with a gold hairline
      ctx.fillStyle = t.track; ctx.fillRect(pad + labelW, y + 4, barW, rowH - 8);
      ctx.strokeStyle = t.gold; ctx.lineWidth = 1; ctx.strokeRect(pad + labelW + .5, y + 4.5, barW - 1, rowH - 9);
      const fw = Math.max(r.v > 0 ? 2 : 0, Math.round((r.v / max) * barW));
      if (r.name === 'bound') {
        // the entropy bound is a marker, not a fill
        ctx.fillStyle = withAlpha(r.color, .22); ctx.fillRect(pad + labelW, y + 4, fw, rowH - 8);
        ctx.fillStyle = r.color; ctx.fillRect(pad + labelW + fw - 1, y + 1, 2, rowH - 2);
      } else {
        ctx.fillStyle = r.color; ctx.fillRect(pad + labelW, y + 4, fw, rowH - 8);
      }
      ctx.fillStyle = t.fg; ctx.textAlign = 'right';
      ctx.fillText(fmtBits(r.v), w - pad, y + rowH / 2 - 7);
    });
    // bound marker over the output bar too
    if (bound > 0) {
      const bx = pad + labelW + Math.round((bound / max) * barW);
      ctx.fillStyle = t.mark; ctx.fillRect(bx - 1, pad + 2 * rowH + 1, 2, rowH - 2);
    }
    ctx.textAlign = 'left';
    const ratio = out > 0 ? (raw / out) : 0;
    const side = +viz.sideInfoBytes || 0;
    label(ctx, `ratio ×${ratio ? ratio.toFixed(2) : '—'}   side info ${side ? fmtBytes(side) : 'none'}`, pad, pad + rowH * 3 + 2, t.dim, `12px ${t.font}`, 'left', t);
  }

  const RENDERERS = {
    text: drawText, waveform: drawWaveform, grid: drawGrid, heatmap: drawHeatmap,
    histogram: drawHistogram, scatter: drawScatter, bits: drawBits,
  };

  function draw(canvas, viz, opts) {
    if (!canvas || !canvas.getContext) return;
    const { ctx, w, h } = setup(canvas);
    const t = tokens(canvas, opts);
    try {
      if (!viz || !RENDERERS[viz.type]) return empty(ctx, w, h, t, viz && viz.type ? `unknown viz: ${viz.type}` : 'no visualisation');
      RENDERERS[viz.type](ctx, w, h, t, viz);
    } catch (e) {
      empty(ctx, w, h, t, 'viz error: ' + (e && e.message ? e.message : e));
      if (window.console) console.error('CYC.Viz', e);
    }
  }

  return { draw, fmtBits, fmtBytes, TYPES: Object.keys(RENDERERS) };
})();
