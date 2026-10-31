// Choose Your Codec: canvas renderers for the engine's generic Viz descriptors.
// CYC.Viz.draw(canvas, viz [, opts]) draws any descriptor into the canvas, sized to the
// canvas's CSS box, devicePixelRatio aware, pixel look. Colors come from CSS custom
// properties (--viz-*) and the role color (--rc) resolved on the canvas element.

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

  function hexToRgb(hex) {
    const m = /^#([0-9a-f]{6})$/i.exec(hex || '');
    if (!m) {
      const r = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(hex || '');
      return r ? [+r[1], +r[2], +r[3]] : [255, 43, 214];
    }
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function tokens(canvas, opts) {
    const cs = getComputedStyle(canvas);
    const root = getComputedStyle(document.documentElement);
    const get = (name, fallback) => {
      const v = (cs.getPropertyValue(name) || root.getPropertyValue(name) || '').trim();
      return v || fallback;
    };
    const accent = (opts && opts.color) || get('--rc', '#ff2bd6');
    return {
      bg: normColor(get('--viz-bg', '#07040f'), '#07040f'),
      fg: normColor(get('--viz-fg', '#f6eeff'), '#f6eeff'),
      dim: normColor(get('--viz-dim', '#5c4a80'), '#5c4a80'),
      match: normColor(get('--viz-match', '#4dff7c'), '#4dff7c'),
      dict: normColor(get('--viz-dict', '#ffd23d'), '#ffd23d'),
      run: normColor(get('--viz-run', '#d05cff'), '#d05cff'),
      grid: normColor(get('--viz-grid', '#2a1b4a'), '#2a1b4a'),
      accent: normColor(accent, '#ff2bd6'),
      font: get('--viz-font', '"VT323", monospace').replace(/^["']|["']$/g, ''),
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
    ctx.imageSmoothingEnabled = false;
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

  function label(ctx, text, x, y, color, font, align) {
    ctx.font = font;
    ctx.textAlign = align || 'left';
    ctx.textBaseline = 'top';
    const w = ctx.measureText(text).width;
    const bx = align === 'right' ? x - w - 4 : align === 'center' ? x - w / 2 - 4 : x - 2;
    ctx.fillStyle = 'rgba(0,0,0,.72)';
    ctx.fillRect(bx, y - 1, w + 6, parseInt(font, 10) + 2);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    ctx.textAlign = 'left';
  }

  function clear(ctx, w, h, t) {
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
  }

  function empty(ctx, w, h, t, msg) {
    clear(ctx, w, h, t);
    label(ctx, msg || 'no viz', 6, 6, t.dim, `14px ${t.font}`);
  }

  // ---------- text ----------
  function drawText(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const fs = 15;
    const font = `${fs}px ${t.font}`;
    ctx.font = font;
    ctx.textBaseline = 'top';
    const cw = Math.max(5, ctx.measureText('M').width);
    const lh = fs + 1;
    const pad = 5;
    const cols = Math.max(8, Math.floor((w - pad * 2) / cw));
    const rows = Math.max(1, Math.floor((h - pad * 2) / lh));
    const colorFor = (cls) => cls === 'match' ? t.match : cls === 'dict' ? t.dict : cls === 'run' ? t.run : t.dim;
    const bgFor = (cls) => cls === 'match' ? t.match : cls === 'dict' ? t.dict : cls === 'run' ? t.run : null;

    let col = 0, row = 0, truncated = false;
    const spans = Array.isArray(viz.spans) ? viz.spans : [];
    outer:
    for (const s of spans) {
      const text = String(s.text == null ? '' : s.text).replace(/\r/g, '');
      const cls = s.cls || 'lit';
      for (const ch of text) {
        if (ch === '\n') { col = 0; row++; if (row >= rows) { truncated = true; break outer; } continue; }
        if (col >= cols) { col = 0; row++; if (row >= rows) { truncated = true; break outer; } }
        const x = pad + col * cw, y = pad + row * lh;
        const bg = bgFor(cls);
        if (bg) {
          ctx.globalAlpha = 0.28;
          ctx.fillStyle = bg;
          ctx.fillRect(x, y, Math.ceil(cw), lh);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = cls === 'lit' ? t.dim : colorFor(cls);
        if (cls !== 'lit') ctx.fillStyle = cls === 'match' ? '#dfffe8' : cls === 'dict' ? '#fff3c4' : '#f3dcff';
        const glyph = (ch === '\t' || ch === ' ') ? (cls === 'lit' ? '' : '·') : (ch.charCodeAt(0) < 32 ? '¤' : ch);
        if (glyph) ctx.fillText(glyph, x, y);
        col++;
      }
    }
    if (truncated) label(ctx, '…', w - 14, h - lh - 3, t.fg, font);
  }

  // ---------- waveform ----------
  function drawWaveform(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const series = Array.isArray(viz.series) ? viz.series : [];
    if (!series.length) return empty(ctx, w, h, t, 'no samples');
    const n = series.length;
    const bandH = h / n;
    const palette = [t.accent, t.match, t.dict, t.run, t.fg];
    series.forEach((arr, i) => {
      const top = i * bandH;
      const mid = top + bandH / 2;
      ctx.strokeStyle = t.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Math.round(mid) + .5); ctx.lineTo(w, Math.round(mid) + .5); ctx.stroke();
      if (i > 0) { ctx.beginPath(); ctx.moveTo(0, Math.round(top) + .5); ctx.lineTo(w, Math.round(top) + .5); ctx.stroke(); }
      const len = arr.length;
      if (!len) return;
      ctx.strokeStyle = palette[i % palette.length];
      ctx.lineWidth = 1.5;
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
      label(ctx, lbl, 5, top + 3, palette[i % palette.length], `13px ${t.font}`);
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

  function drawGrid(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const tiles = [];
    if (viz.rgb) { const c = rgbToCanvas(viz.rgb); if (c) tiles.push({ name: viz.rgb.name || 'RGB', c }); }
    for (const p of (viz.planes || [])) { const c = planeToCanvas(p); if (c) tiles.push({ name: p.name || '', c }); }
    if (!tiles.length) return empty(ctx, w, h, t, 'no planes');
    const gap = 8, pad = 6, labelH = 16;
    const availW = (w - pad * 2 - gap * (tiles.length - 1)) / tiles.length;
    const availH = h - pad * 2 - labelH;
    let x = pad;
    for (const tile of tiles) {
      const sx = availW / tile.c.width, sy = availH / tile.c.height;
      let s = Math.min(sx, sy);
      if (s >= 1) s = Math.floor(s);
      const dw = Math.max(1, Math.round(tile.c.width * s)), dh = Math.max(1, Math.round(tile.c.height * s));
      const dx = Math.round(x + (availW - dw) / 2), dy = Math.round(pad + labelH + (availH - dh) / 2);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tile.c, dx, dy, dw, dh);
      ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
      ctx.strokeRect(dx - .5, dy - .5, dw + 1, dh + 1);
      label(ctx, `${tile.name} ${tile.c.width}×${tile.c.height}`.trim(), x + availW / 2, pad, t.fg, `13px ${t.font}`, 'center');
      x += availW + gap;
    }
  }

  // ---------- heatmap ----------
  function drawHeatmap(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const gw = viz.w | 0, gh = viz.h | 0;
    if (gw <= 0 || gh <= 0 || !viz.values) return empty(ctx, w, h, t, 'no coefficients');
    const [ar, ag, ab] = hexToRgb(normColor(t.accent, '#ff2bd6'));
    const off = document.createElement('canvas');
    off.width = gw; off.height = gh;
    const ox = off.getContext('2d');
    const img = ox.createImageData(gw, gh);
    const d = img.data, v = viz.values;
    for (let i = 0, n = gw * gh; i < n; i++) {
      const val = v[i] || 0;
      let r, g, b;
      if (val === 0) { r = 18; g = 10; b = 30; }  // exact zero: dark cell
      else {
        const m = Math.max(0, Math.min(1, val));
        const k = 0.15 + 0.85 * m;
        r = Math.round(ar * k); g = Math.round(ag * k); b = Math.round(ab * k);
        if (m > 0.85) { const q = (m - 0.85) / 0.15; r = Math.round(r + (255 - r) * q * .8); g = Math.round(g + (255 - g) * q * .8); b = Math.round(b + (255 - b) * q * .8); }
      }
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
    }
    ox.putImageData(img, 0, 0);
    const pad = 6;
    let s = Math.min((w - pad * 2) / gw, (h - pad * 2) / gh);
    if (s >= 1) s = Math.floor(s);
    const dw = Math.max(1, Math.round(gw * s)), dh = Math.max(1, Math.round(gh * s));
    const dx = Math.round((w - dw) / 2), dy = Math.round((h - dh) / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, dx, dy, dw, dh);
    if (viz.blockSize && viz.blockSize > 1 && s * viz.blockSize >= 6) {
      ctx.strokeStyle = 'rgba(255,255,255,.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let bx = 0; bx <= gw; bx += viz.blockSize) { const x = dx + Math.round(bx * s) + .5; ctx.moveTo(x, dy); ctx.lineTo(x, dy + dh); }
      for (let by = 0; by <= gh; by += viz.blockSize) { const y = dy + Math.round(by * s) + .5; ctx.moveTo(dx, y); ctx.lineTo(dx + dw, y); }
      ctx.stroke();
    }
    ctx.strokeStyle = t.grid; ctx.strokeRect(dx - .5, dy - .5, dw + 1, dh + 1);
    if (typeof viz.zeroFrac === 'number') {
      label(ctx, `${Math.round(viz.zeroFrac * 100)}% zeros`, w - 6, 5, t.fg, `13px ${t.font}`, 'right');
    }
    label(ctx, `${gw}×${gh}`, 6, 5, t.dim, `13px ${t.font}`);
  }

  // ---------- histogram ----------
  function drawHistogram(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const bins = Array.isArray(viz.bins) ? viz.bins : [];
    if (!bins.length) return empty(ctx, w, h, t, 'no symbols');
    const pad = 6, topH = 18, bottomH = (viz.labels && viz.labels.length) ? 14 : 4;
    const plotH = h - pad - topH - bottomH;
    const n = bins.length;
    const max = Math.max(1e-9, ...bins.map(b => +b || 0));
    const bw = (w - pad * 2) / n;
    const gap = bw > 4 ? 1 : 0;
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, +bins[i] || 0);
      const bh = Math.round((v / max) * plotH);
      const x = Math.round(pad + i * bw), y = topH + plotH - bh;
      ctx.fillStyle = v === 0 ? t.grid : t.accent;
      ctx.fillRect(x, y, Math.max(1, Math.floor(bw) - gap), Math.max(v === 0 ? 1 : 2, bh));
    }
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, topH + plotH + .5); ctx.lineTo(w - pad, topH + plotH + .5); ctx.stroke();
    if (viz.labels && viz.labels.length && bw >= 11 && n <= 40) {
      ctx.font = `11px ${t.font}`; ctx.fillStyle = t.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let i = 0; i < n; i++) {
        const l = String(viz.labels[i] == null ? '' : viz.labels[i]);
        ctx.fillText(l.length > 4 ? l.slice(0, 4) : l, pad + i * bw + bw / 2, topH + plotH + 3);
      }
      ctx.textAlign = 'left';
    } else if (viz.labels && viz.labels.length) {
      label(ctx, `${viz.labels[0]} … ${viz.labels[viz.labels.length - 1]}`, w / 2, topH + plotH + 2, t.dim, `11px ${t.font}`, 'center');
    }
    const H = typeof viz.entropyBitsPerSymbol === 'number' ? viz.entropyBitsPerSymbol : null;
    const F = typeof viz.fixedBits === 'number' ? viz.fixedBits : null;
    const txt = `${n} symbols` + (H != null ? `  H = ${H.toFixed(2)} bits/sym` : '') + (F != null ? `  fixed = ${F} bits` : '');
    label(ctx, txt, 5, 3, t.fg, `13px ${t.font}`);
  }

  // ---------- scatter ----------
  function hsl(i, n, l) { return `hsl(${Math.round((i / Math.max(1, n)) * 360)}, 90%, ${l || 62}%)`; }
  function drawScatter(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const pts = viz.points;
    if (!pts || pts.length < 2) return empty(ctx, w, h, t, 'no points');
    const pad = 8, size = Math.min(w, h) - pad * 2;
    const ox = (w - size) / 2, oy = (h - size) / 2;
    const toX = (x) => ox + (Math.max(-1, Math.min(1, x)) + 1) / 2 * size;
    const toY = (y) => oy + (1 - (Math.max(-1, Math.min(1, y)) + 1) / 2) * size;
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
    ctx.strokeRect(ox + .5, oy + .5, size - 1, size - 1);
    ctx.beginPath(); ctx.moveTo(toX(0) + .5, oy); ctx.lineTo(toX(0) + .5, oy + size); ctx.moveTo(ox, toY(0) + .5); ctx.lineTo(ox + size, toY(0) + .5); ctx.stroke();
    const cents = viz.centroids;
    const k = cents ? Math.floor(cents.length / 2) : 0;
    const asg = viz.assignments;
    const n = Math.floor(pts.length / 2);
    for (let i = 0; i < n; i++) {
      const x = Math.round(toX(pts[i * 2])), y = Math.round(toY(pts[i * 2 + 1]));
      ctx.fillStyle = (asg && k) ? hsl(asg[i] % k, k, 55) : t.dim;
      if (!asg) ctx.fillStyle = t.fg;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
    if (cents && k) {
      for (let c = 0; c < k; c++) {
        const x = Math.round(toX(cents[c * 2])), y = Math.round(toY(cents[c * 2 + 1]));
        ctx.fillStyle = '#000'; ctx.fillRect(x - 5, y - 5, 10, 10);
        ctx.fillStyle = asg ? hsl(c, k, 70) : t.accent; ctx.fillRect(x - 4, y - 4, 8, 8);
        ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, y - 1, 2, 2);
      }
    }
    label(ctx, `${n} pts` + (k ? `  ${k} centroids` : ''), 5, 3, t.fg, `13px ${t.font}`);
  }

  // ---------- bits ----------
  function drawBits(ctx, w, h, t, viz) {
    clear(ctx, w, h, t);
    const rows = [
      { name: 'RAW', v: +viz.raw || 0, color: t.dim },
      { name: 'BOUND', v: +viz.bound || 0, color: t.dict },
      { name: 'OUT', v: +viz.out || 0, color: t.accent },
    ];
    const max = Math.max(1, ...rows.map(r => r.v));
    const pad = 6, labelW = 52, valW = 74;
    const barW = Math.max(10, w - pad * 2 - labelW - valW);
    const rowH = Math.min(30, (h - pad * 2 - 18) / 3);
    ctx.font = `13px ${t.font}`;
    rows.forEach((r, i) => {
      const y = pad + i * rowH;
      ctx.fillStyle = t.fg; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.fillText(r.name, pad, y + rowH / 2 - 7);
      ctx.fillStyle = t.grid;
      ctx.fillRect(pad + labelW, y + 4, barW, rowH - 8);
      const fw = Math.max(r.v > 0 ? 2 : 0, Math.round((r.v / max) * barW));
      ctx.fillStyle = r.color;
      ctx.fillRect(pad + labelW, y + 4, fw, rowH - 8);
      ctx.fillStyle = t.fg; ctx.textAlign = 'right';
      ctx.fillText(fmtBits(r.v), w - pad, y + rowH / 2 - 7);
    });
    ctx.textAlign = 'left';
    const ratio = viz.out > 0 ? (viz.raw / viz.out) : 0;
    const side = +viz.sideInfoBytes || 0;
    label(ctx, `ratio ×${ratio ? ratio.toFixed(2) : '—'}   side info ${side ? fmtBytes(side) : 'none'}`, pad, pad + rowH * 3 + 2, t.dim, `13px ${t.font}`);
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
      if (!viz || !RENDERERS[viz.type]) return empty(ctx, w, h, t, viz && viz.type ? `unknown viz: ${viz.type}` : 'no viz');
      RENDERERS[viz.type](ctx, w, h, t, viz);
    } catch (e) {
      empty(ctx, w, h, t, 'viz error: ' + (e && e.message ? e.message : e));
      if (window.console) console.error('CYC.Viz', e);
    }
  }

  return { draw, fmtBits, fmtBytes, TYPES: Object.keys(RENDERERS) };
})();
