// Choose Your Codec: screens, modes, levels, catalog, live pipeline panel, results, tutor,
// lessons, quizzes, side quests, explore (codecs / roster / jobs / matrix / duel / trade-offs).
// Depends on CYC (data.js, curriculum.js, quests_data.js). Optional modules are guarded:
// CYC.Engine, CYC.Rules, CYC.Viz, CYC.Demos, CYC.Quests. Missing ones show placeholders.

window.CYC = window.CYC || {};

CYC.UI = (function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const PKEY = 'cyc.progress';
  const TKEY = 'cyc.tutor';
  const ROLE_ORDER = Object.keys(CYC.ROLES);
  const MAX_LEVEL = CYC.LEVELS[CYC.LEVELS.length - 1].id;
  const RANKS = ['D', 'C', 'B', 'A', 'S'];
  const PAYLOADS = ['json', 'prose', 'image', 'audio', 'video', 'vectors', 'weights', 'activations'];
  const STAT_ROWS = [
    ['enc', 'ENC COST'], ['dec', 'DEC COST'], ['ratio', 'SQUEEZE'],
    ['latency', 'DELAY'], ['mem', 'MEMORY'], ['ra', 'RANDOM ACCESS'],
  ];
  const REQ_ROWS = [
    ['lossless', 'LOSSLESS', 'bool'], ['latency', 'LATENCY TOLERANCE'], ['encBudget', 'ENCODER BUDGET'],
    ['decBudget', 'DECODER BUDGET'], ['ra', 'RANDOM ACCESS'], ['bandwidth', 'BANDWIDTH PRESSURE'],
    ['decMem', 'DECODER MEMORY'], ['priorOK', 'CAN SHIP SIDE INFO', 'bool'],
  ];
  const MATRIX_COLS = [
    { label: 'NEEDS LOSSLESS', f: (s) => (s.lossless ? 2 : 0) },
    { label: 'NEEDS LOW LATENCY', f: (s) => (s.latency <= 1 ? 2 : s.latency <= 3 ? 1 : 0) },
    { label: 'CHEAP DECODER', f: (s) => (s.dec <= 1 ? 2 : s.dec <= 3 ? 1 : 0) },
    { label: 'CHEAP ENCODER', f: (s) => (s.enc <= 1 ? 2 : s.enc <= 3 ? 1 : 0) },
    { label: 'RANDOM ACCESS', f: (s) => (s.ra >= 4 ? 2 : s.ra >= 2 ? 1 : 0) },
    { label: 'TIGHT BANDWIDTH', f: (s) => (s.ratio >= 4 ? 2 : s.ratio === 3 ? 1 : 0) },
    { label: 'SMALL DECODER MEMORY', f: (s) => (s.mem <= 1 ? 2 : s.mem <= 3 ? 1 : 0) },
  ];
  const OK_COLOR = { ideal: 'var(--ok-ideal)', accept: 'var(--ok-accept)', meh: 'var(--ok-meh)', wrong: 'var(--ok-wrong)' };
  const EXPLORE_TABS = ['codecs', 'roster', 'jobs', 'matrix', 'duel', 'tradeoffs'];

  const state = {
    mode: 'heist',          // 'heist' | 'explore'
    screen: 'title',
    ctx: 'heist',           // squad-room context: 'heist' (level roles, progression) | 'explore' (all roles)
    level: CYC.LEVELS[0],
    heist: null,
    codec: null,
    squad: {},
    result: null,
    engineError: null,
    progress: loadProgress(),
    tutor: loadTutor(),
    chooser: { open: false, role: null, family: null, focus: null },
    exploreTab: 'codecs',
    heistTab: 'heists',
    lastScore: null,
    pendingQuiz: null,      // level whose checkpoint is due after the results screen
    quiz: null,             // { level, i, picked, correct }
    quest: null,            // { id, step, done, result }
    duel: null,             // { payload, role, a, b, squadBase, lesson, preset }
    tradeoff: { x: 'dec', y: 'ratio', note: null },
    fixer: { queue: [], current: null, line: 0 },
  };
  let runTimer = 0, resizeTimer = 0;

  // ---------------- helpers ----------------
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') n.className = v;
        else if (k === 'style' && typeof v === 'object') {
          for (const [p, val] of Object.entries(v)) { if (p.startsWith('--')) n.style.setProperty(p, val); else n.style[p] = val; }
        }
        else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
        else if (k === 'dataset') Object.assign(n.dataset, v);
        else if (k === 'text') n.textContent = v;
        else if (k === 'html') n.innerHTML = v;   // only for trusted content from our own data files (sub/sup/b)
        else if (v === true) n.setAttribute(k, '');
        else n.setAttribute(k, v);
      }
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return n;
  }
  const clearEl = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
  const engineReady = () => !!(CYC.Engine && typeof CYC.Engine.run === 'function');
  const rulesReady = () => !!(CYC.Rules && typeof CYC.Rules.score === 'function');
  const vizReady = () => !!(CYC.Viz && typeof CYC.Viz.draw === 'function');
  const demosHas = (id) => { try { return !!(CYC.Demos && typeof CYC.Demos.render === 'function' && typeof CYC.Demos.has === 'function' && CYC.Demos.has(id)); } catch (e) { return false; } };
  const questsReady = () => !!(CYC.Quests && typeof CYC.Quests.render === 'function');
  const fighter = (id) => CYC.byId[id] || null;
  const roleColor = (role) => (CYC.ROLES[role] && CYC.ROLES[role].color) || '#777';
  const heistsOf = (lv) => CYC.HEISTS.filter((h) => h.level === lv.id);
  const levelOf = (id) => CYC.LEVELS.find((l) => l.id === id) || CYC.LEVELS[0];
  const payloadLabel = (kind) => { try { return CYC.Engine && typeof CYC.Engine.payloadLabel === 'function' ? CYC.Engine.payloadLabel(kind) : kind; } catch (e) { return kind; } };
  const fmtBits = (b) => (vizReady() && CYC.Viz.fmtBits ? CYC.Viz.fmtBits(b) : `${Math.round(b / 8)} B`);
  const fmtBytes = (b) => fmtBits(b * 8);
  const rankIdx = (r) => RANKS.indexOf(r);
  const visibleRoles = () => (state.ctx === 'explore' ? ROLE_ORDER : (state.level.roles || []).slice().sort((a, b) => ROLE_ORDER.indexOf(a) - ROLE_ORDER.indexOf(b)));
  const maxLevel = () => (state.ctx === 'explore' ? MAX_LEVEL : state.level.id);
  const initials = (name) => {
    const words = String(name || '?').split(/\s+/).filter((w) => w && !/^the$/i.test(w));
    if (!words.length) return '?';
    if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
    return (words[0][0] + words[1][0]).toUpperCase();
  };
  function safe(fn, fallback) { try { return fn(); } catch (e) { if (window.console) console.error(e); return fallback; } }

  // ---------------- progress / tutor persistence ----------------
  function loadProgress() {
    let p = null;
    try { p = JSON.parse(localStorage.getItem(PKEY)); } catch (e) { /* storage unavailable: play without it */ }
    p = p && typeof p === 'object' ? p : {};
    return {
      solved: p.solved && typeof p.solved === 'object' ? p.solved : {},
      unlockedAll: !!p.unlockedAll,
      quiz: p.quiz && typeof p.quiz === 'object' ? p.quiz : {},
      quests: p.quests && typeof p.quests === 'object' ? p.quests : {},
    };
  }
  function saveProgress() { try { localStorage.setItem(PKEY, JSON.stringify(state.progress)); } catch (e) { /* ignore */ } }
  function loadTutor() {
    let t = null;
    try { t = JSON.parse(localStorage.getItem(TKEY)); } catch (e) { /* ignore */ }
    t = t && typeof t === 'object' ? t : {};
    return { on: t.on !== false, seen: new Set(Array.isArray(t.seen) ? t.seen : []) };
  }
  function saveTutor() { try { localStorage.setItem(TKEY, JSON.stringify({ on: state.tutor.on, seen: [...state.tutor.seen] })); } catch (e) { /* ignore */ } }
  const isSolved = (h) => !!state.progress.solved[h.id];
  const levelSolved = (lv) => heistsOf(lv).every(isSolved);
  function levelUnlocked(lv) {
    if (state.progress.unlockedAll || lv.id === 1) return true;
    const prev = CYC.LEVELS.find((l) => l.id === lv.id - 1);
    return !!prev && levelSolved(prev);
  }

  // ---------------- screens / modes ----------------
  function show(id) {
    document.querySelectorAll('#app .screen').forEach((s) => { s.hidden = s.id !== `screen-${id}`; });
    state.screen = id;
    window.scrollTo({ top: 0, behavior: 'auto' });
    updateStatus();
    // messages queued for a previous screen are stale; the one on display gets replaced by the next trigger
    state.fixer.queue = [];
    if (state.fixer.current) state.fixer.stale = true;
  }
  function updateStatus() {
    const s = $('topbar-status');
    if (!s) return;
    let txt = '';
    if (state.screen === 'explore') txt = 'EXPLORE';
    else if (state.screen === 'squad' || state.screen === 'results') txt = state.heist ? `${state.ctx === 'explore' ? 'EXPLORE' : 'LV ' + state.level.id} · ${state.heist.name}` : '';
    else if (state.screen === 'heists' || state.screen === 'quiz' || state.screen === 'quest') txt = `LV ${state.level.id} · ${state.level.name}`;
    s.textContent = txt;
  }
  function setMode(mode) {
    state.mode = mode;
    $('mode-heist').classList.toggle('is-on', mode === 'heist');
    $('mode-explore').classList.toggle('is-on', mode === 'explore');
    $('mode-heist').setAttribute('aria-pressed', String(mode === 'heist'));
    $('mode-explore').setAttribute('aria-pressed', String(mode === 'explore'));
    if (mode === 'explore') { renderExplore(); show('explore'); fixer('explore_intro'); }
    else { renderLevels(); show('levels'); }
  }

  // ---------------- THE FIXER (tutor) ----------------
  function fixer(key, opts) {
    opts = opts || {};
    if (!state.tutor.on || !CYC.TUTOR || !CYC.TUTOR.lines) return;
    const lines = CYC.TUTOR.lines[key];
    if (!lines || !lines.length) return;
    const always = key.startsWith('result:') || opts.always;
    if (!always && state.tutor.seen.has(key)) return;
    if (!always) { state.tutor.seen.add(key); saveTutor(); }
    state.fixer.queue.push({ key, lines });
    if (!state.fixer.current || state.fixer.stale) nextFixerMessage();
  }
  function nextFixerMessage() {
    state.fixer.current = state.fixer.queue.shift() || null;
    state.fixer.line = 0;
    state.fixer.stale = false;
    renderFixer();
  }
  function renderFixer() {
    const box = $('fixer');
    const cur = state.fixer.current;
    if (!cur) { box.hidden = true; document.body.classList.remove('fixer-open'); return; }
    const last = state.fixer.line >= cur.lines.length - 1 && state.fixer.queue.length === 0;
    $('fixer-text').innerHTML = cur.lines[state.fixer.line];   // trusted content from curriculum.js
    $('fixer-count').textContent = cur.lines.length > 1 ? `${state.fixer.line + 1}/${cur.lines.length}` : '';
    $('btn-fixer-next').textContent = last ? 'GOT IT' : 'NEXT ▶';
    box.hidden = false;
    document.body.classList.add('fixer-open');
  }
  function fixerAdvance() {
    const cur = state.fixer.current;
    if (!cur) return;
    if (state.fixer.line < cur.lines.length - 1) { state.fixer.line++; renderFixer(); }
    else nextFixerMessage();
  }
  function fixerDismiss() { state.fixer.queue = []; state.fixer.current = null; renderFixer(); }
  function setTutorOn(on) {
    state.tutor.on = on; saveTutor();
    const b = $('btn-fixer');
    b.textContent = `FIXER: ${on ? 'ON' : 'OFF'}`;
    b.setAttribute('aria-pressed', String(on));
    b.classList.toggle('is-on', on);
    if (!on) fixerDismiss();
  }

  // ---------------- small components ----------------
  function portrait(f, extraClass) {
    const isEmpty = !f || f.id === 'none';
    const color = isEmpty ? '#555' : roleColor(f.role);
    const box = el('div', { class: `portrait ${extraClass || ''} ${isEmpty ? 'is-empty' : ''}`, style: { '--rc': color }, 'aria-hidden': 'true' },
      el('span', { class: 'initials', text: isEmpty ? '—' : initials(f.name) }));
    const img = el('img', { alt: '', src: `img/fighters/${isEmpty ? 'none' : f.id}.png`, loading: 'lazy', decoding: 'async' });
    img.addEventListener('error', () => img.remove());
    box.append(img);
    return box;
  }
  function poster(h) {
    const color = roleColor(ROLE_ORDER[(h.id.length) % ROLE_ORDER.length]);
    const box = el('div', { class: 'poster', style: { '--rc': color }, 'aria-hidden': 'true' },
      el('span', { class: 'initials', text: initials(h.name) }));
    const img = el('img', { alt: '', src: `img/heists/${h.id}.png`, loading: 'lazy', decoding: 'async' });
    img.addEventListener('error', () => img.remove());
    box.append(img);
    return box;
  }
  function segs(value, color) {
    const v = Math.max(0, Math.min(5, Math.round(+value || 0)));
    return el('div', { class: 'segs', style: color ? { '--rc': color } : null, role: 'img', 'aria-label': `${v} of 5` },
      Array.from({ length: 5 }, (_, i) => el('i', { class: `seg ${i < v ? 'on' : ''}` })));
  }
  function stamp(on) { return el('span', { class: `stamp ${on ? 'on' : 'off'}`, text: on ? 'ON' : 'OFF' }); }
  function statBars(stats, color, compact) {
    return el('div', { class: `bars ${compact ? 'bars-compact' : ''}` },
      STAT_ROWS.map(([k, label]) => [el('span', { class: 'bar-label', text: label }), segs(stats[k], color)]));
  }
  // Requirement bars with heist metaphors: metaphor label, real term, low/high ends, hint on hover/tap.
  function reqBars(req, compact) {
    const M = CYC.METAPHORS || {};
    return el('div', { class: `bars req-bars ${compact ? 'bars-compact' : ''}` },
      REQ_ROWS.map(([k, label, kind]) => {
        const m = M[k] || {};
        const hint = m.hint || '';
        const hintEl = hint ? el('span', { class: 'req-hint', text: hint, hidden: true }) : null;
        const labelCell = el('button', {
          type: 'button', class: 'req-label', title: hint || null, 'aria-expanded': hint ? 'false' : null,
          onclick: hintEl ? (e) => { e.stopPropagation(); hintEl.hidden = !hintEl.hidden; e.currentTarget.setAttribute('aria-expanded', String(!hintEl.hidden)); } : null,
        },
          el('span', { class: 'req-meta', text: m.label || label }),
          el('span', { class: 'req-term', text: m.term || label.toLowerCase() }),
          hint ? el('span', { class: 'req-q', 'aria-hidden': 'true', text: '?' }) : null);
        const valueCell = kind === 'bool'
          ? el('div', { class: 'req-value' }, stamp(!!req[k]))
          : el('div', { class: 'req-value' },
            segs(req[k], 'var(--gold)'),
            !compact && (m.low || m.high) ? el('div', { class: 'req-ends' }, el('span', { text: m.low || '' }), el('span', { text: m.high || '' })) : null);
        return [labelCell, el('div', { class: 'req-cell' }, valueCell, hintEl)];
      }));
  }
  function levelBadge(f) { return el('span', { class: 'badge badge-level', text: `LV ${f.level}` }); }
  function losslessBadge() { return el('span', { class: 'badge badge-lossless', text: 'LOSSLESS' }); }
  function chip(text, color, soft) { return el('span', { class: `chip ${soft ? 'chip-soft' : ''}`, style: color ? { '--rc': color } : null, text }); }
  function rankBadge(rank) { return el('span', { class: `rank-badge rank-${rank}`, text: rank }); }

  function fighterCard(f, opts) {
    opts = opts || {};
    const isEmpty = f.id === 'none';
    const color = isEmpty ? '#555' : roleColor(f.role);
    return el('button', {
      type: 'button',
      class: `fcard ${isEmpty ? 'fcard-empty' : ''} ${opts.picked ? 'is-picked' : ''} ${opts.locked ? 'is-locked' : ''}`,
      style: { '--rc': color },
      'aria-pressed': opts.pickable ? String(!!opts.picked) : null,
      'aria-label': `${f.name}${opts.locked ? ', locked until level ' + f.level : ''}`,
      onclick: opts.onclick, onmouseenter: opts.onhover, onfocus: opts.onhover,
    },
      opts.picked ? el('span', { class: 'fcard-pick', text: opts.pickIndex != null ? `#${opts.pickIndex + 1}` : '✔' }) : null,
      portrait(f),
      el('div', { class: 'fcard-name', text: f.name }),
      el('div', { class: 'fcard-real', text: f.real }),
      el('div', { class: 'fcard-badges' },
        opts.locked ? el('span', { class: 'badge badge-level badge-locked', text: `🔒 LV ${f.level}` }) : (opts.showLevel && !isEmpty ? levelBadge(f) : null),
        f.stats && f.stats.lossless && !isEmpty ? losslessBadge() : null),
      opts.bars === false ? null : statBars(f.stats, color, true),
    );
  }

  // Special Move lab (CYC.Demos) embedded lazily into a container.
  function demoBox(fighterId, title) {
    if (!demosHas(fighterId)) return null;
    const f = fighter(fighterId);
    const box = el('div', { class: 'demo-box', style: { '--role-color': f ? roleColor(f.role) : 'var(--hot)', '--rc': f ? roleColor(f.role) : 'var(--hot)' } });
    const mount = el('div', { class: 'demo-mount' });
    box.append(el('h4', { class: 'demo-title', text: title || 'SPECIAL MOVE TRAINING' }), mount);
    requestAnimationFrame(() => {
      try { CYC.Demos.render(mount, fighterId); }
      catch (e) { if (window.console) console.error('CYC.Demos.render', e); mount.append(el('div', { class: 'callout callout-red', text: `Lab failed to load: ${e.message || e}` })); }
    });
    return box;
  }

  function detailCard(f, opts) {
    opts = opts || {};
    const isEmpty = f.id === 'none';
    const color = isEmpty ? '#777' : roleColor(f.role);
    const role = isEmpty ? null : CYC.ROLES[f.role];
    const fam = isEmpty ? null : CYC.FAMILIES[f.family];
    const actions = [];
    if (opts.pickable && !opts.locked) {
      actions.push(el('button', {
        type: 'button', class: `btn ${opts.picked ? 'btn-ghost' : 'btn-run'}`, id: 'btn-detail-pick',
        text: opts.picked ? (isEmpty ? 'CHAIR IS EMPTY' : 'UNPICK') : (isEmpty ? 'LEAVE CHAIR EMPTY' : 'PICK THIS FIGHTER'),
        disabled: opts.picked && isEmpty, onclick: opts.onpick,
      }));
    }
    if (opts.locked) actions.push(el('span', { class: 'badge badge-locked', text: `🔒 UNLOCKS AT LEVEL ${f.level}` }));
    return el('article', { class: 'detail', style: { '--rc': color } },
      el('div', { class: 'detail-top' },
        portrait(f),
        el('div', { class: 'stack' },
          el('div', { class: 'detail-name', text: f.name }),
          el('div', { class: 'detail-real', text: f.real }),
          role ? el('div', { class: 'detail-fam', text: `${role.codename} · ${fam ? fam.name : ''}` }) : el('div', { class: 'detail-fam', text: 'ANY CHAIR' }),
          el('div', { class: 'row' }, isEmpty ? null : levelBadge(f), f.stats && f.stats.lossless && !isEmpty ? losslessBadge() : null),
        )),
      el('h4', { text: 'SIGNATURE MOVE' }), el('p', { class: 'move', text: f.move }),
      el('h4', { text: 'WHY IT WORKS' }), el('p', { text: f.why }),
      f.usedIn && f.usedIn.length ? [el('h4', { text: 'USED IN' }), el('ul', { class: 'used' }, f.usedIn.map((u) => el('li', { text: u })))] : null,
      f.weakness ? [el('h4', { text: 'WEAKNESS' }), el('p', { class: 'weakness', text: f.weakness })] : null,
      el('h4', { text: 'STATS' }), statBars(f.stats, color, false),
      f.domains && f.domains.length ? [el('h4', { text: 'DOMAINS' }), el('div', { class: 'chips' }, f.domains.map((d) => chip(d, null, true)))] : null,
      actions.length ? el('div', { class: 'detail-actions' }, actions) : null,
      isEmpty ? null : demoBox(f.id),
    );
  }

  // ---------------- level select ----------------
  function renderLevels() {
    const grid = clearEl($('level-grid'));
    for (const lv of CYC.LEVELS) {
      const hs = heistsOf(lv);
      const solved = hs.filter(isSolved).length;
      const unlocked = levelUnlocked(lv);
      const quiz = state.progress.quiz[lv.id];
      const quests = (CYC.SIDEQUESTS || []).filter((q) => q.level === lv.id);
      const qdone = quests.filter((q) => state.progress.quests[q.id] != null).length;
      grid.append(el('button', {
        type: 'button', class: 'level-card', id: `level-${lv.id}`, disabled: !unlocked,
        'aria-label': `Level ${lv.id} ${lv.name}${unlocked ? '' : ', locked'}`,
        onclick: () => enterLevel(lv),
      },
        unlocked ? null : el('span', { class: 'lock', 'aria-hidden': 'true', text: '🔒' }),
        quiz && quiz.passed ? el('span', { class: 'cleared-stamp', text: 'CLEARED' }) : null,
        el('div', { class: 'level-num', text: `LV ${lv.id}` }),
        el('div', { class: 'level-name', text: lv.name }),
        el('div', { class: 'level-tag', text: lv.tagline }),
        el('div', { class: 'level-roles' }, lv.roles.map((r) => chip(CYC.ROLES[r].codename, roleColor(r)))),
        el('div', { class: 'level-progress', text: unlocked ? `${solved} / ${hs.length} HEISTS SOLVED${quests.length ? ` · ${qdone}/${quests.length} SIDE QUESTS` : ''}` : `SOLVE ALL OF LEVEL ${lv.id - 1} TO UNLOCK` }),
      ));
    }
  }
  function enterLevel(lv, tab) {
    state.level = lv;
    state.heistTab = tab || 'heists';
    renderHeists();
    show('heists');
    fixer(`level_intro:${lv.id}`);
  }

  // ---------------- heist select (heists / side quests / notebook) ----------------
  function heistCard(h, opts) {
    opts = opts || {};
    const solved = state.progress.solved[h.id];
    const showCodec = opts.showCodec || !!solved;
    return el('article', { class: 'heist-card', id: `heist-${h.id}${opts.idSuffix || ''}` },
      poster(h),
      el('div', { class: 'row' },
        el('div', { class: 'heist-name', text: h.name }),
        opts.showLevel ? el('span', { class: 'badge badge-level', text: `LV ${h.level}` }) : null,
        solved ? rankBadge(solved) : null),
      el('div', { class: `heist-codec ${showCodec ? '' : 'hidden-codec'}`, text: showCodec ? `THE REAL CREW: ${h.codec}` : 'THE REAL CREW: ??? (solve to reveal)' }),
      el('p', { class: 'heist-brief', text: h.brief }),
      reqBars(h.req, true),
      el('details', { class: 'briefing' },
        el('summary', { text: 'BRIEFING' }),
        el('ul', {}, (h.reqNotes || []).map((n) => el('li', { text: n })))),
      el('div', { class: 'heist-foot' },
        el('button', { type: 'button', class: 'btn btn-run', text: opts.buttonText || 'TAKE THE JOB', onclick: () => opts.onopen(h) }),
        el('span', { class: 'dim', text: `sample: ${payloadLabel(h.payload)}` })),
    );
  }
  function setHeistTab(tab) { state.heistTab = tab; renderHeists(); }
  function renderHeists() {
    const lv = state.level;
    $('heists-title').textContent = `LV ${lv.id} · ${lv.name}`;
    $('heists-tagline').textContent = lv.tagline;
    const quests = (CYC.SIDEQUESTS || []).filter((q) => q.level <= lv.id);
    const lessons = (CYC.LESSONS && CYC.LESSONS[lv.id]) || [];
    document.querySelectorAll('#heists-subtabs .tab').forEach((b) => {
      const on = b.dataset.tab === state.heistTab;
      b.classList.toggle('is-on', on); b.setAttribute('aria-selected', String(on));
      if (b.dataset.tab === 'quests') b.textContent = `SIDE QUESTS (${quests.length})`;
      if (b.dataset.tab === 'notebook') b.textContent = `THE FIXER'S NOTEBOOK (${lessons.length})`;
    });
    const body = clearEl($('heists-body'));
    if (state.heistTab === 'quests') body.append(renderQuestRow(quests));
    else if (state.heistTab === 'notebook') body.append(renderNotebook(lessons));
    else {
      const grid = el('div', { class: 'heist-grid' }, heistsOf(lv).map((h) => heistCard(h, { onopen: (hh) => openHeist(hh, { ctx: 'heist' }) })));
      body.append(grid);
      const quiz = state.progress.quiz[lv.id];
      if (CYC.QUIZZES && CYC.QUIZZES[lv.id] && levelSolved(lv)) {
        body.append(el('div', { class: 'checkpoint-strip' },
          el('span', { class: 'pix', text: quiz && quiz.passed ? `CHECKPOINT CLEARED · ${quiz.score}/${CYC.QUIZZES[lv.id].length}` : (quiz ? `CHECKPOINT: ${quiz.score}/${CYC.QUIZZES[lv.id].length} · RETAKE?` : 'ALL HEISTS SOLVED: THE CHECKPOINT IS OPEN') }),
          el('button', { type: 'button', class: 'btn btn-gold', id: 'btn-checkpoint', text: quiz ? 'RETAKE CHECKPOINT' : 'TAKE THE CHECKPOINT', onclick: () => startQuiz(lv) })));
      }
    }
  }
  function renderQuestRow(quests) {
    if (!quests.length) return el('div', { class: 'placeholder', text: 'NO SIDE QUESTS AT THIS LEVEL' });
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Hands-on jobs that teach one mechanism each. Play them in any order; a stamp marks the ones you finished.' }),
      el('div', { class: 'quest-grid' }, quests.map((q) => {
        const done = state.progress.quests[q.id];
        return el('button', { type: 'button', class: 'quest-card', id: `quest-${q.id}`, onclick: () => openQuest(q) },
          done != null ? el('span', { class: 'cleared-stamp', text: `DONE · ${done}` }) : null,
          el('div', { class: 'row' }, el('span', { class: 'quest-name', text: q.name }), el('span', { class: 'badge badge-level', text: `LV ${q.level}` })),
          el('div', { class: 'quest-concept', text: q.concept }),
          el('div', { class: 'mini-squad' }, (q.fighters || []).map(fighter).filter(Boolean).map((f) => el('span', { class: 'mini' }, portrait(f), el('span', { class: 'mini-name', text: f.name })))),
        );
      })));
  }
  function renderNotebook(lessons) {
    if (!lessons.length) return el('div', { class: 'placeholder', text: 'THE NOTEBOOK IS EMPTY FOR THIS LEVEL' });
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'The Fixer\'s notes for this level: the theory behind the chairs, each with a lab where one exists.' }),
      el('div', { class: 'lesson-list' }, lessons.map((L, i) => el('article', { class: 'lesson', id: `lesson-${L.id}` },
        el('div', { class: 'lesson-head' }, el('span', { class: 'lesson-num', text: String(i + 1).padStart(2, '0') }), el('h3', { class: 'h-pixel', text: L.title })),
        el('div', { class: 'lesson-body', html: L.body }),
        L.takeaway ? el('div', { class: 'takeaway' }, el('span', { class: 'pix-sm', text: 'TAKEAWAY' }), el('span', { text: L.takeaway })) : null,
        L.demo ? (demoBox(L.demo, `LAB: ${(fighter(L.demo) || {}).name || L.demo}`) || el('p', { class: 'dim', text: `Lab for ${(fighter(L.demo) || {}).name || L.demo}: loading (js/demos.js).` })) : null,
        L.duel != null && CYC.DUELS && CYC.DUELS[L.duel] ? el('button', { type: 'button', class: 'btn btn-cyan', text: 'STAGE THE DUEL', onclick: () => openDuelPreset(L.duel) }) : null,
      ))));
  }

  // ---------------- squad room ----------------
  function openHeist(h, opts) {
    opts = opts || {};
    state.ctx = opts.ctx || 'heist';
    state.heist = h;
    state.level = levelOf(h.level);
    state.codec = opts.codec || null;
    state.squad = {};
    if (opts.squad) for (const [r, ids] of Object.entries(opts.squad)) state.squad[r] = (ids || []).slice();
    state.result = null; state.engineError = null; state.lastScore = null;
    renderSquadRoom();
    show('squad');
    scheduleRun(0);
    if (state.ctx === 'heist') fixer(`heist_intro:${h.id}`);
  }
  function renderSquadRoom() {
    const h = state.heist;
    $('squad-title').textContent = state.codec ? `SQUAD ROOM · ${state.codec.name}` : `SQUAD ROOM · ${h.name}`;
    $('btn-back-heists').textContent = state.ctx === 'explore' ? '← EXPLORE' : '← HEISTS';
    const job = clearEl($('squad-job'));
    job.append(
      el('div', { class: 'stack' },
        el('div', { class: 'job-meta' },
          el('span', { class: 'pix', style: { color: 'var(--gold)' }, text: h.name }),
          el('span', { class: 'badge badge-level', text: `LV ${h.level}` }),
          state.ctx === 'explore' ? el('span', { class: 'badge badge-level', text: `CODEC: ${h.codec}` }) : null,
          el('span', { class: 'dim', text: `sample payload: ${payloadLabel(h.payload)}` })),
        el('p', { class: 'heist-brief', text: h.brief }),
        el('details', { class: 'briefing' }, el('summary', { text: 'BRIEFING' }), el('ul', {}, (h.reqNotes || []).map((n) => el('li', { text: n })))),
        state.codec ? el('p', { class: 'dim', text: state.codec.notes }) : null,
      ),
      el('div', {}, el('div', { class: 'pix', style: { color: 'var(--ink-dim)', marginBottom: '6px' }, text: 'JOB SHEET' }), reqBars(h.req, false)),
    );
    renderChairs();
    renderCombos();
  }
  function renderChairs() {
    const row = clearEl($('chair-row'));
    visibleRoles().forEach((role, i) => {
      const R = CYC.ROLES[role];
      const ids = state.squad[role] || [];
      const seats = [];
      if (ids.length === 1 && ids[0] === 'none') {
        seats.push(el('div', { class: 'seat seat-open seat-empty' }, portrait(CYC.EMPTY), el('span', { text: 'EMPTY CHAIR (ON PURPOSE)' })));
      } else {
        const real = ids.filter((id) => id !== 'none' && fighter(id));
        for (const id of real) {
          const f = fighter(id);
          seats.push(el('div', { class: 'seat' }, portrait(f), el('span', { class: 'seat-name', text: f.name })));
        }
        if (!real.length) seats.push(el('div', { class: 'seat seat-open' }, el('span', { text: 'OPEN CHAIR\nCLICK TO CHOOSE' })));
        else if (real.length === 1) seats.push(el('div', { class: 'seat seat-open' }, el('span', { text: '+ ONE MORE?' })));
      }
      row.append(el('div', {
        class: 'chair', id: `chair-${role}`, role: 'button', tabindex: '0',
        style: { '--rc': roleColor(role) },
        'aria-label': `${R.codename}, ${R.title}. ${ids.length ? 'Filled' : 'Open chair'}. Click to choose fighters.`,
        onclick: () => openChooser(role),
        onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openChooser(role); } },
      },
        el('div', { class: 'chair-stripe' }),
        el('span', { class: 'chair-order', 'aria-hidden': 'true', text: String(i + 1) }),
        el('div', { class: 'chair-code', text: R.codename }),
        el('div', { class: 'chair-title', text: R.title }),
        el('div', { class: 'chair-q', text: R.question }),
        el('div', { class: 'chair-seats' }, seats),
      ));
    });
  }
  function comboColors(ids) {
    const cols = [...new Set(ids.map((id) => fighter(id)).filter(Boolean).map((f) => roleColor(f.role)))];
    if (!cols.length) return 'var(--hot)';
    if (cols.length === 1) return cols[0];
    return `linear-gradient(90deg, ${cols.join(', ')})`;
  }
  function comboBanner(c, kind) {
    const ids = (c.needs || []);
    return el('div', { class: `combo ${kind === 'anti' ? 'combo-anti' : 'combo-tag'}`, style: { '--combo-bg': kind === 'anti' ? 'var(--red)' : comboColors(ids) } },
      el('div', { class: 'combo-flash', text: kind === 'anti' ? 'FRIENDLY FIRE' : 'TAG TEAM!' }),
      el('div', { class: 'combo-body' },
        el('div', { class: 'row' }, el('span', { class: 'combo-name', text: c.name }), ids.map(fighter).filter(Boolean).map((f) => portrait(f, 'portrait-xs'))),
        el('p', { class: 'combo-text', text: c.text })));
  }
  function renderCombos() {
    const bar = clearEl($('combo-bar'));
    if (!state.heist || !CYC.Rules) return;
    const squad = squadForEngine();
    const combos = safe(() => (CYC.Rules.activeCombos ? CYC.Rules.activeCombos(squad, state.heist) : []), []);
    const anti = safe(() => (CYC.Rules.activeAnticombos ? CYC.Rules.activeAnticombos(squad, state.heist, visibleRoles()) : []), []);
    combos.forEach((c) => bar.append(comboBanner(c, 'tag')));
    anti.forEach((a) => bar.append(comboBanner(a, 'anti')));
  }

  function squadForEngine() {
    const out = {};
    for (const role of visibleRoles()) {
      const ids = (state.squad[role] || []).filter((id) => CYC.byId[id]);
      if (ids.length) out[role] = ids.slice();
    }
    return out;
  }
  function onSquadChange() {
    renderChairs();
    renderCombos();
    if (state.chooser.open) renderChooser();
    scheduleRun();
  }
  function togglePick(role, id) {
    let arr = (state.squad[role] || []).filter((x) => x !== 'none');
    let picked = false;
    if (id === 'none') arr = ['none'];
    else if (arr.includes(id)) arr = arr.filter((x) => x !== id);
    else { if (arr.length >= 2) arr.shift(); arr.push(id); picked = true; }
    state.squad[role] = arr;
    state.chooser.focus = id;
    onSquadChange();
    if (picked) fixer(`pick:${id}`);
  }

  // ---------------- chooser overlay ----------------
  function familiesFor(role) { return Object.entries(CYC.FAMILIES).filter(([, f]) => f.role === role).map(([id, f]) => ({ id, ...f })); }
  function openChooser(role) {
    const picked = (state.squad[role] || []).filter((id) => id !== 'none');
    const fams = familiesFor(role);
    const first = picked.length ? fighter(picked[0]) : null;
    state.chooser = { open: true, role, family: (first && first.family) || (fams[0] && fams[0].id) || null, focus: picked[0] || null };
    renderChooser();
    $('overlay-fighter').hidden = false;
    document.body.style.overflow = 'hidden';
    setTimeout(() => { const b = $('btn-close-fighter'); if (b) b.focus(); }, 0);
    fixer(`chair_open:${role}`);
  }
  function closeChooser() {
    state.chooser.open = false;
    $('overlay-fighter').hidden = true;
    document.body.style.overflow = '';
    const chair = $(`chair-${state.chooser.role}`);
    if (chair) chair.focus();
  }
  function renderChooser() {
    const { role, family } = state.chooser;
    const R = CYC.ROLES[role];
    const color = roleColor(role);
    const pickedIds = (state.squad[role] || []);
    const picked = pickedIds.filter((id) => id !== 'none');
    const chairEmpty = pickedIds.length === 1 && pickedIds[0] === 'none';
    const lvl = maxLevel();

    const roleBox = clearEl($('fighter-overlay-role'));
    roleBox.append(chip(R.codename, color), el('span', { text: R.title }), el('span', { class: 'pix-sm', text: `· ${picked.length}/2 PICKED (a 3rd pick replaces the oldest)` }));

    const col = clearEl($('family-col'));
    for (const fam of familiesFor(role)) {
      const fighters = CYC.FIGHTERS.filter((f) => f.family === fam.id);
      const nPicked = fighters.filter((f) => picked.includes(f.id)).length;
      col.append(el('button', {
        type: 'button', class: `family-btn ${fam.id === family ? 'is-on' : ''}`, id: `fam-${fam.id}`, style: { '--rc': color },
        'aria-pressed': String(fam.id === family),
        onclick: () => { state.chooser.family = fam.id; renderChooser(); },
      },
        el('span', { class: 'fam-name', text: fam.name }),
        el('span', { class: 'fam-count', text: `${fighters.filter((f) => f.level <= lvl).length}/${fighters.length} AVAILABLE` }),
        nPicked ? el('span', { class: 'fam-pick', text: ` · ${nPicked} PICKED` }) : null,
        el('span', { class: 'fam-note', text: fam.note }),
      ));
    }
    col.append(el('button', {
      type: 'button', class: `family-btn ${chairEmpty ? 'is-on' : ''}`, id: 'fam-empty', style: { '--rc': '#777' },
      onclick: () => { state.chooser.focus = 'none'; renderDetailCol(); },
    }, el('span', { class: 'fam-name', text: 'EMPTY CHAIR' }), el('span', { class: 'fam-note', text: CYC.EMPTY.move })));

    const grid = clearEl($('fighter-grid'));
    const fam = CYC.FAMILIES[family];
    if (fam) {
      grid.append(el('div', { class: 'fam-head', style: { '--rc': color } }, el('div', { class: 'pix-md', text: fam.name }), el('p', { text: fam.note })));
      for (const f of CYC.FIGHTERS.filter((x) => x.family === family)) {
        const locked = f.level > lvl;
        const idx = picked.indexOf(f.id);
        grid.append(fighterCard(f, {
          pickable: true, picked: idx >= 0, pickIndex: idx >= 0 ? idx : null, locked, showLevel: true,
          onhover: () => { if (state.chooser.focus !== f.id) { state.chooser.focus = f.id; renderDetailCol(); } },
          onclick: () => { state.chooser.focus = f.id; if (locked) { renderDetailCol(); return; } togglePick(role, f.id); },
        }));
      }
    }
    grid.append(fighterCard(CYC.EMPTY, {
      pickable: true, picked: chairEmpty, bars: false,
      onhover: () => { if (state.chooser.focus !== 'none') { state.chooser.focus = 'none'; renderDetailCol(); } },
      onclick: () => togglePick(role, 'none'),
    }));
    renderDetailCol();
  }
  function renderDetailCol() {
    const { role, focus } = state.chooser;
    const pickedIds = state.squad[role] || [];
    const det = clearEl($('fighter-detail'));
    let f = focus ? fighter(focus) : null;
    if (!f) {
      const firstPicked = pickedIds.find((id) => id !== 'none');
      f = firstPicked ? fighter(firstPicked) : CYC.FIGHTERS.find((x) => x.family === state.chooser.family) || CYC.EMPTY;
    }
    const isEmpty = f.id === 'none';
    const picked = isEmpty ? (pickedIds.length === 1 && pickedIds[0] === 'none') : pickedIds.includes(f.id);
    const locked = !isEmpty && f.level > maxLevel();
    det.append(detailCard(f, { pickable: true, picked, locked, onpick: () => togglePick(role, f.id) }));
  }

  // ---------------- engine run + pipeline panel ----------------
  function scheduleRun(ms) {
    clearTimeout(runTimer);
    runTimer = setTimeout(runEngine, ms == null ? 50 : ms);
  }
  function runSquad(payload, squad) {
    if (!engineReady()) return { result: null, error: null };
    try { return { result: CYC.Engine.run(payload, squad), error: null }; }
    catch (e) { if (window.console) console.error('CYC.Engine.run failed', e); return { result: null, error: e }; }
  }
  function runEngine() {
    state.result = null; state.engineError = null;
    if (!state.heist) return;
    const r = runSquad(state.heist.payload, squadForEngine());
    state.result = r.result; state.engineError = r.error;
    renderPipeline();
  }
  function stageCard(st, i) {
    const f = fighter(st.fighterId) || CYC.EMPTY;
    const color = roleColor(st.role);
    const R = CYC.ROLES[st.role];
    const m = st.metrics || {};
    const metrics = [];
    if (m.symbols != null) metrics.push(el('span', {}, 'symbols ', el('b', { text: (+m.symbols).toLocaleString() })));
    if (m.alphabet != null) metrics.push(el('span', {}, 'alphabet ', el('b', { text: String(m.alphabet) })));
    if (m.entropyBitsPerSymbol != null) metrics.push(el('span', {}, 'H ', el('b', { text: `${(+m.entropyBitsPerSymbol).toFixed(2)} b/sym` })));
    if (m.bits != null) metrics.push(el('span', {}, 'bits ', el('b', { text: `${Math.round(m.bits).toLocaleString()} (${fmtBits(m.bits)})` })));
    if (m.sideInfoBytes != null) metrics.push(el('span', {}, 'side info ', el('b', { text: fmtBytes(m.sideInfoBytes) })));
    if (m.distortion && m.distortion.label) metrics.push(el('span', {}, m.distortion.metric || 'distortion', ' ', el('b', { text: m.distortion.label })));
    const canvas = el('canvas', { width: '320', height: '150', 'aria-label': `${st.title || f.name} visualization (${st.viz && st.viz.type ? st.viz.type : 'none'})`, role: 'img' });
    canvas._viz = st.viz;
    return el('article', { class: 'stage', style: { '--rc': color }, id: `stage-${i}` },
      el('div', { class: 'stage-stripe' }),
      el('div', { class: 'stage-head' },
        portrait(f),
        el('div', {},
          el('div', { class: 'stage-role', text: `${i + 1}. ${R ? R.codename : st.role}` }),
          el('div', { class: 'stage-name', text: f.name }),
          el('div', { class: 'stage-title', text: st.title || '' }))),
      canvas,
      el('p', { class: 'stage-caption', text: st.caption || '' }),
      metrics.length ? el('div', { class: 'stage-metrics' }, metrics) : null,
    );
  }
  function hbar(label, value, max, color, valueText, over) {
    const pct = max > 0 ? Math.max(value > 0 ? 0.5 : 0, Math.min(100, (value / max) * 100)) : 0;
    return el('div', { class: 'hbar-row' },
      el('span', { class: 'hbar-label', text: label }),
      el('div', { class: `hbar ${over ? 'over' : ''}`, role: 'img', 'aria-label': `${label}: ${valueText}` }, el('i', { style: { width: `${pct}%`, '--hc': color } })),
      el('span', { class: 'hbar-val', text: valueText }));
  }
  function sideInfoLabel(bytes) {
    if (!bytes) return 'none: self-describing stream';
    return `${fmtBytes(bytes)} (${Math.round(bytes).toLocaleString()} bytes)`;
  }
  function distortionText(r) { return r.distortion ? (r.distortion.label || `${r.distortion.metric} ${r.distortion.value}`) : 'not measured'; }
  function metersBlock(r) {
    const max = Math.max(r.rawBits || 0, r.entropyBoundBits || 0, r.finalBits || 0, 1);
    const ratio = r.finalBits > 0 ? r.rawBits / r.finalBits : 0;
    const side = r.sideInfoBytes || 0;
    const sideFrac = side > 0 ? Math.max(0.03, Math.min(1, Math.log10(side) / 8)) : 0; // log scale, 100 MB = full
    return el('div', { class: 'meters' },
      el('div', { class: 'meter-box' },
        el('div', { class: 'meter-title' }, el('span', { text: 'SIZE METER' }), el('span', { class: 'ratio', text: ratio ? `RATIO ×${ratio.toFixed(2)}` : 'RATIO —' })),
        hbar('RAW', r.rawBits, max, 'var(--ink-faint)', `${fmtBits(r.rawBits)} · ${Math.round(r.rawBits || 0).toLocaleString()} bits`),
        hbar('ENTROPY BOUND', r.entropyBoundBits, max, 'var(--gold)', `${fmtBits(r.entropyBoundBits)} · ${Math.round(r.entropyBoundBits || 0).toLocaleString()} bits`),
        hbar('FINAL', r.finalBits, max, r.finalBits > r.rawBits ? 'var(--red)' : 'var(--acid)', `${fmtBits(r.finalBits)} · ${Math.round(r.finalBits || 0).toLocaleString()} bits`, r.finalBits > r.rawBits),
        el('p', { class: 'dim', style: { marginTop: '6px', marginBottom: 0 }, text: 'Entropy bound = order-0 Shannon entropy of the symbols that reached the Packer. Final = the honest output stream incl. header/table overhead. Side info is not counted here.' }),
      ),
      el('div', { class: 'stack' },
        el('div', { class: 'meter-box' },
          el('div', { class: 'meter-title' }, el('span', { text: 'SIDE INFO' }), el('span', { class: 'dim', text: 'decoder must already hold' })),
          el('div', { class: 'hbar-row' },
            el('span', { class: 'hbar-label', text: 'LOG SCALE' }),
            el('div', { class: 'hbar', role: 'img', 'aria-label': `side info ${sideInfoLabel(side)}` }, el('i', { style: { width: `${sideFrac * 100}%`, '--hc': 'var(--orange)' } })),
            el('span', { class: 'hbar-val', text: side ? fmtBytes(side) : '0 B' })),
          el('p', { class: 'dim', style: { marginBottom: 0 }, text: sideInfoLabel(side) })),
        el('div', { class: 'meter-box' },
          el('div', { class: 'meter-title' }, el('span', { text: 'DISTORTION' })),
          r.lossless
            ? el('div', { class: 'row' }, el('span', { class: 'badge badge-exact', text: 'BIT-EXACT' }), el('span', { class: 'dim', text: 'no quantizer ran, nothing was thrown away' }))
            : el('div', { class: 'row' },
              el('span', { class: 'badge badge-lossy', text: 'LOSSY' }),
              r.distortion ? el('span', { class: 'distortion-val', text: distortionText(r) }) : el('span', { class: 'dim', text: 'distortion not measured' })),
        )),
    );
  }
  function warningsBlock(warnings, extraClass) {
    if (!warnings || !warnings.length) return document.createDocumentFragment();
    return el('div', { class: 'warn-list' }, warnings.map((w) => el('div', { class: `callout ${extraClass || ''}`, text: w })));
  }
  function enginePlaceholder() {
    return el('div', {},
      el('div', { class: 'placeholder', text: 'ENGINE LOADING…' }),
      el('p', { class: 'dim', style: { marginTop: '8px' }, text: 'CYC.Engine is not available yet (js/engine.js). The pipeline panel comes alive as soon as it loads; the squad can be built in the meantime.' }));
  }
  function renderPipeline() {
    const panel = clearEl($('pipeline'));
    const h = state.heist;
    if (!h) return;
    const r = state.result;
    panel.append(el('div', { class: 'pipeline-head' },
      el('h3', { class: 'h-pixel', text: 'LIVE PIPELINE' }),
      el('span', { class: 'pipeline-sub', text: `${payloadLabel(h.payload)}${r ? ` · ${r.stages.length} stage${r.stages.length === 1 ? '' : 's'}` : ''}` })));
    if (!engineReady()) { panel.append(enginePlaceholder()); return; }
    if (state.engineError) { panel.append(warningsBlock([`Engine error: ${state.engineError.message || state.engineError}`], 'callout-red')); return; }
    if (!r) { panel.append(el('div', { class: 'placeholder', text: 'RUNNING…' })); return; }
    if (!r.stages || !r.stages.length) panel.append(el('div', { class: 'placeholder', text: 'EMPTY SQUAD: RAW FIXED-LENGTH BITS PASS STRAIGHT THROUGH' }));
    else panel.append(el('div', { class: 'stage-row' }, r.stages.map(stageCard)));
    panel.append(metersBlock(r), warningsBlock(r.warnings));
    requestAnimationFrame(drawCanvases);
  }
  function drawCanvases() {
    if (!vizReady()) return;
    document.querySelectorAll('canvas').forEach((c) => { if (c._viz && !c.closest('[hidden]')) CYC.Viz.draw(c, c._viz); });
  }

  // ---------------- results ----------------
  function runHeist() {
    const h = state.heist;
    if (!h) return;
    if (!state.result && engineReady() && !state.engineError) runEngine();
    const squad = squadForEngine();
    let score = null, error = null;
    if (rulesReady()) {
      try { score = CYC.Rules.score(h, squad, state.result, visibleRoles()); }
      catch (e) { error = e; if (window.console) console.error('CYC.Rules.score failed', e); }
    }
    state.lastScore = score;
    state.pendingQuiz = null;
    let unlockedLevel = null, levelDone = false;
    if (score && state.ctx === 'heist' && rankIdx(score.rank) >= rankIdx('B')) {
      const wasDone = levelSolved(state.level);
      const prev = state.progress.solved[h.id];
      if (!prev || rankIdx(score.rank) > rankIdx(prev)) state.progress.solved[h.id] = score.rank;
      saveProgress();
      if (!wasDone && levelSolved(state.level)) {
        levelDone = true;
        const next = CYC.LEVELS.find((l) => l.id === h.level + 1);
        if (next && !state.progress.unlockedAll) unlockedLevel = next;
        if (CYC.QUIZZES && CYC.QUIZZES[state.level.id]) state.pendingQuiz = state.level;
      }
    }
    renderResults(score, error, unlockedLevel);
    show('results');
    if (score) fixer(`result:${score.rank}`);
    if (levelDone) fixer(`level_complete:${state.level.id}`);
  }
  function nextHeist() {
    if (state.ctx === 'explore') { setMode('explore'); return; }
    if (state.pendingQuiz) { const lv = state.pendingQuiz; state.pendingQuiz = null; startQuiz(lv); return; }
    const inLevel = heistsOf(state.level);
    const idx = inLevel.indexOf(state.heist);
    const unsolved = inLevel.find((x, i) => i > idx && !isSolved(x)) || inLevel.find((x) => !isSolved(x) && x !== state.heist);
    if (unsolved) { openHeist(unsolved, { ctx: 'heist' }); return; }
    const after = inLevel[idx + 1];
    if (after) { openHeist(after, { ctx: 'heist' }); return; }
    const next = CYC.LEVELS.find((l) => l.id === state.level.id + 1);
    if (next && levelUnlocked(next)) { enterLevel(next); return; }
    renderLevels(); show('levels');
  }
  function verdictCard(v) {
    const f = v.fighterId ? fighter(v.fighterId) : null;
    const R = CYC.ROLES[v.role];
    return el('article', { class: 'verdict', style: { '--vc': OK_COLOR[v.ok] || 'var(--ink-faint)', '--rc': roleColor(v.role) } },
      el('div', { class: 'verdict-head' },
        portrait(f || CYC.EMPTY),
        el('div', {}, el('div', { class: 'verdict-role', text: R ? R.codename : v.role }), el('div', { class: 'verdict-fighter', text: f ? f.name : 'EMPTY CHAIR' })),
        el('span', { class: 'verdict-ok', text: String(v.ok || '').toUpperCase() })),
      el('p', { class: 'verdict-text', text: v.text || '' }));
  }
  function comboVerdictCard(v) {
    const list = v.combo ? (CYC.COMBOS || []) : (CYC.ANTICOMBOS || []);
    const c = list.find((x) => x.id === (v.combo || v.anticombo));
    if (c) return comboBanner(c, v.anticombo ? 'anti' : 'tag');
    return el('div', { class: `combo ${v.ok === 'wrong' ? 'combo-anti' : 'combo-tag'}`, style: { '--combo-bg': v.ok === 'wrong' ? 'var(--red)' : 'var(--hot)' } },
      el('div', { class: 'combo-flash', text: v.ok === 'wrong' ? 'FRIENDLY FIRE' : 'TAG TEAM!' }),
      el('div', { class: 'combo-body' }, el('p', { class: 'combo-text', text: v.text })));
  }
  function renderResults(score, error, unlockedLevel) {
    const box = clearEl($('results'));
    const h = state.heist;
    const r = state.result;
    if (!score) {
      box.append(el('div', { class: 'results-hero' },
        el('div', { class: 'rank-letter', text: '?' }),
        el('div', {},
          el('div', { class: 'results-headline', text: error ? 'THE JUDGE TRIPPED' : 'JUDGE LOADING…' }),
          el('p', { class: 'dim', text: error ? `CYC.Rules.score threw: ${error.message || error}` : 'CYC.Rules is not available yet (js/rules.js). Scoring appears once it loads.' }))));
    } else {
      const solved = state.progress.solved[h.id];
      const verdicts = score.verdicts || [];
      const chairV = verdicts.filter((v) => v.role !== 'combo');
      const comboV = verdicts.filter((v) => v.role === 'combo');
      box.append(el('div', { class: 'results-hero' },
        el('div', { class: `rank-letter rank-${score.rank}`, text: score.rank, 'aria-label': `Rank ${score.rank}` }),
        el('div', {},
          el('div', { class: 'results-headline', text: score.headline || '' }),
          el('div', { class: 'results-score', text: `SCORE ${Math.round(score.score)} / 100 · ${h.name}` }),
          el('div', { class: 'results-solved row' },
            state.ctx === 'heist'
              ? (rankIdx(score.rank) >= rankIdx('B')
                ? el('span', { class: 'badge badge-exact', text: `HEIST SOLVED${solved && solved !== score.rank ? ` · BEST RANK ${solved}` : ''}` })
                : el('span', { class: 'badge badge-lossy', text: 'RANK B OR BETTER SOLVES THE HEIST' }))
              : el('span', { class: 'badge badge-level', text: 'EXPLORE MODE: NO PROGRESSION' }),
            unlockedLevel ? el('span', { class: 'badge badge-exact blink', text: `LEVEL ${unlockedLevel.id} ${unlockedLevel.name} UNLOCKED` }) : null,
            state.pendingQuiz ? el('span', { class: 'badge badge-gold', text: 'CHECKPOINT UNLOCKED' }) : null))));
      if (comboV.length) {
        box.append(el('h3', { class: 'h-pixel', text: 'SQUAD CHEMISTRY' }));
        box.append(el('div', { class: 'combo-list' }, comboV.map(comboVerdictCard)));
      }
      box.append(el('h3', { class: 'h-pixel', text: 'PER-CHAIR VERDICTS' }));
      box.append(el('div', { class: 'verdicts' }, chairV.map(verdictCard)));
    }
    if (r) { box.append(el('h3', { class: 'h-pixel', text: 'THE TAKE' }), metersBlock(r), warningsBlock(r.warnings)); }
    if (score || state.ctx === 'explore') {
      box.append(el('section', { class: 'reveal' },
        el('h3', { class: 'h-pixel', text: 'THE REAL CREW' }),
        el('div', { class: 'codec', text: h.codec }),
        el('p', { text: h.reveal })));
    }
    const nextLabel = state.ctx === 'explore' ? 'BACK TO EXPLORE' : state.pendingQuiz ? 'TAKE THE CHECKPOINT' : 'NEXT HEIST';
    box.append(el('div', { class: 'results-actions' },
      el('button', { type: 'button', class: 'btn btn-ghost btn-big', id: 'btn-try-again', text: 'TRY AGAIN', onclick: () => { show('squad'); requestAnimationFrame(drawCanvases); } }),
      el('button', { type: 'button', class: `btn ${state.pendingQuiz ? 'btn-gold' : 'btn-run'} btn-big`, id: 'btn-next-heist', text: nextLabel, onclick: nextHeist })));
  }

  // ---------------- checkpoint quiz ----------------
  function startQuiz(lv) {
    const qs = CYC.QUIZZES && CYC.QUIZZES[lv.id];
    if (!qs || !qs.length) { enterLevel(lv); return; }
    state.level = lv;
    state.quiz = { level: lv, i: 0, picked: null, correct: 0, done: false };
    renderQuiz();
    show('quiz');
  }
  function renderQuiz() {
    const Q = state.quiz;
    const box = clearEl($('quiz'));
    if (!Q) return;
    const qs = CYC.QUIZZES[Q.level.id];
    box.append(el('div', { class: 'screen-head' },
      el('button', { type: 'button', class: 'btn btn-small btn-ghost', text: '← HEISTS', onclick: () => enterLevel(Q.level) }),
      el('h2', { class: 'h-pixel', text: `CHECKPOINT · LV ${Q.level.id} ${Q.level.name}` })));
    if (Q.done) {
      const passed = Q.correct >= 3;
      box.append(el('div', { class: 'quiz-card quiz-final' },
        el('div', { class: `rank-letter ${passed ? 'rank-A' : 'rank-C'}`, text: `${Q.correct}/${qs.length}` }),
        el('div', { class: 'stack' },
          el('div', { class: 'results-headline', text: passed ? 'LEVEL CLEARED' : 'NOT YET: READ THE WHYS AND RETAKE' }),
          el('p', { class: 'dim', text: passed ? 'The Fixer stamps your file. The checkpoint teaches; the next level was already unlocked by the heists.' : 'Three of four clears the checkpoint. Nothing is gated on it: the next level is open either way.' }),
          el('div', { class: 'row' },
            passed ? null : el('button', { type: 'button', class: 'btn btn-gold', text: 'RETAKE', onclick: () => startQuiz(Q.level) }),
            el('button', { type: 'button', class: 'btn btn-run', text: 'CONTINUE', onclick: () => { renderLevels(); show('levels'); } })))));
      return;
    }
    const q = qs[Q.i];
    box.append(el('div', { class: 'quiz-progress' }, qs.map((_, i) => el('i', { class: `qdot ${i < Q.i ? 'done' : i === Q.i ? 'now' : ''}` }))));
    const card = el('div', { class: 'quiz-card' },
      el('div', { class: 'pix-sm dim', text: `QUESTION ${Q.i + 1} OF ${qs.length}` }),
      el('p', { class: 'quiz-q', text: q.q }),
      el('div', { class: 'quiz-options' }, q.options.map((opt, k) => el('button', {
        type: 'button', id: `quiz-opt-${k}`,
        class: `quiz-opt ${Q.picked == null ? '' : k === q.answer ? 'is-right' : k === Q.picked ? 'is-wrong' : 'is-off'}`,
        disabled: Q.picked != null,
        onclick: () => { Q.picked = k; if (k === q.answer) Q.correct++; renderQuiz(); },
      }, el('span', { class: 'quiz-letter', text: 'ABCD'[k] }), el('span', { text: opt })))),
    );
    if (Q.picked != null) {
      const right = Q.picked === q.answer;
      card.append(el('div', { class: `quiz-why ${right ? 'right' : 'wrong'}` },
        el('div', { class: 'pix', text: right ? 'CORRECT' : 'INCORRECT' }),
        el('p', { text: q.why })),
        el('button', { type: 'button', class: 'btn btn-run', id: 'btn-quiz-next', text: Q.i + 1 < qs.length ? 'NEXT QUESTION' : 'FINISH', onclick: () => {
          if (Q.i + 1 < qs.length) { Q.i++; Q.picked = null; renderQuiz(); }
          else {
            Q.done = true;
            const passed = Q.correct >= 3;
            const prev = state.progress.quiz[Q.level.id];
            state.progress.quiz[Q.level.id] = { score: Math.max(Q.correct, prev ? prev.score : 0), passed: passed || !!(prev && prev.passed) };
            saveProgress();
            renderQuiz();
          }
        } }));
      setTimeout(() => { const b = $('btn-quiz-next'); if (b) b.focus(); }, 0);
    }
    box.append(card);
  }

  // ---------------- side quests ----------------
  function openQuest(q) {
    state.quest = { id: q.id, q, step: 0, done: false, result: null };
    renderQuest();
    show('quest');
  }
  function renderQuest() {
    const S = state.quest;
    const box = clearEl($('quest'));
    if (!S) return;
    const q = S.q;
    const howto = q.howto || [];
    const atWatch = S.step >= howto.length;
    box.append(el('div', { class: 'screen-head' },
      el('button', { type: 'button', class: 'btn btn-small btn-ghost', text: '← SIDE QUESTS', onclick: () => enterLevel(state.level, 'quests') }),
      el('h2', { class: 'h-pixel', text: q.name }),
      el('span', { class: 'dim', text: q.concept })));
    const fixerBox = el('div', { class: 'quest-fixer' },
      el('div', { class: 'portrait portrait-fixer', style: { '--rc': '#777' } }, el('span', { class: 'initials', text: 'F' }), el('img', { alt: '', src: (CYC.TUTOR && CYC.TUTOR.portrait) || 'img/fighters/none.png', onerror: (e) => e.target.remove() })),
      el('div', { class: 'stack' },
        el('div', { class: 'pix', style: { color: 'var(--gold)' }, text: (CYC.TUTOR && CYC.TUTOR.name) || 'THE FIXER' }),
        atWatch
          ? el('p', { class: 'quest-watch', html: `<b>WATCH:</b> ${q.watch || ''}` })
          : el('p', { class: 'quest-line', html: howto[S.step] }),
        atWatch ? null : el('div', { class: 'row' },
          el('span', { class: 'pix-sm dim', text: `${S.step + 1}/${howto.length}` }),
          el('button', { type: 'button', class: 'btn btn-small', id: 'btn-quest-next', text: S.step + 1 < howto.length ? 'NEXT ▶' : 'GOT IT', onclick: () => { S.step++; renderQuest(); } }))));
    box.append(fixerBox);
    const field = el('div', { class: 'quest-field cyc-quest', id: 'quest-field', style: { '--role-color': roleColor((fighter((q.fighters || [])[0]) || {}).role || 'predict') } });
    box.append(field);
    if (S.done && S.result) {
      const res = S.result;
      box.append(el('div', { class: 'quest-result' },
        el('div', { class: 'row' }, el('span', { class: 'pix', style: { color: 'var(--gold)' }, text: 'DEBRIEF' }), el('span', { class: `rank-badge ${res.score >= 50 ? 'rank-A' : 'rank-C'}`, text: String(Math.round(res.score)) })),
        res.summary ? el('p', { class: 'body-lg', text: res.summary }) : null,
        el('p', { html: q.debrief || '' }),
        res.score < 50 && q.fail ? el('div', { class: 'callout', text: q.fail }) : null,
        el('div', { class: 'row' },
          el('button', { type: 'button', class: 'btn btn-ghost', text: 'PLAY AGAIN', onclick: () => openQuest(q) }),
          el('button', { type: 'button', class: 'btn btn-run', text: 'BACK TO SIDE QUESTS', onclick: () => enterLevel(state.level, 'quests') }))));
    }
    if (!S.mounted) {
      S.mounted = true;
      if (!questsReady()) {
        field.append(el('div', { class: 'placeholder', text: 'QUEST MODULE LOADING…' }), el('p', { class: 'dim', text: 'CYC.Quests is not available yet (js/quests.js). The playfield appears once it loads.' }));
      } else {
        requestAnimationFrame(() => {
          try {
            CYC.Quests.render(field, q.id, { onComplete: (result) => {
              S.done = true; S.result = result || { score: 100, summary: '' };
              const sc = Math.round(S.result.score == null ? 100 : S.result.score);
              const prev = state.progress.quests[q.id];
              state.progress.quests[q.id] = prev == null ? sc : Math.max(prev, sc);
              saveProgress();
              S.keepField = field;
              renderQuestResultOnly();
            } });
          } catch (e) {
            if (window.console) console.error('CYC.Quests.render', e);
            field.append(el('div', { class: 'callout callout-red', text: `Quest failed to load: ${e.message || e}` }));
          }
        });
      }
    } else if (S.keepField) {
      field.replaceWith(S.keepField);
    }
  }
  // After completion, re-render the frame but keep the playfield DOM the quest module owns.
  function renderQuestResultOnly() {
    const S = state.quest;
    const field = $('quest-field');
    S.keepField = field;
    renderQuest();
    setTimeout(() => { const r = document.querySelector('.quest-result'); if (r) r.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, 0);
  }

  // ---------------- explore ----------------
  function setExploreTab(tab) {
    state.exploreTab = tab;
    document.querySelectorAll('#explore-tabs .tab').forEach((b) => {
      const on = b.dataset.tab === tab;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-selected', String(on));
    });
    renderExplore();
    if (tab === 'duel') fixer('duel_intro');
    if (tab === 'matrix') fixer('matrix_intro');
  }
  function renderExplore() {
    const body = clearEl($('explore-body'));
    const tab = state.exploreTab;
    if (tab === 'codecs') body.append(renderCodecs());
    else if (tab === 'roster') body.append(renderRoster());
    else if (tab === 'jobs') body.append(renderJobs());
    else if (tab === 'matrix') body.append(renderMatrix());
    else if (tab === 'duel') body.append(renderDuel());
    else if (tab === 'tradeoffs') body.append(renderTradeoffs());
  }
  function renderCodecs() {
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Every real codec is a squad. Click one to load it into the squad room on its own heist, with the live pipeline running. All chairs are visible in Explore mode and you can swap fighters freely.' }),
      el('div', { class: 'codec-grid' }, CYC.CODECS.map((c) => {
        const h = CYC.HEISTS.find((x) => x.id === c.heist);
        const members = ROLE_ORDER.flatMap((role) => (c.squad[role] || []).map((id) => fighter(id)).filter(Boolean));
        return el('button', {
          type: 'button', class: 'codec-card', id: `codec-${c.id}`,
          onclick: () => { if (h) openHeist(h, { ctx: 'explore', squad: c.squad, codec: c }); },
        },
          el('div', { class: 'row' }, el('span', { class: 'codec-name', text: c.name }), el('span', { class: 'codec-year', text: String(c.year) })),
          members.length
            ? el('div', { class: 'mini-squad' }, members.map((f) => el('span', { class: 'mini' }, portrait(f), el('span', { class: 'mini-name', text: f.name }))))
            : el('div', { class: 'empty-squad', text: 'Nobody at the table.' }),
          el('p', { class: 'codec-notes', text: c.notes }),
          h ? el('div', { class: 'codec-heist', text: `HEIST: ${h.name} · ${payloadLabel(h.payload)}` }) : null,
        );
      })));
  }
  function renderRoster() {
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Every fighter in the roster, grouped by chair and family. Click a card for the full fighter file and its Special Move training lab.' }),
      ROLE_ORDER.map((role) => {
        const R = CYC.ROLES[role];
        return el('section', { class: 'roster-role', style: { '--rc': R.color }, id: `roster-${role}` },
          el('div', { class: 'roster-role-head' }, el('h3', { class: 'h-pixel', text: R.codename }), el('span', { class: 'role-title', text: `${R.title} · "${R.question}"` })),
          el('p', { class: 'roster-blurb', text: R.blurb }),
          el('div', { class: 'chips' }, (R.jargon || []).map((j) => chip(j, null, true))),
          familiesFor(role).map((fam) => el('div', { class: 'roster-fam' },
            el('div', { class: 'roster-fam-head', text: fam.name }, el('small', { text: fam.note })),
            el('div', { class: 'fighter-grid' }, CYC.FIGHTERS.filter((f) => f.family === fam.id).map((f) =>
              fighterCard(f, { showLevel: true, onclick: () => openDetail(f) }))))),
        );
      }));
  }
  function renderJobs() {
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'All heists side by side: the requirement profile is the job\'s stat sheet, and the real crew is named. Open any one in the squad room with all chairs visible.' }),
      el('div', { class: 'heist-grid' }, CYC.HEISTS.map((h) => heistCard(h, {
        showCodec: true, showLevel: true, idSuffix: '-job', buttonText: 'OPEN SQUAD ROOM',
        onopen: (hh) => openHeist(hh, { ctx: 'explore' }),
      }))));
  }
  function renderMatrix() {
    const mark = (v) => (v === 2 ? el('span', { class: 'cell-yes', text: '✔', 'aria-label': 'satisfies' }) : v === 1 ? el('span', { class: 'cell-mid', text: '~', 'aria-label': 'partly' }) : el('span', { class: 'cell-no', text: '✘', 'aria-label': 'does not' }));
    const rows = [];
    for (const role of ROLE_ORDER) {
      const R = CYC.ROLES[role];
      rows.push(el('tr', { class: 'role-row', style: { '--rc': R.color } }, el('td', { colspan: String(MATRIX_COLS.length + 1), text: `${R.codename} · ${R.title}` })));
      for (const f of CYC.FIGHTERS.filter((x) => x.role === role)) {
        rows.push(el('tr', {},
          el('td', { class: 'rowhead' }, f.name, el('small', { text: `${f.real} · LV ${f.level}` })),
          MATRIX_COLS.map((c) => el('td', {}, mark(c.f(f.stats))))));
      }
    }
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Which fighter properties satisfy which job requirements, derived from the stat sheets. ✔ satisfies, ~ partly, ✘ does not.' }),
      el('div', { class: 'matrix-wrap' },
        el('table', { class: 'matrix' },
          el('thead', {}, el('tr', {}, el('th', { class: 'rowhead', text: 'FIGHTER' }), MATRIX_COLS.map((c) => el('th', { text: c.label })))),
          el('tbody', {}, rows)),
        el('div', { class: 'matrix-legend', text: 'Derivation: lossless ← stats.lossless · low latency ← DELAY ≤1 ✔, 2–3 ~ · cheap decoder ← DEC COST ≤1 ✔, 2–3 ~ · cheap encoder ← ENC COST ≤1 ✔, 2–3 ~ · random access ← RANDOM ACCESS ≥4 ✔, 2–3 ~ · tight bandwidth ← SQUEEZE ≥4 ✔, 3 ~ · small decoder memory ← MEMORY ≤1 ✔, 2–3 ~.' })));
  }
  function openDetail(f) {
    const body = clearEl($('detail-body'));
    body.append(detailCard(f, {}));
    $('overlay-detail').hidden = false;
    document.body.style.overflow = 'hidden';
    setTimeout(() => { const b = $('btn-close-detail'); if (b) b.focus(); }, 0);
  }
  function closeDetail() {
    $('overlay-detail').hidden = true;
    document.body.style.overflow = '';
  }

  // ---------------- explore: DUEL ----------------
  function duelFromPreset(idx) {
    const p = (CYC.DUELS || [])[idx];
    if (!p) return null;
    const fa = fighter(p.a), fb = fighter(p.b);
    const role = p.role || (fa && fa.id !== 'none' ? fa.role : fb ? fb.role : 'entropy');
    return { payload: p.payload, role, a: p.a, b: p.b, squadBase: Object.assign({}, p.squadBase || {}), lesson: p.lesson || '', preset: idx };
  }
  function ensureDuel() {
    if (!state.duel) state.duel = duelFromPreset(0) || { payload: 'json', role: 'entropy', a: 'huffman', b: 'arith', squadBase: { predict: ['lz77'] }, lesson: '', preset: null };
    return state.duel;
  }
  function openDuelPreset(idx) {
    state.duel = duelFromPreset(idx) || ensureDuel();
    state.exploreTab = 'duel';
    if (state.mode !== 'explore' || state.screen !== 'explore') { setMode('explore'); }
    setExploreTab('duel');
  }
  function select(id, options, value, onchange) {
    const s = el('select', { id, class: 'sel', onchange: (e) => onchange(e.target.value) });
    for (const o of options) s.append(el('option', { value: o.value, selected: o.value === value ? true : null, text: o.label }));
    return s;
  }
  function renderDuel() {
    const D = ensureDuel();
    const roleFighters = [{ value: 'none', label: 'EMPTY CHAIR' }].concat(CYC.FIGHTERS.filter((f) => f.role === D.role).map((f) => ({ value: f.id, label: `${f.name} (${f.real})` })));
    const change = (patch) => { Object.assign(D, patch); D.preset = null; D.lesson = ''; renderExplore(); };
    const controls = el('div', { class: 'duel-controls' },
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'PAYLOAD' }), select('duel-payload', PAYLOADS.map((p) => ({ value: p, label: payloadLabel(p) })), D.payload, (v) => change({ payload: v }))),
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'CHAIR' }), select('duel-role', ROLE_ORDER.map((r) => ({ value: r, label: `${CYC.ROLES[r].codename} · ${CYC.ROLES[r].title}` })), D.role, (v) => {
        const fs = CYC.FIGHTERS.filter((f) => f.role === v);
        change({ role: v, a: fs[0] ? fs[0].id : 'none', b: fs[1] ? fs[1].id : 'none', squadBase: {} });
      })),
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'FIGHTER A' }), select('duel-a', roleFighters, D.a, (v) => change({ a: v }))),
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'FIGHTER B' }), select('duel-b', roleFighters, D.b, (v) => change({ b: v }))),
    );
    const baseIds = ROLE_ORDER.flatMap((r) => (D.squadBase[r] || []).filter((id) => r !== D.role)).map(fighter).filter(Boolean);
    const presets = el('div', { class: 'preset-row' },
      el('span', { class: 'pix-sm dim', text: 'PRESETS:' }),
      (CYC.DUELS || []).map((p, i) => el('button', {
        type: 'button', class: `btn btn-small ${D.preset === i ? '' : 'btn-ghost'}`, id: `duel-preset-${i}`,
        text: `${(fighter(p.a) || {}).name || p.a} vs ${(fighter(p.b) || {}).name || p.b} · ${p.payload}`,
        onclick: () => { state.duel = duelFromPreset(i); renderExplore(); },
      })));
    const out = el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Same chair, same data, two fighters. The rest of the squad stays fixed; only one seat changes.' }),
      controls, presets,
      baseIds.length ? el('div', { class: 'row' }, el('span', { class: 'pix-sm dim', text: 'FIXED SQUAD:' }), baseIds.map((f) => el('span', { class: 'mini' }, portrait(f, 'portrait-xs'), el('span', { class: 'mini-name', text: f.name })))) : el('p', { class: 'dim', text: 'Fixed squad: nobody else at the table.' }),
      D.lesson ? el('div', { class: 'callout callout-lesson', text: D.lesson }) : null);
    if (!engineReady()) { out.append(enginePlaceholder()); return out; }
    const squadA = Object.assign({}, D.squadBase, { [D.role]: [D.a] });
    const squadB = Object.assign({}, D.squadBase, { [D.role]: [D.b] });
    const ra = runSquad(D.payload, squadA), rb = runSquad(D.payload, squadB);
    const column = (f, run, tag) => el('div', { class: 'duel-col', style: { '--rc': f && f.id !== 'none' ? roleColor(f.role) : '#777' } },
      el('div', { class: 'duel-col-head' }, portrait(f || CYC.EMPTY), el('div', {}, el('div', { class: 'pix-sm dim', text: `FIGHTER ${tag}` }), el('div', { class: 'stage-name', text: f ? f.name : '?' }))),
      run.error ? warningsBlock([`Engine error: ${run.error.message || run.error}`], 'callout-red')
        : run.result && run.result.stages.length ? el('div', { class: 'stage-stack' }, run.result.stages.map(stageCard))
          : el('div', { class: 'placeholder', text: 'NO STAGES' }),
      run.result ? warningsBlock(run.result.warnings) : null);
    const fa = fighter(D.a) || CYC.EMPTY, fb = fighter(D.b) || CYC.EMPTY;
    out.append(el('div', { class: 'duel-cols' }, column(fa, ra, 'A'), column(fb, rb, 'B')));
    if (ra.result && rb.result) {
      const A = ra.result, B = rb.result;
      const row = (label, va, vb, better) => el('tr', {}, el('th', { text: label }),
        el('td', { class: better === 'a' ? 'duel-win' : '' , text: va }), el('td', { class: better === 'b' ? 'duel-win' : '', text: vb }));
      const less = (x, y) => (x === y ? null : x < y ? 'a' : 'b');
      const more = (x, y) => (x === y ? null : x > y ? 'a' : 'b');
      const ratioA = A.finalBits ? A.rawBits / A.finalBits : 0, ratioB = B.finalBits ? B.rawBits / B.finalBits : 0;
      out.append(el('div', { class: 'matrix-wrap' }, el('table', { class: 'matrix duel-table' },
        el('thead', {}, el('tr', {}, el('th', { text: 'METRIC' }), el('th', { text: fa.name }), el('th', { text: fb.name }))),
        el('tbody', {},
          row('ENTROPY BOUND', `${fmtBits(A.entropyBoundBits)} · ${Math.round(A.entropyBoundBits).toLocaleString()} bits`, `${fmtBits(B.entropyBoundBits)} · ${Math.round(B.entropyBoundBits).toLocaleString()} bits`, less(A.entropyBoundBits, B.entropyBoundBits)),
          row('FINAL BITS', `${fmtBits(A.finalBits)} · ${Math.round(A.finalBits).toLocaleString()} bits`, `${fmtBits(B.finalBits)} · ${Math.round(B.finalBits).toLocaleString()} bits`, less(A.finalBits, B.finalBits)),
          row('RATIO', `×${ratioA.toFixed(2)}`, `×${ratioB.toFixed(2)}`, more(ratioA, ratioB)),
          row('DISTORTION', A.lossless ? 'BIT-EXACT' : distortionText(A), B.lossless ? 'BIT-EXACT' : distortionText(B), null),
          row('SIDE INFO', sideInfoLabel(A.sideInfoBytes), sideInfoLabel(B.sideInfoBytes), less(A.sideInfoBytes || 0, B.sideInfoBytes || 0)),
        ))));
      requestAnimationFrame(drawCanvases);
    }
    return out;
  }

  // ---------------- explore: TRADE-OFFS ----------------
  function renderTradeoffs() {
    const T = state.tradeoff;
    const axes = CYC.TRADEOFF_AXES || [];
    const axis = (id) => axes.find((a) => a.id === id) || { id, label: id.toUpperCase(), hi: '' };
    const ax = axis(T.x), ay = axis(T.y);
    const opts = axes.map((a) => ({ value: a.id, label: a.label }));
    const controls = el('div', { class: 'duel-controls' },
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'X AXIS' }), select('tf-x', opts, T.x, (v) => { T.x = v; T.note = null; renderExplore(); })),
      el('label', { class: 'ctl' }, el('span', { class: 'pix-sm', text: 'Y AXIS' }), select('tf-y', opts, T.y, (v) => { T.y = v; T.note = null; renderExplore(); })));
    const presets = el('div', { class: 'preset-row' },
      el('span', { class: 'pix-sm dim', text: 'PRESETS:' }),
      (CYC.TRADEOFF_PRESETS || []).map((p, i) => el('button', {
        type: 'button', class: `btn btn-small ${T.note === p.note ? '' : 'btn-ghost'}`, id: `tf-preset-${i}`, text: p.title,
        onclick: () => { T.x = p.x; T.y = p.y; T.note = p.note; renderExplore(); },
      })));
    // group fighters by cell and spread them deterministically
    const cells = {};
    CYC.FIGHTERS.forEach((f) => { const k = `${f.stats[T.x]},${f.stats[T.y]}`; (cells[k] = cells[k] || []).push(f); });
    const OFF = [[0, 0], [-16, -12], [16, 12], [16, -12], [-16, 12], [0, -22], [0, 22], [-24, 0], [24, 0], [-30, -24], [30, 24], [30, -24]];
    const pts = [];
    for (const group of Object.values(cells)) {
      group.forEach((f, i) => {
        const o = OFF[i % OFF.length];
        pts.push(el('div', {
          class: 'tf-pt', tabindex: '0', title: `${f.name}: ${ax.label} ${f.stats[T.x]}, ${ay.label} ${f.stats[T.y]}`,
          style: { left: `calc(28px + ${(f.stats[T.x] / 5)} * (100% - 56px) + ${o[0]}px)`, bottom: `calc(28px + ${(f.stats[T.y] / 5)} * (100% - 56px) + ${o[1]}px)`, '--rc': roleColor(f.role) },
          onclick: () => openDetail(f),
        }, portrait(f, 'portrait-tf'), el('span', { class: 'tf-name', text: f.name })));
      });
    }
    const plot = el('div', { class: 'tf-plot' },
      el('div', { class: 'tf-area' }, pts),
      el('div', { class: 'tf-xlab' }, el('span', { class: 'pix-sm', text: `${ax.label} →` }), el('span', { class: 'dim', text: ax.hi })),
      el('div', { class: 'tf-ylab' }, el('span', { class: 'pix-sm', text: `${ay.label} ↑` }), el('span', { class: 'dim', text: ay.hi })));
    return el('div', { class: 'stack' },
      el('p', { class: 'tagline', text: 'Every fighter plotted on two stats (0 to 5). Hover or focus a portrait for the name; click it for the fighter file.' }),
      controls, presets,
      T.note ? el('div', { class: 'callout callout-lesson', text: T.note }) : null,
      plot,
      el('div', { class: 'chips' }, ROLE_ORDER.map((r) => chip(CYC.ROLES[r].codename, roleColor(r)))));
  }

  // ---------------- init ----------------
  function init() {
    $('btn-start').addEventListener('click', () => setMode('heist'));
    $('btn-title-explore').addEventListener('click', () => setMode('explore'));
    $('btn-home').addEventListener('click', () => { if (state.chooser.open) closeChooser(); closeDetail(); show('title'); });
    $('mode-heist').addEventListener('click', () => setMode('heist'));
    $('mode-explore').addEventListener('click', () => setMode('explore'));
    $('btn-fixer').addEventListener('click', () => setTutorOn(!state.tutor.on));
    $('btn-fixer-next').addEventListener('click', fixerAdvance);
    $('btn-fixer-close').addEventListener('click', fixerDismiss);
    $('btn-unlock-all').addEventListener('click', () => { state.progress.unlockedAll = true; saveProgress(); renderLevels(); });
    $('btn-reset-progress').addEventListener('click', () => {
      state.progress = { solved: {}, unlockedAll: false, quiz: {}, quests: {} }; saveProgress();
      state.tutor.seen = new Set(); saveTutor();
      renderLevels();
    });
    $('btn-back-levels').addEventListener('click', () => { renderLevels(); show('levels'); });
    $('btn-back-heists').addEventListener('click', () => { if (state.ctx === 'explore') setMode('explore'); else enterLevel(state.level); });
    $('btn-run').addEventListener('click', runHeist);
    $('btn-run-2').addEventListener('click', runHeist);
    ['btn-run', 'btn-run-2'].forEach((id) => { ['mouseenter', 'focus'].forEach((ev) => $(id).addEventListener(ev, () => fixer('before_run'))); });
    $('btn-close-fighter').addEventListener('click', closeChooser);
    $('btn-close-detail').addEventListener('click', closeDetail);
    $('overlay-fighter').addEventListener('click', (e) => { if (e.target === e.currentTarget) closeChooser(); });
    $('overlay-detail').addEventListener('click', (e) => { if (e.target === e.currentTarget) closeDetail(); });
    document.querySelectorAll('#explore-tabs .tab').forEach((b) => b.addEventListener('click', () => setExploreTab(b.dataset.tab)));
    document.querySelectorAll('#heists-subtabs .tab').forEach((b) => b.addEventListener('click', () => setHeistTab(b.dataset.tab)));
    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (state.chooser.open) closeChooser();
      else if (!$('overlay-detail').hidden) closeDetail();
    });
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawCanvases, 150); });

    setTutorOn(state.tutor.on);
    const note = $('title-engine-note');
    const missing = [!engineReady() && 'engine (js/engine.js)', !rulesReady() && 'rules (js/rules.js)', !(CYC.Demos && CYC.Demos.render) && 'labs (js/demos.js)', !questsReady() && 'side quests (js/quests.js)'].filter(Boolean);
    if (missing.length) { note.hidden = false; note.textContent = `Still loading: ${missing.join(', ')}. Those parts show a placeholder until they arrive.`; }

    show('title');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

  return { state, show, setMode, openHeist, openChooser, runHeist, drawCanvases, render: renderPipeline, fixer, startQuiz, openQuest, openDuelPreset, setExploreTab, enterLevel };
})();
