// Choose Your Codec — scoring rules.
// The whole lesson of the game lives here: a requirement of the task meets a property of an algorithm.
// Every "wrong" verdict names the requirement and the property that conflict.

window.CYC = window.CYC || {};

(function () {
  const ROLE_ORDER = Object.keys(CYC.ROLES);

  const LATENCY_WORDS = ['nothing', 'a few samples', 'a whole block', 'a frame plus lookahead', 'several frames', 'the whole file'];

  function fightersIn(squad, role) {
    return (squad[role] || []).filter(id => id && id !== 'none').map(id => CYC.byId[id]).filter(Boolean);
  }

  function stageName(role) {
    const R = CYC.ROLES[role];
    return R ? R.title : role;
  }
  function algoName(id, fallback) {
    const f = CYC.byId && CYC.byId[id];
    return f ? f.name : fallback;
  }

  // Hard conflicts between one requirement and one property. Returns a reason string or null.
  function collide(heist, f) {
    const r = heist.req, s = f.stats;

    if (r.lossless && f.role === 'quantize')
      return `${f.name} is a quantizer: it rounds values onto a grid or a codebook, and the original values cannot be recovered. This task requires a bit-exact output, so the ${stageName('quantize')} stage must stay empty.`;
    if (r.lossless && f.role === 'perceptual')
      return `${f.name} decides which errors the consumer will not notice and allows them. "Not noticed" is not "identical". A lossless task has no ${stageName('perceptual').toLowerCase()}.`;
    if (r.lossless && !s.lossless)
      return `${f.name} is not integer-reversible: real-valued coefficients are rounded somewhere and the round trip is no longer exact. A lossless task needs a lossless ${stageName(f.role).toLowerCase()} stage, or none.`;
    if (!f.domains.includes(heist.domain))
      return `${f.name} is built for ${f.domains.join(' / ')} data. This task is ${heist.domain}. The structure it exploits is not in this data.`;
    if (r.latency <= 1 && s.latency >= 3)
      return `${f.name} must buffer ${LATENCY_WORDS[s.latency]} before it can emit anything. This task tolerates almost no delay.`;
    if (r.decBudget === 0 && s.dec >= 2)
      return `The decoder here is a kernel: one multiply and one add per element, fully parallel. ${f.name} needs real computation (or sequential state) per element to decode.`;
    if (r.decBudget <= 1 && s.dec >= 3)
      return `${f.name} is expensive to decode (decoder cost ${s.dec}/5). This task's decoder runs on a tiny budget: a battery, a browser, a 2005 CPU.`;
    if (r.encBudget <= 1 && s.enc >= 3)
      return `${f.name} spends real compute encoding (encoder cost ${s.enc}/5): searching, fitting, iterating. This task encodes online with almost no budget.`;
    if (r.ra >= 4 && s.ra <= 1)
      return `${f.name} produces variable-length output or depends on everything decoded before it. This task must read any element without touching the rest: fixed-length codes only.`;
    if (r.decMem <= 1 && s.mem >= 3)
      return `${f.name} needs ${s.mem >= 4 ? 'a large' : 'a sizeable'} working set in the decoder. This task's decoder has almost no memory.`;
    if (!r.priorOK && f.role === 'prior' && (f.family === 'trained' || f.family === 'weights'))
      return `${f.name} is data-specific side information. This task's decoders are everywhere and expect a self-describing stream: a dictionary or a model cannot be shipped to all of them.`;
    return null;
  }

  // Soft reasons why a legal but un-ideal pick is not pulling its weight.
  function mehReason(heist, f) {
    const r = heist.req, s = f.stats;
    if (r.ra >= 4 && s.ra === 2) return `${f.name} works, but its codes are variable-length, so random access needs an index on top. The task wanted fixed-length codes.`;
    if (r.bandwidth >= 4 && s.ratio <= 2) return `${f.name} is allowed here but compresses little (ratio ${s.ratio}/5), and this task pays for every bit.`;
    if (r.decBudget <= 1 && s.dec === 2) return `${f.name} is affordable, but a cheaper decoder exists for this task.`;
    if (r.encBudget >= 4 && s.ratio <= 3 && f.role !== 'prior') return `${f.name} leaves ratio unused. This task has encoder compute to spend; a heavier algorithm would pay off.`;
    return `${f.name} is allowed here and does no harm, but the real codec gets more out of this stage. Weakness: ${f.weakness}`;
  }

  function idealText(heist, f) {
    return `Correct. ${f.move} This is what ${heist.codec.split(' (')[0]} does.`;
  }

  function acceptText(heist, f) {
    return `Valid. The real codec chose differently, but nothing in this task rules ${f.name} out. ${f.why.split('. ')[0]}.`;
  }

  function emptyVerdict(heist, role, roles, squad) {
    const ideal = (heist.ideal || {})[role] || [];
    const accept = (heist.accept || {})[role] || [];
    const R = CYC.ROLES[role];
    if (ideal.includes('none'))
      return { ok: 'ideal', text: emptyIdealText(heist, role) };
    if (accept.includes('none'))
      return { ok: 'accept', text: `No ${R.title.toLowerCase()} stage, and that is a legitimate design here. ${emptyNeutralText(heist, role)}` };
    if (role === 'predict' && allIds(squad).has('bwt'))
      return { ok: 'accept', text: `No predictor, but ${algoName('bwt', 'BWT')} in the ${stageName('transform')} stage is doing the modeling: after BWT the structure shows up as runs, which the entropy coder reads directly.` };
    if (ideal.length === 0) {
      return { ok: 'ideal', text: `${R.title} not needed. ${emptyNeutralText(heist, role)}` };
    }
    // Ideal wanted an algorithm here.
    if (role === 'entropy' && heist.req.lossless)
      return { ok: 'wrong', text: `No entropy coder. Every stage upstream only reshaped the symbol distribution; without an entropy coder each symbol still costs its full fixed width, and the modeling saved zero bits.` };
    if (role === 'quantize' && !heist.req.lossless)
      return { ok: 'wrong', text: `The one stage allowed to discard information is empty. A lossless pipeline on ${heist.domain} data tops out near 2:1; this task needs far more, and the consumer would not notice the loss.` };
    if (role === 'prior')
      return { ok: 'meh', text: `Self-describing stream. Valid, but this task allows side information, and that is where the real codec finds its advantage.` };
    return { ok: 'meh', text: `${R.title} stage empty. The data has structure this stage exploits (${R.question}) and nothing is exploiting it.` };
  }

  function emptyIdealText(heist, role) {
    if (role === 'entropy')
      return `Correct: no entropy coder. Fixed-length codes are the point here: random access and a kernel-speed decoder beat the last 10-20% of ratio. (GGUF, KV caches and Faiss PQ all leave this stage empty.)`;
    if (role === 'transform' && heist.id === 'kvcache')
      return `Fine: quantize in place. A Hadamard rotation would reduce the error from outlier channels, but it is optional.`;
    return `Correct: this stage stays empty on purpose.`;
  }

  function emptyNeutralText(heist, role) {
    const map = {
      transform: 'This data has no basis in which it compacts better than the one it arrived in.',
      predict: 'Nothing here to predict that the other stages do not already exploit.',
      quantize: 'Lossless task: no rounding is allowed.',
      entropy: 'Fixed-length output is what the task wants.',
      perceptual: 'No perceptual model applies: there is no consumer whose tolerance for error can be exploited.',
      prior: 'A self-describing stream is exactly right: nothing to ship, nothing to version.',
    };
    return map[role] || '';
  }

  function score(heist, squad, result, roles) {
    roles = roles || (CYC.LEVELS.find(l => l.id === heist.level) || CYC.LEVELS[3]).roles;
    const ideal = heist.ideal || {}, accept = heist.accept || {};
    const verdicts = [];
    let points = 0, weight = 0;

    for (const role of ROLE_ORDER) {
      if (!roles.includes(role)) continue;
      const w = ideal[role] && ideal[role].length ? 2 : 1;
      const picks = fightersIn(squad, role);

      if (picks.length === 0) {
        const v = emptyVerdict(heist, role, roles, squad);
        verdicts.push({ role, fighterId: null, ok: v.ok, text: v.text });
        points += w * ({ ideal: 100, accept: 75, meh: 50, wrong: 0 })[v.ok];
        weight += w;
        continue;
      }

      let rolePoints = 0;
      for (const f of picks) {
        let ok, text;
        const hard = collide(heist, f);
        if (hard) { ok = 'wrong'; text = hard; }
        else if ((ideal[role] || []).includes(f.id)) { ok = 'ideal'; text = idealText(heist, f); }
        else if ((accept[role] || []).includes(f.id)) { ok = 'accept'; text = acceptText(heist, f); }
        else { ok = 'meh'; text = mehReason(heist, f); }
        verdicts.push({ role, fighterId: f.id, ok, text });
        rolePoints += ({ ideal: 100, accept: 75, meh: 40, wrong: 0 })[ok];
      }
      // A wrong pick zeroes the stage; otherwise average the picks.
      const anyWrong = verdicts.slice(-picks.length).some(v => v.ok === 'wrong');
      points += w * (anyWrong ? 0 : rolePoints / picks.length);
      weight += w;
    }

    // Pipeline-level interactions: synergies and conflicts between stages.
    const combos = activeCombos(squad, heist);
    const anti = activeAnticombos(squad, heist, roles);
    for (const c of combos) {
      verdicts.push({ role: 'combo', fighterId: null, ok: 'ideal', text: `Synergy: ${c.name}. ${c.text}`, combo: c.id });
    }
    for (const a of anti) {
      verdicts.push({ role: 'combo', fighterId: null, ok: 'wrong', text: `Conflict: ${a.name}. ${a.text}`, anticombo: a.id });
      points -= 8;
    }
    points += 3 * combos.length;

    const quant = fightersIn(squad, 'quantize'), prior = fightersIn(squad, 'prior');
    if (roles.includes('prior') && quant.some(f => f.family === 'vector') && !prior.some(f => f.id === 'codebook' || f.id === 'tables')) {
      verdicts.push({ role: 'prior', fighterId: null, ok: 'meh',
        text: `A vector quantizer's indices point into a codebook, and the decoder must already hold it. Put the ${algoName('codebook', 'VQ codebook')} in the ${stageName('prior')} stage, or the indices are meaningless integers.` });
      points -= 5;
    }
    if (roles.includes('prior') && fightersIn(squad, 'transform').some(f => f.id === 'neural_enc') && !prior.some(f => f.id === 'decoder_net')) {
      verdicts.push({ role: 'prior', fighterId: null, ok: 'meh',
        text: `A learned encoder produces latents only its matching decoder can read. The decoder network (${algoName('decoder_net', 'Learned decoder')}) is side information the receiver must already have.` });
      points -= 5;
    }
    if (result && heist.req.lossless && result.finalBits > result.rawBits) {
      verdicts.unshift({ role: 'entropy', fighterId: null, ok: 'wrong',
        text: `Your pipeline made the data bigger (${fmtBits(result.finalBits)} out of ${fmtBits(result.rawBits)} in). Header and table overhead exceeded the savings. On tiny payloads this is the normal failure of gzip, and the reason trained dictionaries exist.` });
      points -= 10 * (weight / 1);
    }

    let total = Math.max(0, Math.min(100, Math.round(points / Math.max(weight, 1))));
    // A hard conflict anywhere caps the score: an otherwise perfect pipeline with one algorithm that violates a requirement cannot pass.
    if (verdicts.some(v => v.ok === 'wrong')) total = Math.min(total, 60);
    const rank = total >= 95 ? 'S' : total >= 85 ? 'A' : total >= 70 ? 'B' : total >= 50 ? 'C' : 'D';

    let headline;
    const wrong = verdicts.filter(v => v.ok === 'wrong');
    if (rank === 'S') headline = 'Exactly the real codec\'s choices.';
    else if (rank === 'A') headline = 'A different pipeline than the real codec, equally valid for this task.';
    else if (wrong.length) headline = wrong[0].role === 'combo' ? `Conflict between stages. ${firstSentence(wrong[0].text.replace(/^Conflict: [^.]*\. /, ''))}` : `Conflict at the ${stageName(wrong[0].role)} stage. ${firstSentence(wrong[0].text)}`;
    else if (rank === 'B') headline = 'The task is solved, but a better-matched pipeline would compress more.';
    else headline = 'No hard conflicts, but most of the available compression was left unused.';

    return { score: total, rank, verdicts, headline };
  }

  function allIds(squad) {
    return new Set(Object.values(squad).flat().filter(id => id && id !== 'none'));
  }
  function activeCombos(squad, heist) {
    const ids = allIds(squad);
    return CYC.COMBOS.filter(c => c.needs.every(id => ids.has(id)));
  }
  function activeAnticombos(squad, heist, roles) {
    const ids = allIds(squad);
    return CYC.ANTICOMBOS.filter(a =>
      a.needs.every(id => ids.has(id)) &&
      (a.lacks || []).every(id => !ids.has(id)) &&
      (!a.domains || a.domains.includes(heist.domain)) &&
      (!a.roles || a.roles.every(r => roles.includes(r))));
  }

  function firstSentence(s) { const m = s.match(/^.*?[.!?](?=\s|$)/); return m ? m[0] : s; }
  function fmtBits(b) {
    if (b >= 8 * 1024 * 1024) return (b / 8 / 1024 / 1024).toFixed(1) + ' MB';
    if (b >= 8 * 1024) return (b / 8 / 1024).toFixed(1) + ' KB';
    return Math.round(b / 8) + ' B';
  }

  CYC.Rules = { score, collide, fmtBits, activeCombos, activeAnticombos };
})();
