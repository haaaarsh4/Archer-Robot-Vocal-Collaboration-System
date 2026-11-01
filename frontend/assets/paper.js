// The reading pages' small instruments.
// Loaded after site.js. Every function is global because the markup calls it
// from inline handlers, the same convention the rest of the site uses.

// ---------------------------------------------------------------- selector
(function () {
  const state = { protocol: true, sensitive: false, voiced: true, ended: false,
                  fusion: false, fast: false, long: false };
  const NAMES = {
    silent: 'protocol_silence', call: 'call_and_response', drone: 'drone_support',
    triadic: 'triadic_harmony', contour: 'contour_following',
    timbral: 'timbral_thickening', unison: 'unison_shadowing'
  };
  const WHY = {
    silent: 'Sovereignty is off, or the current phoneme is in the protected set. Nothing else is consulted; the answer is silence.',
    call: 'No pitch is present and the phrase just ended. The responder composes an answer from the phrase that finished.',
    drone: 'No pitch is present and the phrase has not ended. The drone holds the floor until the singer returns.',
    triadic: 'Fusion mode is on and a singer is present. The chord tone is chosen above the melody, then snapped to the inferred key.',
    contour: 'The tempo is at or above the fast threshold. The shadow follows the shape of the line rather than a fixed interval.',
    timbral: 'The note has been held for thirty frames. The pitch stays the same and the colour changes.',
    unison: 'A singer is present, the tempo is calm, and the note is new. The configured default answers.'
  };
  window.hsToggle = function (key) {
    const order = ['protocol', 'sensitive', 'voiced', 'ended', 'fusion', 'fast', 'long'];
    if (key === 'ended' && !state.ended) state.voiced = false;
    if (key === 'voiced' && !state.voiced) state.ended = false;
    state[key] = !state[key];
    order.forEach(k => {
      const el = document.getElementById('hs-' + k);
      if (!el) return;
      el.classList.toggle('is-on', !!state[k]);
      if (k === 'protocol') el.textContent = state.protocol ? 'ON' : 'OFF';
    });
    hsEvaluate();
  };
  function hsEvaluate() {
    let mode;
    if (!state.protocol || state.sensitive) mode = 'silent';
    else if (!state.voiced) mode = state.ended ? 'call' : 'drone';
    else if (state.fusion) mode = 'triadic';
    else if (state.fast) mode = 'contour';
    else if (state.long) mode = 'timbral';
    else mode = 'unison';
    const out = document.getElementById('hs-out');
    if (out) out.innerHTML = '<b>' + NAMES[mode] + '</b>\n' + WHY[mode];
  }
  window.hsEvaluate = hsEvaluate;
})();

// ---------------------------------------------------------------- formants
(function () {
  const FORMANT = {
    a: { f1: 730, f2: 1090, f3: 2440, label: 'aah', note: 'A bright, open vowel. The first formant sits high, the second close enough to it to keep the sound round.' },
    e: { f1: 530, f2: 1840, f3: 2480, label: 'eh', note: 'The front of the mouth lifts. F1 drops and F2 rises, which is what makes the vowel read as brighter than aah.' },
    i: { f1: 270, f2: 2290, f3: 3010, label: 'ee', note: 'The closest, brightest vowel. F1 is low and F2 is high; the gap between them is the shape of a smile.' },
    o: { f1: 570, f2: 840, f3: 2410, label: 'oh', note: 'A round back vowel. Both formants sit low and close together, which is the acoustic signature of a rounded mouth.' },
    u: { f1: 300, f2: 870, f3: 2240, label: 'oo', note: 'The darkest of the five. F1 is very low and F2 sits near it, so the spectrum has one broad low peak.' }
  };
  let vowel = 'a';
  let ctx = null, nodes = null;

  function hz(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }
  function midiName(m) {
    const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    return names[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
  }
  function formants() {
    const f = FORMANT[vowel];
    const bright = Number(document.getElementById('fs-bright').value) / 100;
    const nasal = Number(document.getElementById('fs-nasal').value) / 100;
    return {
      f1: f.f1 * (1 + 0.06 * nasal),
      f2: f.f2 * (1 + 0.18 * bright) * (1 - 0.05 * nasal),
      f3: f.f3 * (1 + 0.10 * bright),
      label: f.label, note: f.note
    };
  }
  window.fsVowel = function (v, btn) {
    vowel = v;
    document.querySelectorAll('[data-vowel]').forEach(b => b.classList.remove('is-on'));
    if (btn) btn.classList.add('is-on');
    fsUpdate();
    if (nodes) buildNodes();
  };
  window.fsUpdate = function () {
    const pitch = Number(document.getElementById('fs-pitch').value);
    const bright = Number(document.getElementById('fs-bright').value) / 100;
    const nasal = Number(document.getElementById('fs-nasal').value) / 100;
    const f = formants();
    const midi = Math.round(69 + 12 * Math.log2(pitch / 440));
    document.getElementById('fs-pitch-out').textContent = midiName(midi) + ' · ' + pitch.toFixed(1) + ' Hz';
    document.getElementById('fs-bright-out').textContent = bright.toFixed(2);
    document.getElementById('fs-nasal-out').textContent = nasal.toFixed(2);
    document.getElementById('fs-readout').innerHTML =
      'F1 ' + Math.round(f.f1) + ' Hz · F2 ' + Math.round(f.f2) + ' Hz · F3 ' + Math.round(f.f3) + ' Hz\n' + f.note;
    draw(f, bright, nasal);
    if (nodes) buildNodes();
  };
  function draw(f, bright, nasal) {
    const canvas = document.getElementById('fs-canvas');
    if (!canvas) return;
    const g = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    g.clearRect(0, 0, W, H);
    g.fillStyle = '#0b1120'; g.fillRect(0, 0, W, H);
    g.strokeStyle = 'rgba(111,140,255,0.18)'; g.lineWidth = 1;
    for (let x = 0; x <= W; x += W / 8) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    g.beginPath();
    const peaks = [[f.f1, 1.0], [f.f2, 0.72 + 0.2 * bright], [f.f3, 0.4 + 0.3 * bright]];
    for (let x = 0; x <= W; x++) {
      const hz = 80 + (x / W) * 3600;
      let y = 0.05;
      peaks.forEach(([c, a], i) => {
        const bw = i === 0 ? 90 : 130;
        y += a * Math.exp(-Math.pow((hz - c) / bw, 2));
        if (nasal > 0.02) y += 0.18 * nasal * Math.exp(-Math.pow((hz - 260) / 120, 2));
      });
      y += 0.03 * Math.exp(-hz / 900);
      const py = H - 12 - Math.min(1, y) * (H - 26);
      if (x === 0) g.moveTo(x, py); else g.lineTo(x, py);
    }
    g.lineTo(W, H); g.lineTo(0, H); g.closePath();
    const grad = g.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(111,140,255,0.55)'); grad.addColorStop(1, 'rgba(111,140,255,0.03)');
    g.fillStyle = grad; g.fill();
    g.strokeStyle = '#8ea6ff'; g.lineWidth = 1.4; g.stroke();
    g.fillStyle = 'rgba(224,160,106,0.9)';
    peaks.forEach(([c]) => {
      const x = ((c - 80) / 3600) * W;
      g.fillRect(x - 1, 10, 2, H - 22);
    });
    g.fillStyle = '#6f86aa'; g.font = '9px "Space Mono", monospace';
    g.fillText('80 Hz', 6, H - 3); g.fillText('3.6 kHz', W - 48, H - 3);
  }
  function buildNodes() {
    if (!ctx) return;
    const pitch = Number(document.getElementById('fs-pitch').value);
    const f = formants();
    if (nodes) {
      nodes.osc.frequency.setTargetAtTime(pitch, ctx.currentTime, 0.05);
      nodes.bands.forEach((b, i) => b.frequency.setTargetAtTime([f.f1, f.f2, f.f3][i], ctx.currentTime, 0.05));
    }
  }
  window.fsPlay = function () {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      osc.type = 'sawtooth';
      const gain = ctx.createGain();
      gain.gain.value = 0;
      const out = ctx.createGain();
      out.gain.value = 0.5;
      const bands = [0, 1, 2].map(() => {
        const b = ctx.createBiquadFilter();
        b.type = 'bandpass'; b.Q.value = 7;
        return b;
      });
      const mix = ctx.createGain();
      osc.connect(gain);
      bands.forEach(b => { gain.connect(b); b.connect(mix); });
      mix.connect(out); out.connect(ctx.destination);
      osc.start();
      nodes = { osc, gain, bands, out };
    }
    const f = formants();
    const pitch = Number(document.getElementById('fs-pitch').value);
    nodes.osc.frequency.setValueAtTime(pitch, ctx.currentTime);
    nodes.bands.forEach((b, i) => b.frequency.setValueAtTime([f.f1, f.f2, f.f3][i], ctx.currentTime));
    nodes.gain.gain.cancelScheduledValues(ctx.currentTime);
    nodes.gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    nodes.gain.gain.exponentialRampToValueAtTime(0.20, ctx.currentTime + 0.06);
    nodes.gain.gain.setValueAtTime(0.20, ctx.currentTime + 1.05);
    nodes.gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.65);
    if (ctx.state === 'suspended') ctx.resume();
  };
  window.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('fs-canvas')) fsUpdate();
  });
})();

// ---------------------------------------------------------------- orthography
window.ogConvert = async function () {
  const input = document.getElementById('og-input');
  if (!input) return;
  const out = { sro: document.getElementById('og-sro'), syl: document.getElementById('og-syl'),
                ang: document.getElementById('og-ang'), syls: document.getElementById('og-syllables') };
  out.sro.textContent = '…';
  try {
    const r = await fetch('/api/cree/orthography', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: input.value })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
    out.sro.textContent = d.sro || '·';
    out.syl.textContent = d.syllabics || '·';
    out.ang.textContent = d.anglicized || '·';
    out.syls.textContent = (d.syllables || []).length
      ? 'Syllables: ' + d.syllables.join(' · ')
      : 'No syllables returned.';
  } catch (e) {
    out.sro.textContent = 'Could not reach the converter: ' + e.message;
    out.syl.textContent = '·'; out.ang.textContent = '·';
    out.syls.textContent = '';
  }
};

// ---------------------------------------------------------------- retrieval
window.ragQuery = async function () {
  const input = document.getElementById('rag-input');
  const out = document.getElementById('rag-out');
  if (!input || !out) return;
  out.textContent = 'Searching the notes…';
  try {
    const r = await fetch('/api/chat/retrieve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input.value || 'song' })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
    if (!d.chunks || !d.chunks.length) {
      out.textContent = 'No passage cleared the 0.15 similarity floor. The chat model would be told there are no notes on this question.';
      return;
    }
    out.innerHTML = '<b>' + d.chunks.length + ' passages returned · ' + d.indexed + ' chunks indexed</b>\n' +
      d.chunks.map((c, i) =>
        (i + 1) + '. ' + c.source + ' · cosine ' + Number(c.score).toFixed(3) + '\n   ' +
        c.text.replace(/\*\*/g, '').replace(/\s+/g, ' ').slice(0, 240) + (c.text.length > 240 ? '…' : '')
      ).join('\n\n');
  } catch (e) {
    out.textContent = 'Retrieval failed: ' + e.message;
  }
};
window.ragHealth = async function () {
  const out = document.getElementById('rag-health');
  if (!out) return;
  out.textContent = 'Checking…';
  try {
    const r = await fetch('/api/chat/health');
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
    out.textContent = 'notes index: ' + (d.rag_ready ? 'ready' : 'building') +
      '\nchunks indexed: ' + d.rag_chunks_indexed +
      '\ncree model: ' + (d.cree_model_configured
        ? d.cree_model + ' (configured)'
        : 'not configured, Cree questions stay on the general model') +
      '\nspoken replies: ' + (d.tts_ready ? 'Piper voice configured' : 'text replies, with spoken replies available when a Piper voice is set');
  } catch (e) {
    out.textContent = 'Health check failed: ' + e.message;
  }
};

// ---------------------------------------------------------------- Cree model
window.creeRoute = async function () {
  const input = document.getElementById('cree-route-input');
  const out = document.getElementById('cree-route-out');
  if (!input || !out) return;
  const sel = document.getElementById('cree-route-language');
  const message = input.value.trim();
  if (!message) { out.textContent = 'Type a question first.'; return; }
  out.textContent = 'Asking the router…';
  try {
    const r = await fetch('/api/chat/route', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message, language: sel ? sel.value : 'auto' })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
    const lines = [
      'route       ' + d.route + (d.route === 'cree' ? '  (the trained Cree model)' : '  (the general model)'),
      'text        ' + (d.looks_like_cree ? 'Cree' : 'not Cree') + (d.script ? '  ·  script ' + d.script : ''),
      'asks for    ' + (d.wants_cree ? 'Cree output' : 'no Cree output'),
      'setting     ' + (d.configured ? 'cree_model is ' + d.model : 'cree_model is empty, so this checkout answers on the general model')
    ];
    if (d.cree_prompt) {
      lines.push('', 'prompt this request would carry on the Cree route', d.cree_prompt);
    } else {
      lines.push('', 'The general model gets the retrieval system prompt, not a Cree prompt.');
    }
    out.textContent = lines.join('\n');
  } catch (e) {
    out.textContent = 'The chat server did not answer: ' + e.message + '\nStart server.py, then try again.';
  }
};

window.creeLicence = function (kind, btn) {
  let shown = 0;
  document.querySelectorAll('#cree-licence-table tbody tr').forEach(tr => {
    const show = kind === 'all' || tr.getAttribute('data-licence') === kind;
    tr.hidden = !show;
    if (show) shown++;
  });
  document.querySelectorAll('[data-licence-filter]').forEach(b => b.classList.toggle('is-on', b === btn));
  const count = document.getElementById('cree-licence-count');
  if (count) count.textContent = 'Showing ' + shown + ' of 15 collections.';
};

window.creeStage = function (name) {
  document.querySelectorAll('[data-cree-stage]').forEach(el => {
    el.hidden = el.getAttribute('data-cree-stage') !== name;
  });
  document.querySelectorAll('[data-cree-stage-tab]').forEach(b => {
    b.classList.toggle('is-on', b.getAttribute('data-cree-stage-tab') === name);
  });
};

window.creeTokenF1 = function () {
  const goldEl = document.getElementById('cree-f1-gold');
  const predEl = document.getElementById('cree-f1-pred');
  const out = document.getElementById('cree-f1-out');
  if (!goldEl || !predEl || !out) return;
  const tokens = s => (s.toLowerCase().match(/[\p{L}\p{M}]+/gu) || []);
  const gold = tokens(goldEl.value);
  const pred = tokens(predEl.value);
  const counts = new Map();
  gold.forEach(w => counts.set(w, (counts.get(w) || 0) + 1));
  let overlap = 0;
  pred.forEach(w => {
    const n = counts.get(w) || 0;
    if (n > 0) { overlap++; counts.set(w, n - 1); }
  });
  const precision = pred.length ? overlap / pred.length : 0;
  const recall = gold.length ? overlap / gold.length : 0;
  const f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0;
  out.textContent = 'gold tokens      ' + gold.length +
    '\nmodel tokens    ' + pred.length +
    '\noverlap         ' + overlap +
    '\nprecision       ' + precision.toFixed(3) +
    '\nrecall          ' + recall.toFixed(3) +
    '\nF1              ' + f1.toFixed(3);
};
window.openArcherChat = function () {
  if (typeof toggleArcherChat === 'function') {
    const panel = document.getElementById('archer-chat-panel');
    if (!panel || !panel.classList.contains('open')) toggleArcherChat();
  }
};

// ---------------------------------------------------------------- SingSong
window.ssLen = function (sec, btn) {
  document.querySelectorAll('[data-sec]').forEach(b => b.classList.remove('is-on'));
  if (btn) btn.classList.add('is-on');
  const sem = 25 * sec, coarse = 200 * sec, fine = 400 * sec;
  const set = (id, n) => { const el = document.getElementById(id); if (el) el.textContent = n.toLocaleString('en-US'); };
  set('ss-sem', sem);
  set('ss-stage12', sem + coarse);
  set('ss-fine', fine);
  set('ss-samples', 16000 * sec);
};

// ---------------------------------------------------------------- status
async function pageStatus(url, elId, describe) {
  const el = document.getElementById(elId);
  if (!el) return;
  try {
    const r = await fetch(url);
    const d = await r.json();
    el.innerHTML = describe(d, r.ok);
  } catch (e) {
    el.textContent = 'The server is not reachable from this page: ' + e.message;
  }
}
window.addEventListener('DOMContentLoaded', () => {
  pageStatus('/singsong/status', 'ss-status', d => {
    if (d.ready) return '<b>Neural checkpoint present.</b> The trained stages are loaded and the panel will use them.';
    return '<b>Built-in band ready.</b> The panel writes a band under the take now; the neural stages load automatically when their trained artefacts are present.';
  });
  pageStatus('/api/mysong/status', 'ms-page-status', d => {
    if (d.ready) return '<b>MySong ready.</b> ' + d.songs + ' songs in the structural model · tagger ' +
      (d.tagger ? 'loaded' : 'not loaded') + ' · key classifier ' + (d.key_classifier ? 'loaded' : 'not loaded');
    return '<b>MySong model not found.</b> ' + (d.error || 'Run python -m mysong.cli train to build it.');
  });
});
