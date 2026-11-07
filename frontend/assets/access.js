/* ============================================================
   Archer Pechawis, Vocal Collaboration System
   Private access: the door, and how long it stays open.

   This site is served as static files, which decides the shape of
   the whole thing: there is no server to ask, so the check runs in
   the browser and the password is never in the page. What is in the
   page is its SHA-256, and what the field is compared against is
   that digest. Reading the file tells you nothing you can type at
   the door, which is the most a static host can promise. (A server
   that could refuse the bytes would be the real lock; on a static
   site the lock is on the page, not on the files.)

   The session is a stamp in localStorage, not a cookie: an expiry
   time, and the time the streak began.

     exp   slides forward on use, so a visitor reading for an hour
           is never thrown out mid-sentence.
     born  does not slide, and caps the whole streak at twelve
           hours, so a machine left open overnight is locked in the
           morning even if something kept touching it.

   The line is thirty quiet minutes: no pointer, key, wheel or
   scroll for that long and the door comes back. Anything at all
   pushes it out again. The check itself is a fifteen-second tick,
   and the stamp is only rewritten once every ten seconds, so a
   page being read costs one small write a minute at most.

   The script is loaded in <head> and blocks, deliberately: it
   decides whether the sheet is gated before the body is parsed, so
   a cold load never paints a readable page for a frame and then
   covers it.
   ============================================================ */
(function () {
  'use strict';

  var KEY = 'archer.access.v1';

  /* SHA-256 of the password. Lowercase hex, one line, no spaces. */
  var DIGEST = '6f86a8314f563e6a319072c40277bef0f95d10ed01a722e2597c7e5b9ecc208a';

  /* Only reached on an origin where SubtleCrypto is missing (a plain-http
     host other than localhost). The deployed site is https, so this is the
     safety net, not the lock. */
  var LEGACY = 'QWJ1bmRhbnRfSW50ZWxsaWdlbmNlczIwMjY=';

  var IDLE_MS = 30 * 60 * 1000;    /* thirty quiet minutes */
  var CEILING_MS = 12 * 60 * 60 * 1000;
  var TOUCH_MS = 10 * 1000;
  var TICK_MS = 15 * 1000;

  var doc = document.documentElement;
  var gate = null, card = null, form = null, input = null, msg = null,
      reveal = null, submit = null;
  var sealed = [];
  var session = null;
  var busy = false;
  var ticker = null;
  var lastTouch = 0;
  var wired = false;
  var watching = false;

  function clock() { return Date.now(); }

  /* ---------- the stamp ---------- */

  function readSession() {
    try {
      var raw = window.localStorage.getItem(KEY);
      if (!raw) return null;
      var s = JSON.parse(raw);
      if (!s || typeof s.exp !== 'number' || typeof s.born !== 'number') return null;
      return s;
    } catch (e) { return null; }
  }

  function saveSession(s) {
    try { window.localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
  }

  function forgetSession() {
    try { window.localStorage.removeItem(KEY); } catch (e) {}
  }

  function isLive(s) {
    if (!s) return false;
    var t = clock();
    return s.exp > t && s.born + CEILING_MS > t;
  }

  function slide() {
    if (!session) return;
    var t = clock();
    if (t - lastTouch < TOUCH_MS) return;
    lastTouch = t;
    session.exp = Math.min(t + IDLE_MS, session.born + CEILING_MS);
    saveSession(session);
  }

  /* ---------- before the body exists ---------- */

  session = readSession();
  if (isLive(session)) {
    /* arriving on a page is use: the clock starts again on the way in */
    session.exp = Math.min(clock() + IDLE_MS, session.born + CEILING_MS);
    saveSession(session);
  } else {
    session = null;
    doc.classList.add('archer-gated');
  }

  /* ---------- the door ---------- */

  var MESSAGES = {
    expired: 'This visit closed after a while without use. Enter the password to carry on.',
    'signed-out': 'You were signed out. Enter the password to carry on.'
  };

  /* Everything that is not the door goes inert while the door is up: the sheet,
     the bar, the chat button and the chat panel. Without this the page behind a
     blur is still tabbable, and a keyboard would walk into a site nobody has
     been let into. The previous aria-hidden value is remembered per element so
     that closing the door restores what the app itself had set, rather than
     claiming the chat panel is visible when it is shut. */
  function seal() {
    var kids = document.body.children;
    for (var i = 0; i < kids.length; i++) {
      var el = kids[i];
      if (el === gate || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') continue;
      if (el.inert) continue;
      el.inert = true;
      sealed.push([el, el.getAttribute('aria-hidden')]);
      el.setAttribute('aria-hidden', 'true');
    }
  }

  function unseal() {
    for (var i = 0; i < sealed.length; i++) {
      var el = sealed[i][0], was = sealed[i][1];
      el.inert = false;
      if (was === null) el.removeAttribute('aria-hidden');
      else el.setAttribute('aria-hidden', was);
    }
    sealed = [];
  }

  function show(reason) {
    if (!gate) return;
    gate.hidden = false;
    document.body.classList.add('gate-open');
    doc.classList.add('archer-gated');
    seal();

    /* the entrance runs from its starting point only if the browser has laid
       the card out at that starting point first */
    void gate.offsetWidth;
    gate.classList.add('is-open');
    /* One frame later, and not before: the sheet of paper that covers the
       window from the first paint (see access.css) starts to fade, the veil
       arrives under the card, and the page behind is allowed to blur. Doing it
       in the same frame as the card would cross-fade two paper layers and show
       the page through both; a frame of separation lets the card lead. */
    window.requestAnimationFrame(function () {
      doc.classList.add('gate-ready');
    });

    if (msg) msg.textContent = MESSAGES[reason] || '';
    if (input) input.value = '';
    if (input && reveal) {
      input.type = 'password';
      reveal.textContent = 'Show';
      reveal.setAttribute('aria-pressed', 'false');
    }

    window.setTimeout(function () {
      try { input.focus({ preventScroll: true }); } catch (e) { input.focus(); }
    }, 260);
  }

  function open() {
    doc.classList.remove('gate-ready');
    doc.classList.remove('archer-gated');
    document.body.classList.remove('gate-open');
    unseal();
    if (gate) {
      gate.classList.remove('is-open');
      window.setTimeout(function () {
        if (gate && !gate.classList.contains('is-open')) gate.hidden = true;
      }, 520);
    }
    if (input) input.value = '';
    if (msg) msg.textContent = '';
    watch();
  }

  function signOut(reason) {
    forgetSession();
    session = null;
    stopWatch();
    show(reason);
  }

  function inSite() {
    return !gate || gate.hidden;
  }

  /* ---------- use ---------- */

  var ACTS = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart', 'scroll'];

  function watch() {
    if (watching) return;
    watching = true;
    lastTouch = clock();
    if (wired) {
      ACTS.forEach(function (name) {
        document.addEventListener(name, slide, { passive: true, capture: true });
      });
    }
    ticker = window.setInterval(function () {
      if (!isLive(session)) signOut('expired');
    }, TICK_MS);
  }

  function stopWatch() {
    if (ticker) { window.clearInterval(ticker); ticker = null; }
    if (!watching) return;
    watching = false;
    ACTS.forEach(function (name) {
      document.removeEventListener(name, slide, { passive: true, capture: true });
    });
  }

  /* ---------- the check ---------- */

  function hex(buffer) {
    var bytes = new Uint8Array(buffer);
    var out = '';
    for (var i = 0; i < bytes.length; i++) {
      out += (bytes[i] < 16 ? '0' : '') + bytes[i].toString(16);
    }
    return out;
  }

  function legacyMatch(pw) {
    try { return window.atob(LEGACY) === pw; } catch (e) { return false; }
  }

  function matches(pw) {
    if (window.crypto && window.crypto.subtle && window.TextEncoder) {
      try {
        return window.crypto.subtle
          .digest('SHA-256', new TextEncoder().encode(pw))
          .then(function (buf) { return hex(buf) === DIGEST; })
          .catch(function () { return legacyMatch(pw); });
      } catch (e) { /* fall through */ }
    }
    return Promise.resolve(legacyMatch(pw));
  }

  function fail(text) {
    if (msg) msg.textContent = text;
    if (card) {
      card.classList.remove('is-wrong');
      void card.offsetWidth;
      card.classList.add('is-wrong');
      window.setTimeout(function () { card.classList.remove('is-wrong'); }, 640);
    }
    /* the caret goes back to the field; the characters are left where they are
       so a near miss can be corrected rather than typed again from nothing */
    if (input) input.focus();
  }

  function onSubmit(e) {
    e.preventDefault();
    if (busy) return;
    var pw = input ? input.value : '';
    if (!pw) { fail('Enter the password to come in.'); return; }

    busy = true;
    if (submit) submit.disabled = true;

    matches(pw).then(function (ok) {
      busy = false;
      if (submit) submit.disabled = false;
      if (!ok) { fail('That is not the password.'); return; }
      var t = clock();
      session = { born: t, exp: t + IDLE_MS };
      saveSession(session);
      open();
    });
  }

  function toggleReveal() {
    if (!input) return;
    var shown = input.type === 'text';
    input.type = shown ? 'password' : 'text';
    if (reveal) {
      reveal.textContent = shown ? 'Show' : 'Hide';
      reveal.setAttribute('aria-pressed', shown ? 'false' : 'true');
    }
    input.focus();
  }

  /* ---------- wiring ---------- */

  function bind() {
    gate = document.getElementById('archer-gate');
    if (!gate) {
      /* no markup on this page: never leave the sheet gated behind a door that
         does not exist */
      doc.classList.remove('archer-gated');
      stopWatch();
      return;
    }
    card = gate.querySelector('.gate-card');
    form = document.getElementById('gate-form');
    input = document.getElementById('gate-input');
    msg = document.getElementById('gate-msg');
    reveal = document.getElementById('gate-reveal');
    submit = document.getElementById('gate-submit');

    if (form) form.addEventListener('submit', onSubmit);
    /* Enter is the obvious way through a one-field form, and asking the browser
       to do it implicitly is not dependable (it depends on how many fields a
       form has, and on whether anything upstream has claimed the key). The
       keyboard is answered here instead, from the gate, so Enter inside the
       field and Enter on the button are the same submit. */
    gate.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' || !form) return;
      if (e.target && e.target.tagName === 'BUTTON') return;
      e.preventDefault();
      if (typeof form.requestSubmit === 'function') form.requestSubmit();
      else onSubmit(e);
    });
    if (reveal) reveal.addEventListener('click', toggleReveal);
    if (input) {
      input.addEventListener('input', function () {
        if (msg) msg.textContent = '';
      });
    }

    wired = true;
    if (session && isLive(session)) open();
    else { session = null; show(''); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind, { once: true });
  } else {
    bind();
  }

  /* A second tab is the same visit. Locking or entering in one is not a
     private event on this machine, so it happens in both. */
  window.addEventListener('storage', function (e) {
    if (e.key !== KEY || !gate) return;
    var s = readSession();
    if (isLive(s)) {
      if (inSite()) return;
      session = s;
      open();
    } else if (!inSite()) {
      signOut('signed-out');
    }
  });

  /* Back from the back/forward cache is a page nobody re-ran: check it. */
  window.addEventListener('pageshow', function (e) {
    if (e.persisted && gate && inSite() && !isLive(session)) signOut('expired');
  });
})();
