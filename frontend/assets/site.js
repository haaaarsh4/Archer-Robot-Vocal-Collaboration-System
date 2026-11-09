/* ============================================================
   Archer Pechawis, Vocal Collaboration System
   Site shell: home filmstrip, the open transition, the section
   menu, and the scroll reveals.

   This file only drives the chrome. Every tool (Live Demo, Harmony,
   Synthesis, Sentiment, Chat, Architecture) is untouched and still
   runs from the original inline script in index.html.
   ============================================================ */
(function () {
  'use strict';

  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* One percent of the window height, which is the unit the whole strip is
     built from: a plate is twelve of these wide. The reference sets the same
     custom property from script; `1vh` is the fallback if this never runs. */
  function setVh() {
    document.documentElement.style.setProperty('--vh', (window.innerHeight / 100) + 'px');
  }
  setVh();
  window.addEventListener('resize', setVh);
  window.addEventListener('orientationchange', setVh);
  /* the resize event is not the only way the window can change size (a panel
     dragged open, a phone toolbar retracting), and the whole strip is measured
     from this number, so watch the document box as well */
  if (window.ResizeObserver) {
    try { new ResizeObserver(setVh).observe(document.documentElement); } catch (e) {}
  }

  /* The plates live in the markup and carry their own words. `data-title`,
     `data-sub` and `data-kicker` are the caption a plate wears when it takes
     focus; `data-page` is where it opens, and `data-work` points at a row of
     the work list on the About page for the plates that are works rather than
     sections. */
  var REST_TITLE = 'Featured Work';
  var REST_SUB = 'Explore some of Archer\u2019s work';

  /* i is -1 when no plate holds the strip: that is the resting state, and it
     is where the page opens. */
  var state = { i: -1 };
  var home = document.getElementById('page-home');
  var film = document.getElementById('film');
  if (!film) return;

  var strip = document.getElementById('film-strip');
  var titleEl = document.getElementById('film-title');
  var subEl = document.getElementById('film-sub');

  var cards = Array.prototype.slice.call(strip.querySelectorAll('.film-card'));
  var pieces = cards.map(function (c) {
    return {
      page: c.dataset.page,
      work: c.dataset.work || '',
      title: c.dataset.title || '',
      sub: c.dataset.sub || ''
    };
  });

  /* A plate holds the strip while the pointer is ON IT, and only while it is on
     it. The strip is the whole window, a row centred inside a full-bleed box
     so waiting for a `mouseleave` on the strip alone meant a plate stayed
     open and coloured wherever the pointer went, even parked in the paper on
     the far side of the row. Each plate now hands the strip back the moment the
     pointer leaves that plate. Sliding along the row does not flicker: the
     leave and the enter are dispatched in the same frame, so the only thing
     ever painted is one plate closing and the next opening. */
  cards.forEach(function (b, i) {
    b.addEventListener('mouseenter', function () { focusCard(i); });
    b.addEventListener('focus', function () { focusCard(i); });
    b.addEventListener('click', function () { clickCard(i); });
    b.addEventListener('mouseleave', function (e) {
      var to = e.relatedTarget;
      if (to && to.closest && to.closest('.film-card')) return;
      focusCard(-1);
    });
  });

  /* ================================================= wordmark build */
  function letters(text) {
    /* per-letter spans so the wordmark can be written on */
    var out = document.createElement('span');
    var n = 0;
    text.split(' ').forEach(function (word, wi) {
      if (wi) {
        var sp = document.createElement('span');
        sp.className = 'sp';
        sp.style.width = '0.24em';
        sp.style.display = 'inline-block';
        sp.innerHTML = '&nbsp;';
        out.appendChild(sp);
      }
      var w = document.createElement('span');
      w.style.display = 'inline-block';
      w.style.whiteSpace = 'nowrap';
      Array.prototype.forEach.call(word, function (ch) {
        var l = document.createElement('span');
        l.className = 'lt';
        l.style.setProperty('--i', n++);
        l.textContent = ch;
        w.appendChild(l);
      });
      out.appendChild(w);
    });
    return out;
  }

  /* the wordmark is written on rather than swapped; on the very first paint
     there is nothing to animate from, so it is simply set */
  function write(text, instant) {
    titleEl.classList.remove('writing');
    titleEl.classList.remove('settled');
    void titleEl.offsetWidth;
    titleEl.innerHTML = '';
    titleEl.appendChild(letters(text));
    titleEl.classList.add(instant ? 'settled' : 'writing');
  }

  /* Taking focus widens one plate to 24vh in place and slides the plates
     either side of it out by half the growth (6vh), the reference's own
     tween, and what keeps the strip centred while a plate opens. The caption
     under the strip, the wordmark line and the small line under it
     follows: the drawn wordmark at rest, our own section name once a plate is
     open, which is the swap the reference makes too. */
  function focusCard(i, instant) {
    if (i < 0) {
      state.i = -1;
      film.classList.remove('on-plate');
      cards.forEach(function (c) { c.classList.remove('is-focus', 'is-before', 'is-after'); });
      write(REST_TITLE, instant);
      subEl.textContent = REST_SUB;
      return;
    }
    i = (i + pieces.length) % pieces.length;
    if (i === state.i && cards[i].classList.contains('is-focus')) return;
    state.i = i;
    var s = pieces[i];
    cards.forEach(function (c, j) {
      c.classList.toggle('is-focus', j === i);
      c.classList.toggle('is-before', j < i);
      c.classList.toggle('is-after', j > i);
    });
    film.classList.add('on-plate');
    write(s.title, instant);
    subEl.textContent = s.sub;
    /* on a phone the strip scrolls sideways, so follow the plate in focus */
    if (window.matchMedia('(max-width: 760px)').matches) {
      try { cards[i].scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); } catch (e) {}
    }
  }

  function clickCard(i) {
    if (i === state.i) { openSection(i); return; }
    focusCard(i);
  }

  /* No autoplay, and nothing moves on its own: the strip takes focus under
     the pointer or the keyboard, and it settles back to rest when the
     pointer leaves it, the row, or the window itself. */
  strip.addEventListener('mouseleave', function () { focusCard(-1); });
  document.addEventListener('mouseleave', function () { focusCard(-1); });
  titleEl.addEventListener('click', function () {
    if (state.i >= 0) openSectionIndex(state.i);
  });
  titleEl.style.cursor = 'pointer';

  /* =================================================== open transition */
  /* The reference's own open move, `itemSelected` in their bundle, and
     their numbers, kept. It runs on one timeline:

     0.00s          the plate you opened is lifted out of the row: a copy of
                    it, pinned exactly where it already stands, so its first
                    frame is the frame you clicked and nothing jumps.
     0.00 - 0.65s   every plate slides down its own stagger step and the row
                    lands on one line, their `y: 50 - elementYOffsets[i]`,
                    which is why the row reads as aligning before it opens.
     0.65s          a sheet of black turns on. It is already the exact size
                    and place of the plate, scaled to that slot, so
                    switching it on costs nothing, and it grows out from
                    under the plate rather than arriving over it.
     0.65 - 1.65s   the lifted plate is blown up until it covers the window
                    and dissolved into the black over the first half of that;
                    the sheet grows with it, one number for both sides; the
                    plates either side of it are shoved out of the row.
     0.00 - 1.50s   the chrome clears the stage: the bar up out of the top
                    and the caption down out of the bottom, on one curve.
     1.65s          the window is black. The page swaps underneath and the
                    black is lifted off it.

     The bottom bar this move used to send down is gone: with it went the
     last thing on the page that was not the reference's own, so the chrome
     that clears the stage is the bar at the top and the caption at the
     bottom, and nothing else moves.

     Their curve is the CustomEase `M0,0 C0.6,0 0.1,1 1,1`, slow off the
     mark, quick to arrive, which is exactly cubic-bezier(0.6, 0, 0.1, 1),
     so the whole move is carried by CSS transitions and a pair of timers.
     The plate is scaled to cover with one number for both sides
     (`scaleToFill` takes whichever of width and height needs more), so the
     move never stretches the art. */
  var OPEN_EASE = 'cubic-bezier(0.6, 0, 0.1, 1)';                 /* their `et` */
  var OPEN_FADE_EASE = 'cubic-bezier(0.55, 0.085, 0.68, 0.53)';  /* their easeIn */
  var MS_ALIGN = 650;   /* their align tween: 0.65s, from 0 */
  var MS_OPEN = 1000;   /* their expand, shove and black tweens: 1s, from 0.65s */
  var MS_FADE = 500;    /* their `autoAlpha: 0` on the lifted plate */
  var MS_CHROME = 1500; /* their `hideSurroundingUI` */
  var MS_BLACK_OFF = 420;
  var busy = false;

  /* where an element is right now, from its own computed transform: a plate
     caught mid-slide has to be picked up from where it actually is */
  function currentShift(el) {
    var m = window.getComputedStyle(el).transform;
    var n = m && /matrix\(([^)]+)\)/.exec(m);
    if (!n) return { x: 0, y: 0 };
    var v = n[1].split(',');
    return { x: parseFloat(v[4]) || 0, y: parseFloat(v[5]) || 0 };
  }
  /* the three plates that are works point at a row of the work list on the
     About page: open that page, put the row in the middle of the window and
     mark it, so the plate lands somewhere real */
  function revealWork(slug) {
    var row = document.querySelector('#about-rows .row[data-work="' + slug + '"]');
    if (!row) return;
    try { row.scrollIntoView({ block: 'center', behavior: REDUCED ? 'auto' : 'smooth' }); } catch (e) {}
    row.classList.add('is-found');
    window.setTimeout(function () { row.classList.remove('is-found'); }, 2800);
  }
  var chromeUp = document.querySelector('.topbar');
  var chromeCaption = document.getElementById('film-caption');

  function openSectionIndex(i) {
    var card = cards[i];
    var page = card.dataset.page;
    var work = card.dataset.work;
    function land() {
      window.showPage(page);
      if (work) window.setTimeout(function () { revealWork(work); }, 120);
    }
    if (busy) return;
    if (REDUCED) { land(); return; }
    /* the strip springs in on a fresh load, and a hand can click a plate in
       the middle of that, where it is still the sliver it comes in as. The
       arrival is taken off first, before anything is measured, so the move
       starts from a plate rather than from the line that sliver is: the
       class coming off is what stops the spring, and `arrived()` is what
       disarms its timer. Taking the class off is safe on every other path
       it is not there to begin with, and nothing puts it back, and there is
       no check before it, because a click that arrives without a
       pointer press (a key, a script) never gets the listener's cleanup.
       (`arriving` and `arrived` are set up further down the file; `var` and a
       function declaration are both hoisted, so both are there in time.) */
    if (arriving) arrived();
    document.body.classList.remove('is-entering');
    busy = true;

    /* the plate as it is actually on screen, which is not the card's layout
       box: the open card's window is twice its width and pulled half a plate
       to the left of it, so the box it is drawn in is the one to lift */
    var inner = card.querySelector('.film-inner') || card;
    var r = inner.getBoundingClientRect();
    var base = cards[0].offsetTop;
    var offsets = cards.map(function (c) { return c.offsetTop - base; });
    var settle = 50 - offsets[i];  /* where the row's alignment lands */
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    /* scaleToFill: the plate's own box, both sides by one number, whichever
       of width and height needs more to cover, so nothing is stretched */
    var k = Math.max(vw / r.width, vh / r.height);
    var coverW = Math.round(r.width * k);
    var coverH = Math.round(r.height * k);
    var cx = Math.round((vw - coverW) / 2);
    var cy = Math.round((vh - coverH) / 2);
    var barW = 0.12 * vh;  /* a plate's resting width, their own number */
    var gap = 2 * (parseFloat(window.getComputedStyle(card).marginLeft) || 0);
    var shoveLeft = -(r.left + barW);
    var shoveRight = vw - (r.left + r.width + gap - barW);
    var curtain = r.top - cy;  /* how far the chrome is sent away */

    document.body.classList.add('is-locked');
    film.classList.add('is-leaving');
    card.style.opacity = '0';  /* the slot the plate was lifted out of */

    /* the sheet of black: pinned to the top left at the covered size and
       scaled down to the plate's slot, so its first frame is the plate's own
       rectangle and it grows to cover from there */
    var sheet = document.createElement('div');
    sheet.className = 'film-sheet';
    sheet.style.width = coverW + 'px';
    sheet.style.height = coverH + 'px';
    sheet.style.transform = 'translate(' + r.left + 'px,' + (r.top + settle) +
      'px) scale(' + (r.width / coverW) + ',' + (r.height / coverH) + ')';
    document.body.appendChild(sheet);

    /* the plate itself, lifted out of the row at the size and place it
       already occupies: the copy is the same markup, so the pixels are
       identical on the frame it leaves */
    var lift = document.createElement('div');
    lift.className = 'film-lift';
    lift.style.width = r.width + 'px';
    lift.style.height = r.height + 'px';
    lift.style.transform = 'translate(' + r.left + 'px,' + r.top + 'px)';
    var copy = inner.cloneNode(true);
    copy.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;transition:none;';
    var shot = copy.querySelector('.film-shot');
    if (shot) shot.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;transform:translateX(0);transition:none;';
    var art = copy.querySelector('img');
    /* the copy must wear the open plate's own clothes, not the resting
       wash: the stylesheet greys and fades a still plate, and this one is
       standing in for the plate under the pointer */
    if (art) { art.style.filter = 'none'; art.style.opacity = '1'; }
    lift.appendChild(copy);
    document.body.appendChild(lift);

    /* one: the row settles onto a single line, and the lifted plate settles
       with it, their `y: 50 - elementYOffsets[i]`, each plate on its own
       beat because each has its own step to travel */
    cards.forEach(function (c, j) {
      if (j === i) return;
      var el = c.querySelector('.film-inner');
      var now = currentShift(el);
      el.style.transition = 'transform ' + MS_ALIGN + 'ms ' + OPEN_EASE;
      el.style.transform = 'translate(' + now.x + 'px,' + (50 - offsets[j]) + 'px)';
    });
    lift.style.transition = 'transform ' + MS_ALIGN + 'ms ' + OPEN_EASE;
    lift.style.transform = 'translate(' + r.left + 'px,' + (r.top + settle) + 'px)';

    /* and the chrome clears the stage while all of that happens: the bar up
       out of the top, the caption down out of the bottom */
    if (chromeUp) {
      chromeUp.style.transition = 'transform ' + MS_CHROME + 'ms ' + OPEN_EASE;
      chromeUp.style.transform = 'translateY(' + -curtain + 'px)';
    }
    if (chromeCaption) {
      chromeCaption.style.transition = 'transform ' + MS_CHROME + 'ms ' + OPEN_EASE;
      chromeCaption.style.transform = 'translateY(' + curtain + 'px)';
    }

    /* two: the plate is blown up to cover the window and dissolved into the
       black, the black grows with it, and the rest of the row is shoved out
       of the way, all four on the same second, from the same instant */
    window.setTimeout(function () {
      sheet.style.opacity = '1';
      sheet.style.transition = 'transform ' + MS_OPEN + 'ms ' + OPEN_EASE;
      sheet.style.transform = 'translate(' + cx + 'px,' + cy + 'px) scale(1, 1)';
      lift.style.transition = 'transform ' + MS_OPEN + 'ms ' + OPEN_EASE +
        ', opacity ' + MS_FADE + 'ms ' + OPEN_FADE_EASE;
      lift.style.transform = 'translate(' + cx + 'px,' + cy + 'px) scale(' + (coverW / r.width) + ')';
      lift.style.opacity = '0';
      cards.forEach(function (c, j) {
        if (j === i) return;
        var el = c.querySelector('.film-inner');
        el.style.transition = 'transform ' + MS_OPEN + 'ms ' + OPEN_EASE;
        el.style.transform = 'translate(' + (j < i ? shoveLeft : shoveRight) + 'px,' +
          (50 - offsets[j]) + 'px)';
      });
    }, MS_ALIGN);

    /* three: the window is black, so the page can be swapped underneath it
       and the black lifted off */
    window.setTimeout(function () {
      land();
      document.body.classList.remove('is-locked');
      sheet.style.transition = 'opacity ' + MS_BLACK_OFF + 'ms ease';
      sheet.style.opacity = '0';
      window.setTimeout(function () {
        if (lift.parentNode) lift.parentNode.removeChild(lift);
        if (sheet.parentNode) sheet.parentNode.removeChild(sheet);
        card.style.opacity = '';
        cards.forEach(function (c) {
          var el = c.querySelector('.film-inner');
          el.style.transition = '';
          el.style.transform = '';
        });
        if (chromeUp) { chromeUp.style.transition = ''; chromeUp.style.transform = ''; }
        if (chromeCaption) { chromeCaption.style.transition = ''; chromeCaption.style.transform = ''; }
        film.classList.remove('is-leaving');
        busy = false;
      }, MS_BLACK_OFF + 40);
    }, MS_ALIGN + MS_OPEN);
  }
  function openSection(i) { openSectionIndex(i); }

  /* =================================================== keyboard */
  document.addEventListener('keydown', function (e) {
    if (!home.classList.contains('active')) return;
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === 'ArrowRight') { focusCard(state.i + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { focusCard(state.i < 0 ? -1 : state.i - 1); e.preventDefault(); }
    else if (e.key === 'Enter' && state.i >= 0) { openSectionIndex(state.i); }
  });

  /* =================================================== section menu */
  var menu = document.getElementById('menu');
  var menuBtn = document.getElementById('menu-btn');
  var menuBack = document.getElementById('menu-back');
  function setMenu(on) {
    if (!menu) return;
    menu.classList.toggle('open', on);
    if (menuBtn) menuBtn.setAttribute('aria-expanded', String(on));
    if (on) {
      document.body.classList.add('is-locked');
      if (menuBack) menuBack.focus();
    } else {
      document.body.classList.remove('is-locked');
      // The panel took focus when it opened; give it back to the control that
      // opened it rather than leaving it on an element nobody can see.
      if (menuBack && document.activeElement === menuBack && menuBtn) menuBtn.focus();
    }
  }
  /* One button in the corner, two jobs, exactly as the reference's own corner
     control has. On the home page it is the menu: three bars that open the
     panel. On every page behind the home page it is the way back to the row of
     plates, the close mark closes, and there is only ever one row to close
     back to. It used to open the panel from behind the home page as well, which
     meant the one control on the page did the one thing nobody wanted there. */
  if (menuBtn) menuBtn.addEventListener('click', function () {
    if (document.body.classList.contains('on-home')) {
      setMenu(!menu.classList.contains('open'));
    } else {
      closeToHome();
    }
  });
  if (menu) {
    /* The panel opens from the front door, so its cross closes the panel. It
       used to call goBack(), which walked the site's page trail and could
       land on whichever section had been visited before: closing a menu is
       not a navigation, and it should never take you to Sentiment. */
    if (menuBack) menuBack.addEventListener('click', function () { setMenu(false); });
    menu.querySelector('.menu-veil').addEventListener('click', function () { setMenu(false); });
    menu.querySelectorAll('button[data-page]').forEach(function (b) {
      b.addEventListener('click', function () {
        var page = b.dataset.page;
        setMenu(false);
        window.showPage(page);
      });
    });
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') setMenu(false);
  });

  /* =================================================== the paper close
     Coming back to the front door is the reference's own move, taken off their
     own screencast frame by frame, and it is the opposite of the move that
     opens a section: nothing slides, nothing is scaled, nothing is blown up.
     The screen becomes paper, and then becomes the front door again.

     Measured off that recording, on the band of the window that holds the row
     (24fps, x264-free VP8, mean luminance of the band against their paper at
     236):

       0.00 - 0.30s   the section stands exactly still and then washes out.
                      Their whole view, plate, captions, thumbs, the title of
                      the section, goes 52 -> 90 -> 171 -> 222 between 1.75s
                      and 2.05s. Slow off the mark, all of it at the end: their
                      own `M0,0 C0.6,0 0.1,1 1,1` curve, which is
                      cubic-bezier(0.6, 0, 0.1, 1), the same one the section's
                      own open is built on. Finished in a third of a second.
                      Nothing on the page moves by a pixel while it happens.
       0.30 - 0.55s   paper, and empty: their row is not there yet.
       0.55 - 1.00s   the row resolves up out of the paper, in place. The
                      columns of their row read 30% of their final contrast at
                      2.28s and 100% at 2.66s, the same curve again, and then
                      the frame is absolutely still. The left half is a hair
                      ahead of the right, which is their strip's own order and
                      not a stagger a hand can see.

     One second, one curve, one sheet of paper. So this is a veil of the page's
     own paper over the whole window: up over a third of a second on that
     curve, held for a beat while the page is swapped underneath it and the
     window is put back at the top, then lifted on the same curve. The row is
     not faded in behind it and nothing is staggered, because their row is not
     staggered either: at the moment their paper lifts the front door is simply
     there, and it is the paper leaving that makes it arrive.

     Anything that returns to the front door comes through here, the corner
     control, the panel's own links, the panel's close control, the wordmark,
     the browser's back, because the one thing they all pass through is
     `showPage`, and that is where this is called from. One way home. */
  var SETTLE_EASE = 'cubic-bezier(0.22, 1, 0.36, 1)';  /* `--ease`: quick, then a quiet */
  var MS_CLOSE_OUT = 300;   /* the leaving view washes out to paper */
  var MS_CLOSE_BEAT = 220;  /* the paper holds, empty, and then the row starts */
  var MS_CLOSE_IN = 460;    /* the paper is out of the way */
  /* the row outlasts the paper: the caption's own beat plus its own 520ms */
  var MS_CLOSE_ROW = 220 + 180 + 520;
  var paperVeilEl = null;
  function paperVeil() {
    if (!paperVeilEl) {
      /* the veil is built once and kept for the life of the document, so there
         is only ever one of them: anything else wearing the class is not ours
         a restored page, another script, a hand in the console, and two
         sheets of paper would compose, and the second would never lift */
      document.querySelectorAll('.paper-veil').forEach(function (stray) {
        if (stray.parentNode) stray.parentNode.removeChild(stray);
      });
      paperVeilEl = document.createElement('div');
      paperVeilEl.className = 'paper-veil';
      paperVeilEl.setAttribute('aria-hidden', 'true');
      document.body.appendChild(paperVeilEl);
    }
    return paperVeilEl;
  }
  function closeToHome() {
    if (busy) return;
    setMenu(false);
    /* already there: the corner control has nothing to close */
    if (document.body.classList.contains('on-home')) return;
    /* and a reader who asked for less motion gets the front door, not the move */
    if (REDUCED) { window.showPage('home'); return; }
    busy = true;
    var v = paperVeil();
    /* The front door's own arrival is waiting behind the paper for this move,
       and its first beat is the beat the paper is held for. The one number is
       written here, into the property the stylesheet reads, rather than kept in
       both places and hoped over. */
    document.body.style.setProperty('--return-wait', MS_CLOSE_BEAT + 'ms');
    document.body.classList.add('is-returning', 'is-closing', 'is-locked');
    /* The page being left stops gliding for the swap: the site smooth-scrolls,
       and a scroll reset on the way out would still be travelling under the
       paper and land on the front door mid-move. The window is put back at the
       top instantly, which is what the reference's own scrollbar does. */
    var root = document.documentElement;
    var scrollBehaviour = root.style.scrollBehavior;
    root.style.scrollBehavior = 'auto';
    /* The arrival a page wears when it opens belongs to opening a page. The
       front door comes up under its own paper here, and a second thing moving
       under a fade is exactly the muddle this move exists to avoid, so the
       class comes off before the swap and the wrapper leaves it off while the
       paper is down. */
    document.body.classList.remove('page-enter');
    v.style.transition = 'none';
    v.style.opacity = '0';
    void v.offsetWidth;
    v.style.transition = 'opacity ' + MS_CLOSE_OUT + 'ms ' + OPEN_EASE;
    v.style.opacity = '1';
    window.setTimeout(function () {
      window.showPage('home');   /* the swap happens under the paper */
      root.style.scrollBehavior = scrollBehaviour;
      window.setTimeout(function () {
        document.body.classList.remove('is-locked');
        /* The paper leaves on the settle curve, not the stage curve. The stage
           curve is slow off the mark and all of it at the end, which is right
           for a page dissolving and wrong for a sheet that has to get out of
           the way: on it, nine tenths of the row's appearance lands in the last
           150ms and nobody sees a row arrive. This way the paper is clear
           inside the first third of the lift and the plates are what is left to
           watch. The row's own arrival is not waited for, it starts at the
           same instant, on the beat below, and finishes about when the last
           plate of the reference's own row does. */
        v.style.transition = 'opacity ' + MS_CLOSE_IN + 'ms ' + SETTLE_EASE;
        v.style.opacity = '0';
        /* The move is over when the last plate is up, not when the paper is:
           the row's own arrival is the longer of the two by a third of a
           second, and taking the class off early would cut its last plate
           mid-rise. */
        window.setTimeout(function () {
          v.style.transition = 'none';
          document.body.classList.remove('is-closing', 'is-returning');
          busy = false;
        }, Math.max(MS_CLOSE_IN, MS_CLOSE_ROW) + 40);
      }, MS_CLOSE_BEAT);
    }, MS_CLOSE_OUT);
  }

  /* =========================================== page switcheroo hooks */
  /* The browser's own back button is the history API's job, and the menu's
     cross closes the menu rather than walking a trail of pages. There is no
     second back stack to keep in step with the real one. */
  var currentPage = null;

  var originalShowPage = window.showPage;
  window.showPage = function (name, updateUrl) {
    if (!document.getElementById('page-' + name)) name = 'home';
    /* Going home from anywhere behind the front door is the paper close, and it
       is caught here rather than at each of the places that ask for it: the
       corner control, the panel's links, the wordmark and the browser's own
       back all end up in this function, and one way home
       is the point. The veil's own swap calls this with the paper already down,
       so it passes straight through. */
    if (name === 'home' && currentPage && currentPage !== 'home' &&
        !document.body.classList.contains('is-closing') && !REDUCED) {
      closeToHome();
      return;
    }
    currentPage = name;
    originalShowPage(name, updateUrl === undefined ? true : updateUrl);
    syncShell(name);
    if (!document.body.classList.contains('is-closing')) {
      document.body.classList.remove('page-enter');
      void document.body.offsetWidth;
      document.body.classList.add('page-enter');
    }
    /* The page that has just come up has its own blocks to arrive, and they
       have been waiting un-arrived while the page was off screen. Its blocks
       are read again here as well: they were read while the page had no box at
       all, so none of them was ever a line, see maskBlock and refreshPage. */
    window.requestAnimationFrame(function () { refreshPage(document.getElementById('page-' + name)); });
    /* and a take belongs to the artist's page: it does not follow you out */
    if (player && name !== 'about') player.stop();
  };

  function syncShell(name) {
    var onHome = name === 'home';
    document.body.classList.toggle('on-home', onHome);
    document.body.classList.toggle('on-about', name === 'about');
    /* the corner control says what it will do: open the sections on the front
       door, go back to the front door from anywhere else */
    if (menuBtn) {
      var corner = onHome ? 'All sections' : 'Back to home';
      menuBtn.title = corner;
      menuBtn.setAttribute('aria-label', corner);
    }
    if (menu) {
      menu.querySelectorAll('button[data-page]').forEach(function (b) {
        b.classList.toggle('is-here', b.dataset.page === name);
      });
    }
    /* The home page always opens at rest. A plate is only ever selected under a
       pointer that is on it, that is the whole of the strip's behaviour, so
       a home page you come back to cannot wear the plate you left it on: the
       row goes back to its grey wash and the caption to its resting line, which
       is what the page looks like until a hand touches it. */
    if (name === 'home') focusCard(-1, true);
  }

  /* =================================================== top bar
     The bar is the wordmark, the pipeline's state under it and the menu in
     the corner, and no more. The back-to-top triangle that used to sit in
     the top left corner is gone with the dock: on every page the wordmark
     itself goes home, and the home page never scrolls, so there was nothing
     for a second control to do. */
  var topbar = document.querySelector('.topbar');
  function onScroll() {
    if (topbar) topbar.classList.toggle('is-scrolled', window.scrollY > 8);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* =========================================== split lead text into words
     (runs before the reveal wiring below so the words exist to light up) */
  document.querySelectorAll('[data-lead]').forEach(function (lead) {
    var frag = document.createDocumentFragment();
    Array.prototype.slice.call(lead.childNodes).forEach(function (node) {
      if (node.nodeType === 3) {
        node.textContent.split(/(\s+)/).forEach(function (tok) {
          if (!tok) return;
          if (/^\s+$/.test(tok)) { frag.appendChild(document.createTextNode(tok)); return; }
          var s = document.createElement('span');
          s.className = 'w';
          s.textContent = tok;
          frag.appendChild(s);
        });
      } else {
        node.classList.add('chip');
        frag.appendChild(node);
      }
    });
    lead.innerHTML = '';
    lead.appendChild(frag);
  });

  /* =================================================== the arrival, staged
     Every block that has something to say, the artist's four chapters, and
     every section on the pages behind the door, arrives as it comes up the
     screen, and it arrives in reading order. Type is not faded in: this
     measures where each block's lines actually broke, rebuilds the block as one
     masked line per line it already printed, and the stylesheet pulls each line
     up out of the line above it one behind the other, so the words are set
     rather than summoned. Everything that is not type lifts in behind the type,
     and a set of things lifts one after another.

     The mark comes off again when a block has been left wholly below the window,
     so scrolling back up and down reassembles the page instead of playing it
     once. That is the only place it comes off, and it never comes off while the
     block's own arrival is still running: an animation told to stop halfway and
     start again is exactly what the first cut of this did, and it read as a
     broken page rather than as a page that had not been reached yet. */
  var STAGE_SEL = '[data-reveal], #page-about .about-sec, #page-about footer,' +
    ' .page:not(#page-home) .section, .page:not(#page-home) hr.divider,' +
    ' .page:not(#page-home) .demo-intro, .page:not(#page-home) .demo-layout,' +
    ' .page:not(#page-home) footer';
  var LINE_SEL = 'p, h1, h2, blockquote, .about-standfirst, .pull, .section-title';
  var stages = Array.prototype.slice.call(document.querySelectorAll(STAGE_SEL));

  function blockish(el) {
    var d = window.getComputedStyle(el).display;
    return d === 'block' || d === 'flex' || d === 'grid' || d === 'list-item' || d === 'table' || d === 'flow-root';
  }
  /* a block that is nothing but running text: no block-level child inside it,
     and none of the leads, which run their own word-by-word light */
  function onlyText(el) {
    if (!el || el.nodeType !== 1 || !el.textContent.trim()) return false;
    if (el.closest('[data-lead]')) return false;
    var d = window.getComputedStyle(el).display;
    if (d !== 'block' && d !== 'list-item') return false;
    var kids = el.children;
    for (var i = 0; i < kids.length; i++) {
      if (blockish(kids[i])) return false;
    }
    return true;
  }
  /* A block read as a list of words: every word its own inline box, each one
     keeping the look of the element it was written in, and a flag saying
     whether a space stood in front of it. Two things this is careful about.
     The spaces are the original ones, so "Review.\"” does not become "Review ."
     and a line is not measured a word too long. And a word inside an em or a
     strong is wrapped in a copy of that element word by word, because an em
     that runs over a line break would otherwise be one object standing in two
     lines, which is the one thing a mask cannot represent. Anything with
     elements inside it, a chip with an image, a link with a span in it, is
     left whole; if one of those turns out to stand in two lines the block is
     not masked at all (see below). */
  function readTokens(el) {
    var items = [], gap = false;
    function walk(parent, template) {
      Array.prototype.slice.call(parent.childNodes).forEach(function (n) {
        if (n.nodeType === 3) {
          var txt = n.textContent, re = /\S+/g, m, last = 0;
          while ((m = re.exec(txt))) {
            if (/\s/.test(txt.slice(last, m.index))) gap = true;
            var holder;
            if (template) { holder = template.cloneNode(false); holder.textContent = m[0]; }
            else { holder = document.createElement('span'); holder.textContent = m[0]; }
            items.push({ node: holder, space: items.length ? gap : false, word: true });
            gap = false;
            last = m.index + m[0].length;
          }
          /* the whitespace after the last word of a text node belongs to the
             word that comes after it, which is usually in the next node: the
             space in "including <em>Fuse" lives at the tail of the text node */
          if (/\s/.test(txt.slice(last))) gap = true;
        } else if (n.nodeType === 1) {
          if (n.children.length) {
            items.push({ node: n, space: items.length ? gap : false, word: false });
            gap = false;
          } else {
            walk(n, n);
          }
        }
      });
    }
    walk(el, null);
    return items;
  }

  /* rebuild one text block as masked lines, or leave it alone: a block whose
     lines cannot be trusted is lifted in whole rather than clipped, and a block
     whose masking would move anything by so much as a pixel is lifted in whole
     too. That last test is what makes this safe to run over every page: a mask
     is kept only where it is invisible to the layout, so building the masks
     cannot shift a single paragraph on the page. */
  function maskBlock(el, b) {
    var rec = el.__stage;
    if (!rec) rec = el.__stage = { html: el.innerHTML, b: b, pending: false };
    else el.innerHTML = rec.html;
    rec.b = b;

    /* A block on a page that is not on screen has no box at all, so it has no
       lines to read. Reading it there would put every word of the paragraph in
       one line, and the whole block would arrive as a single slab instead of
       line by line, which looks worse than not arriving at all. It is left
       exactly as it was written, and read again the moment its page is opened
       (see refreshPage below). */
    if (!el.getBoundingClientRect().height) {
      rec.pending = true;
      unmask(el, b);
      return;
    }
    rec.pending = false;

    var heightBefore = el.getBoundingClientRect().height;
    var items = readTokens(el);
    if (!items.length) return;

    var probe = document.createDocumentFragment();
    items.forEach(function (it) {
      if (it.space && probe.childNodes.length) probe.appendChild(document.createTextNode(' '));
      probe.appendChild(it.node);
    });
    el.innerHTML = '';
    el.appendChild(probe);

    /* an object standing in two lines, a chip, a link, a block of code, has
       no one line to belong to, so nothing in this block is masked */
    var wordHeight = 0, words = 0, tall = false;
    items.forEach(function (it) {
      var h = it.node.getBoundingClientRect().height;
      if (it.word) { wordHeight += h; words += 1; }
    });
    wordHeight = words ? wordHeight / words : 0;
    items.forEach(function (it) {
      if (!it.word && wordHeight && it.node.getBoundingClientRect().height > wordHeight * 1.8) tall = true;
    });
    if (tall) { el.innerHTML = rec.html; unmask(el, rec.b); return; }

    var lines = [], lastTop = null;
    items.forEach(function (it) {
      var top = it.node.getBoundingClientRect().top;
      if (lastTop === null || Math.abs(top - lastTop) > 2) { lines.push([]); lastTop = top; }
      lines[lines.length - 1].push(it);
    });

    var out = document.createDocumentFragment();
    lines.forEach(function (group, gi) {
      var ln = document.createElement('span');
      ln.className = 'ln';
      var inner = document.createElement('span');
      inner.className = 'ln-i';
      inner.style.setProperty('--i', String(Math.min(gi, 9)));
      group.forEach(function (it) {
        if (it.space) inner.appendChild(document.createTextNode(' '));
        inner.appendChild(it.node);
      });
      ln.appendChild(inner);
      out.appendChild(ln);
    });
    el.innerHTML = '';
    el.appendChild(out);

    /* and the last word on it: if any line came out taller than its own mask,
       the text would be cut off, so the whole block goes back to being one
       piece that lifts. The line's own height is what is read here, not the
       mask's scroll area, that area holds the line that is still standing a
       line below it, waiting to come up */
    var clipped = false;
    Array.prototype.forEach.call(el.querySelectorAll('.ln'), function (ln) {
      var inner = ln.firstChild;
      if (inner && inner.offsetHeight > ln.clientHeight + 2) clipped = true;
    });
    if (clipped) { el.innerHTML = rec.html; unmask(el, rec.b); return; }

    /* and the block has to be exactly as tall as it was: the lines are the
       same lines, and if any one of them came out a hair taller the mask goes
       back in the box and the block arrives as one piece instead */
    if (Math.abs(el.getBoundingClientRect().height - heightBefore) > 1) {
      el.innerHTML = rec.html;
      unmask(el, rec.b);
      return;
    }

    el.style.setProperty('--b', String(Math.min(rec.b, 3)));
    /* the wave across one block is bounded: eight lines of a long paragraph
       should follow one another, not queue up for half a second. The unit is
       written on the number because the stylesheet multiplies it by the line's
       own index, a bare number there multiplies to another bare number, and
       a delay of "0.055" is not a time at all: the whole delay would be thrown
       out and every line of the block would start in the same frame */
    el.style.setProperty('--s', Math.min(0.055, 0.3 / Math.max(1, lines.length - 1)) + 's');
    el.setAttribute('data-staged', 'lines');
  }

  /* not type that can be masked: one piece that lifts instead */
  function unmask(el, b) {
    el.removeAttribute('data-staged');
    el.setAttribute('data-staged', 'block');
    el.style.setProperty('--b', String(Math.min(b, 3)));
  }

  /* One stage: read what is in it and give each thing what it needs, however
     deep it is buried.

     The walk used to stop at the stage's own children, and a child that held
     masked type was skipped outright, "it is already being animated", which
     is true of the type and false of everything standing next to it. On the
     five tool pages that meant every step of every pipeline arrived like this:
     the paragraph printed line by line, and the step's number, the file it
     lives in, its own title, the row of pill badges under its prose and the
     panel of examples beside it were simply there from the first frame. A grid
     of two cards did the same thing in a way that is worse to look at: the card
     whose text was read as lines was skipped, and the card next to it, whose
     text was not, lifted, two neighbours in one row, one arriving and one
     standing still.

     So the walk goes all the way down, and the rule it follows is the same one
     the stage's own children were already read by:

       type that can be masked      becomes lines and prints
       a set of things              arrives one after another (a group)
       one thing                    lifts (a block)
       a container holding type     is a place, not a piece: it never lifts as
                                    a slab, two screens of a step must not
                                    arrive as one rectangle, and the walk
                                    carries on inside it, so what is between its
                                    type still arrives on its own beat

     A component that is itself a wrapper is therefore never a unit, and the
     things inside it always are. --b keeps counting across the whole stage, so
     the walk still reads in order, and the stylesheet caps the delay it is
     spent on anyway. */
  function stageBlock(stage) {
    walk(stage, 0);
  }
  function walk(parent, b0) {
    var b = b0;
    Array.prototype.forEach.call(parent.children, function (child) {
      /* already read: it is either masked type or a unit of its own */
      if (child.hasAttribute('data-staged')) return;
      var tag = child.tagName.toLowerCase();
      if (tag === 'hr') { child.setAttribute('data-staged', 'rule'); return; }
      if (tag === 'ul' && child.classList.contains('rows')) {
        child.setAttribute('data-staged', 'rows');
        child.style.setProperty('--b', String(Math.min(b, 3)));
        Array.prototype.forEach.call(child.children, function (li, i) {
          li.style.setProperty('--r', String(Math.min(i, 14)));
        });
        b += 1;
        return;
      }
      /* type itself, at any depth: it prints as lines */
      if (onlyText(child)) { maskBlock(child, b); b += 1; return; }
      /* a container that holds type: read its own type, then keep going in */
      var inner = Array.prototype.slice.call(child.querySelectorAll(LINE_SEL)).filter(function (el) {
        return onlyText(el) && !el.hasAttribute('data-staged');
      });
      if (inner.length || child.querySelector('[data-staged]')) {
        child.setAttribute('data-staged', 'wrap');
        inner.forEach(function (el, i) { maskBlock(el, Math.min(b + i, 3)); });
        b += inner.length;
        b = walk(child, b);
        return;
      }
      /* nothing but components in here: one thing that lifts, or a set of
         things that lifts one after another */
      var kids = Array.prototype.slice.call(child.children);
      if (kids.length > 1 && kids.every(blockish)) {
        child.setAttribute('data-staged', 'group');
        child.style.setProperty('--b', String(Math.min(b, 3)));
        kids.forEach(function (gc, i) { gc.style.setProperty('--g', String(Math.min(i, 12))); });
        b += 1;
        return;
      }
      child.setAttribute('data-staged', 'block');
      child.style.setProperty('--b', String(Math.min(b, 3)));
      b += 1;
    });
    return b;
  }

  /* still moving? a block is never un-arrived while any of its own type is
     still coming up: half an animation played backwards is a broken page, and
     that is the whole reason the mark lives on the window's far side instead
     of just under the arrival line */
  function calm(el) {
    if (!el.getAnimations) return Date.now() - (parseFloat(el.dataset.at) || 0) > 1800;
    var running = el.getAnimations({ subtree: true });
    for (var i = 0; i < running.length; i++) {
      if (running[i].playState === 'running') return false;
    }
    return true;
  }

  /* Every block of type, and every thing that is not type, became an arrival
     unit when it was read above, and a unit arrives on its own mark, when its
     own top comes up the screen, not when the two-screen-tall section around it
     does. That is the whole difference a reader feels: nothing plays itself out
     while it is still below the fold, and nothing waits on a gate further up
     the page than itself.

     The mark comes off again only when a unit has been left wholly below the
     window, nothing of its own is still moving, and it has been standing for a
     breath or more, so scrolling back a little, the thing a hand does when it
     is deciding whether it has seen something, does not set the page off
     again. */
  var units = [];
  function collectUnits() {
    units = Array.prototype.slice.call(document.querySelectorAll('[data-staged]'));
    /* the label over a chapter draws its own rule and a rule between two blocks
       draws itself: both are arrival units in their own right, whether or not
       something around them is lifting as well */
    units = units.concat(Array.prototype.slice.call(document.querySelectorAll('.about-kicker, .section-label, hr.divider')));
  }

  function paintReveals() {
    if (!units.length) return;
    var vh = window.innerHeight || 800;
    var now = Date.now();
    for (var i = 0; i < units.length; i++) {
      var el = units[i];
      var box = el.getBoundingClientRect();
      /* a unit on a page that is not on screen is not laid out at all, so it has
         no box: it waits un-arrived, and arrives when that page is opened
         rather than having arrived while nobody was looking */
      if (!box.width && !box.height) {
        if (el.classList.contains('on')) el.classList.remove('on');
        el.classList.remove('is-set');
        el.dataset.at = '';
        continue;
      }
      var at = parseFloat(el.dataset.at) || 0;
      /* the foot of the document counts as the arrival line: a short last block
         can sit below 86% of the window even with the page scrolled all the way
         down, and it must not be left waiting there forever */
      var atFoot = (document.documentElement.scrollHeight - window.pageYOffset - vh) < 8;
      /* A unit is marked when any real part of it is in the window, the top
         has to be 32px inside, so it is not marked by the first pixel of a
         border, rather than when it reaches some line up the screen. It used
         to wait until 86% of the window, which left the bottom of the screen a
         band where blocks were on show but had not arrived: a page with a
         paragraph-shaped hole in it, at the foot of every screen. Nothing in
         the window may be waiting. */
      if (box.top < vh - 32 || (atFoot && box.top < vh)) {
        if (!at) el.dataset.at = String(now);
        el.classList.add('on');
      } else if (at && box.top > vh + 24 && now - at > 2500 && calm(el)) {
        el.classList.remove('on');
        el.classList.remove('is-set');
        el.dataset.at = '';
      }
    }
  }

  /* A page that has just been opened has to have its blocks read again: it was
     not on screen when they were first read, so they were left as they were
     written and none of them was ever a line. Opening the page is the moment
     they can be read for real, and it is also the moment they are all waiting
     un-arrived, so the page assembles itself for the visit. */
  function refreshPage(page) {
    if (!page) return;
    Array.prototype.forEach.call(page.querySelectorAll('[data-staged]'), function (u) {
      var rec = u.__stage;
      if (!rec) return;
      u.innerHTML = rec.html;
      maskBlock(u, rec.b);
    });
    collectUnits();
    paintReveals();
  }

  /* and the lines are read again when the real faces arrive, or when the window
     changes shape, because a paragraph set in another face breaks somewhere
     else. A unit that is mid-arrival is not rebuilt underneath its own running
     animation (that would start it over from the bottom, which is the one thing
     this file is careful never to do): it is rebuilt and left standing, is-set,
     rather than played again. */
  function remaskUnits() {
    units.forEach(function (u) {
      var rec = u.__stage;
      if (!rec || rec.pending) return;
      if (u.getAttribute('data-staged') !== 'lines') return;
      var hot = !calm(u);
      u.innerHTML = rec.html;
      maskBlock(u, rec.b);
      if (hot && u.getAttribute('data-staged') === 'lines') u.classList.add('is-set');
    });
    paintReveals();
  }

  /* A line is held where it printed by its own animation while it plays, and
     that belongs to the arrival and ends with it: when the last line of a block
     has printed, the block is marked settled and the animation is dropped. The
     type is then ordinary type again, ordinary type that no engine has a
     reason to keep on a layer of its own, which is what a standing `filter` or
     `will-change` would have done to every line of the site. */
  document.addEventListener('animationend', function (e) {
    var t = e.target;
    if (!t || !t.classList || !t.classList.contains('ln-i')) return;
    if (e.animationName !== 'line-up') return;
    var block = t.parentNode && t.parentNode.parentNode;
    if (!block || !block.classList || !block.classList.contains('on')) return;
    if (block.lastElementChild && t === block.lastElementChild.firstElementChild) block.classList.add('is-set');
  }, true);

  if (stages.length) {
    stages.forEach(function (s) { s.classList.add('stage'); stageBlock(s); });
    collectUnits();
    /* the lines were read off the fallback face; the moment the real faces are
       in, they are read off those instead, because a paragraph set in a wider
       face breaks somewhere else */
    if (document.fonts && document.fonts.ready && document.fonts.ready.then) {
      document.fonts.ready.then(remaskUnits);
    }
    var revealSettle = 0, revealRest = 0;
    window.addEventListener('scroll', function () {
      window.requestAnimationFrame(paintReveals);
      /* the page is smooth-scrolled, so a scroll event can land a long way
         before the window has finished moving: this is the reading taken once
         it has. The second one is taken when the window has rested as well, so
         a block that has just left the foot of the window is read again after
         its own arrival has finished rather than being left waiting to be
         noticed */
      window.clearTimeout(revealSettle);
      window.clearTimeout(revealRest);
      revealSettle = window.setTimeout(paintReveals, 140);
      revealRest = window.setTimeout(paintReveals, 1500);
    }, { passive: true });
    var revealResize = 0;
    window.addEventListener('resize', function () {
      window.clearTimeout(revealResize);
      revealResize = window.setTimeout(remaskUnits, 220);
    });
    /* And only now, with every block read and every unit built, does the page
       say it is being staged. The rules that hide what has not arrived yet hang
       off this one class, so a load in which this file never ran, or threw
       before it got this far, is a finished, readable page: type set, every
       rule drawn, nothing waiting on an animation that will never come. It goes
       on in the same tick the masks are built, before anything is painted, so
       the first frame the eye sees is already the waiting page and the arrival
       starts from it. */
    document.documentElement.classList.add('staged');
    paintReveals();
  }

  /* word-by-word reveal driven by scroll position */
  var leads = Array.prototype.slice.call(document.querySelectorAll('.word-lead'));
  function paintLead(lead) {
    var words = lead.querySelectorAll('.w');
    if (!words.length) return;
    var r = lead.getBoundingClientRect();
    var vh = window.innerHeight || 800;
    var p = (vh * 0.82 - r.top) / Math.max(1, r.height + vh * 0.22);
    p = Math.max(0, Math.min(1, p));
    var upto = Math.ceil(p * words.length);
    words.forEach(function (w, i) { w.classList.toggle('on', i < upto); });
    lead.querySelectorAll('.chip').forEach(function (c) { c.classList.toggle('on', p > 0.5); });
  }
  function paintAll() { leads.forEach(paintLead); }
  if (leads.length) {
    window.addEventListener('scroll', function () { window.requestAnimationFrame(paintAll); }, { passive: true });
    window.addEventListener('resize', paintAll);
    paintAll();
  }

  /* =================================================== work-list filters */
  document.querySelectorAll('[data-row-filter]').forEach(function (bar) {
    var rows = document.querySelectorAll(bar.dataset.rowFilter + ' li.row');
    bar.querySelectorAll('.chip-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        bar.querySelectorAll('.chip-tab').forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var f = btn.dataset.filter;
        /* a row carries the kinds it belongs to, so a work that is both a song
           and a performance answers to either chip */
        rows.forEach(function (r) {
          var kinds = (r.dataset.kind || '').split(/\s+/);
          r.classList.toggle('is-hidden', !(f === 'all' || kinds.indexOf(f) > -1));
        });
      });
    });
  });

  /* =================================================== the works list, playing
     The list opens with everything Archer delivered as a file: the songs he
     made with the people they are named for, and the three performances that
     were recorded. Those rows play the real take, put where the frontend can
     serve it by tools/make_work_audio.py. The works under them have no
     recording, only footage on the artist's own site, so their ring is a link
     and it opens that video in a new tab instead.

     One <audio> element is reused for the whole list, so starting a row always
     stops the row before it and only one row can ever be in its playing state.
     The click is caught on the list itself rather than on the button, because
     the whole row is the target a hand reaches for; the ring is a real button
     (or a real link), so the keyboard reaches the same place through it.

     The hairline under the row is the recording's own position in it, written
     out each frame as a custom property the stylesheet scales from the left.
     That is a running frame loop, so it is started with the sound and stopped
     with it, and it stops the sound as soon as the artist's page stops being
     the page on screen. */
  /* `player` is what the line at the top of the page reads the take through:
     the list owns the sound and hands over three answers and one command, so
     the pen and the row can never disagree about where the song is. */
  var player = null;
  var curvePaint = null;

  var workList = document.getElementById('about-rows');
  if (workList) {
    var take = new Audio();
    var activeRow = null;
    var fillFrame = 0;
    var pendingSeek = null;

    function paintFill() {
      if (!activeRow) return;
      var page = document.getElementById('page-about');
      if (page && !page.classList.contains('active')) { stopTake(); return; }
      var d = take.duration;
      var p = (d && isFinite(d)) ? take.currentTime / d : 0;
      activeRow.style.setProperty('--p', String(Math.max(0, Math.min(1, p))));
      /* the pen at the top of the page is filled from the same frame, so the
         line and the hairline under the row are one reading, not two */
      if (curvePaint) curvePaint();
      fillFrame = window.requestAnimationFrame(paintFill);
    }
    function startFill() {
      if (fillFrame) return;
      fillFrame = window.requestAnimationFrame(paintFill);
    }
    function haltFill() {
      if (fillFrame) window.cancelAnimationFrame(fillFrame);
      fillFrame = 0;
    }
    /* Two marks, not one. `playing` means the take is loaded on this row: the
       ring goes teal and the hairline holds wherever the take has got to, so a
       paused row still shows where it was. `sounding` means sound is actually
       running, and that is what swaps the glyph to the pause bars. */
    function stopTake() {
      haltFill();
      take.pause();
      document.body.classList.remove('take-loaded');
      if (!activeRow) return;
      activeRow.classList.remove('playing', 'sounding');
      activeRow.style.setProperty('--p', '0');
      activeRow = null;
    }
    function playRow(row) {
      if (activeRow === row) {
        if (take.paused) {
          row.classList.add('sounding');
          take.play().catch(function () {});
          startFill();
        } else {
          take.pause();
          haltFill();
          row.classList.remove('sounding');
        }
        return;
      }
      var btn = row.querySelector('.row-play');
      if (!btn || !btn.dataset.audio) return;
      stopTake();
      activeRow = row;
      take.src = btn.dataset.audio;
      document.body.classList.add('take-loaded');
      row.classList.add('playing', 'sounding');
      take.play().catch(function () {
        /* a take that will not start is a missing file, not a broken row:
           the row goes back to its resting state and says nothing */
        stopTake();
      });
      startFill();
    }
    workList.addEventListener('click', function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      /* a link inside the row is the browser's to open in its own tab, so it
         is never also the row's own tap: the row's tap plays what the row has
         in hand, and a row whose only media is a video hands the click back */
      if (t.closest('a')) return;
      var row = t.closest('li.row');
      if (!row) return;
      if (row.querySelector('.row-play[data-audio]')) { playRow(row); return; }
      var video = row.querySelector('.row-play[href]');
      if (video) window.open(video.href, '_blank', 'noopener');
    });
    take.addEventListener('ended', stopTake);
    take.addEventListener('loadedmetadata', function () {
      /* a seek asked for before the take knew its own length is served here */
      if (pendingSeek === null) return;
      take.currentTime = pendingSeek * take.duration;
      pendingSeek = null;
    });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stopTake();
    });

    player = {
      loaded: function () { return !!take.src; },
      /* a take that is only paused is still loaded, and the frame loop that
         would otherwise notice the page has changed is not running while it
         is paused: leaving the artist's page has to say so itself */
      stop: stopTake,
      progress: function () {
        var d = take.duration;
        return (d && isFinite(d)) ? Math.max(0, Math.min(1, take.currentTime / d)) : 0;
      },
      duration: function () {
        var d = take.duration;
        return (d && isFinite(d)) ? d : 0;
      },
      seek: function (p) {
        if (!take.src) return false;
        p = Math.max(0, Math.min(1, p));
        var d = take.duration;
        if (d && isFinite(d)) { take.currentTime = p * d; pendingSeek = null; }
        else { pendingSeek = p; }
        if (activeRow) activeRow.style.setProperty('--p', String(p));
        return true;
      }
    };
  }

  /* =================================================== the line, as a seek bar
     The one drawing on the artist's page is also its one instrument. The pen's
     own path is sampled once into a list of points, and every gesture on the
     line, the pointer moving over it, a click landing on it, is answered by
     the nearest of those points, so the answer is the same wherever the hand
     happens to be: point at the foot of the S and you are near the end of the
     song, point at the curl at the top and you are near the start. The marker
     rides the stroke in the drawing's own units, and the time beside it is
     placed by converting that same point back out to the window, so the two
     never drift apart however the page is scaled. */
  var curveSvg = document.querySelector('#page-about .about-curve');
  var curvePen = curveSvg && curveSvg.querySelector('.curve-pen');
  var curvePlay = curveSvg && curveSvg.querySelector('.curve-play');
  var curveMark = curveSvg && curveSvg.querySelector('.curve-mark');
  var curveTime = document.querySelector('#page-about .curve-time');
  if (curveSvg && curvePen && curvePlay && player) {
    var STEPS = 180;
    var penLength = curvePen.getTotalLength();
    var penPoints = [];
    for (var st = 0; st <= STEPS; st++) penPoints.push(curvePen.getPointAtLength(penLength * st / STEPS));

    function clock(seconds) {
      var s = Math.max(0, Math.round(seconds));
      return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
    }
    function nearestIndex(clientX, clientY) {
      var m = curvePen.getScreenCTM();
      if (!m) return -1;
      var p = curveSvg.createSVGPoint();
      p.x = clientX; p.y = clientY;
      var local = p.matrixTransform(m.inverse());
      var best = 0, bestD = Infinity;
      for (var i = 0; i < penPoints.length; i++) {
        var dx = penPoints[i].x - local.x, dy = penPoints[i].y - local.y;
        var d2 = dx * dx + dy * dy;
        if (d2 < bestD) { bestD = d2; best = i; }
      }
      return best;
    }
    function showAt(i) {
      if (i < 0) return;
      var f = i / STEPS;
      var here = penPoints[i];
      var d = player.duration();
      /* a take whose length is not known yet has no time to point at */
      if (!curveMark || !d) { hideMark(); return; }
      curveMark.setAttribute('cx', here.x);
      curveMark.setAttribute('cy', here.y);
      curveMark.classList.add('on');
      if (!curveTime) return;
      var m = curvePen.getScreenCTM();
      if (!m) return;
      var onScreen = here.matrixTransform(m);
      var hero = curveSvg.parentNode;
      var box = hero.getBoundingClientRect();
      curveTime.textContent = clock(f * d);
      curveTime.style.transform = 'translate(' + (onScreen.x - box.left + 16) + 'px,' + (onScreen.y - box.top - 9) + 'px)';
      curveTime.style.opacity = '0.85';
    }
    function hideMark() {
      if (curveMark) curveMark.classList.remove('on');
      if (curveTime) curveTime.style.opacity = '0';
    }
    curvePaint = function () { curvePlay.style.strokeDashoffset = String(1 - player.progress()); };
    curveSvg.addEventListener('pointermove', function (e) {
      if (!player.loaded()) return;
      showAt(nearestIndex(e.clientX, e.clientY));
    });
    curveSvg.addEventListener('pointerleave', hideMark);
    curveSvg.addEventListener('pointerdown', function (e) {
      if (!player.loaded()) return;
      var i = nearestIndex(e.clientX, e.clientY);
      if (i < 0) return;
      player.seek(i / STEPS);
      showAt(i);
      curvePaint();
    });
  }

  /* =================================================== go */
  focusCard(-1, true);
  var here = document.querySelector('.page.active');
  currentPage = here ? here.id.replace('page-', '') : 'home';
  syncShell(currentPage);
  /* The entrance is the page's own arrival, and it was only ever worn by a page
     that had gone through showPage, which a page opened straight from the
     address bar never does: the shell picks the page up here instead. So the
     class goes on here too, and the artist's page draws its line and lifts its
     sentence for whoever opened it and however they got here. The home page is
     left out on purpose: its own arrival is the reference's, timed value for
     value, and an extra fade over the top of it would only muddy it. */
  if (currentPage !== 'home') {
    document.body.classList.remove('page-enter');
    void document.body.offsetWidth;
    document.body.classList.add('page-enter');
  }

  /* =================================================== the arrival
     The reference plays this once, when the home page first appears: the
     plate in the middle of the row springs out of a sliver, then the six
     around it sweep in a beat later, and the caption rises under them. The
     whole of it lives in the stylesheet (`body.is-entering`) and the markup
     ships the class, so the very first paint is already the sliver rather
     than a flash of the finished strip. Script only does the one thing CSS
     cannot: it takes the class off once the arrival is over, after the
     caption has finished rising, so nothing is left holding a transform.
     Landing straight on a section, or asking for less motion, skips the
     whole thing. */
  var ENTER_MS = 3500;
  var arriving = document.body.classList.contains('is-entering');
  function arrived() {
    if (!arriving) return;
    arriving = false;
    document.body.classList.remove('is-entering');
    ['pointerdown', 'keydown', 'touchstart', 'wheel'].forEach(function (ev) {
      window.removeEventListener(ev, arrived);
    });
  }
  if (arriving) {
    /* A document that has never been painted holds its animations at frame
       zero, a page opened in a background tab, or one the window has not got
       to yet, shows the empty first frame of the arrival the moment it is
       brought forward. Rather than land on that, the arrival is skipped for
       those loads: the page is simply there, finished, the way a page that
       never animated looks. */
    if (REDUCED || document.hidden || (here && here.id !== 'page-home')) {
      arrived();
    } else {
      ['pointerdown', 'keydown', 'touchstart', 'wheel'].forEach(function (ev) {
        window.addEventListener(ev, arrived, { passive: true });
      });
      window.setTimeout(arrived, ENTER_MS);
    }
  }
})();
