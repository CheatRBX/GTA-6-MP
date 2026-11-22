// The scroll story: key frames for the 3D stage plus all scroll-linked UI.
// Stage states are interpolated between key frames placed on the page, so the result is
// deterministic (same scroll position -> same frame), even after a reload in the middle of the page.
import { PARTS } from './scene.js';

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (t) => t * t * (3 - 2 * t);

export const BASE = { ry: 125, zoom: 1, elev: 11, fov: 27, tx: 0, ty: 0.22, tz: 0, shiftX: 0, shiftY: 0, explode: 0, exposure: 1.2, dims: 0, fade: 1, plinth: 1, cut: 10 };

export function createStory({ stage }) {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const sec = {
    hero: $('#hero'), about: $('#about'), tour: $('#tour'), explode: $('#explode'), specs: $('#specs'),
    match: $('#match'), process: $('#process'), values: $('#values'), join: $('#join'), foot: $('#foot'),
  };
  const target = { ...BASE };           // what the stage eases towards
  let stops = [];
  let mobile = false;
  const stations = $$('.station');
  const tourRailItems = $$('#tourRail li');
  const steps = $$('.step');
  const valueItems = $$('.values__list li');
  const phases = $$('.phase');
  const clockTime = $('#clockTime');
  const segs = $$('.seg i');
  const tags = {};
  const tagsEl = $('#tags');
  const dimNS = 'http://www.w3.org/2000/svg';
  const overlay = $('#overlay');
  let dimEls = [];
  let lang = 'ru';

  // ------------------------------------------------------------ tags (callouts)
  const RU_NAMES = { chassis: 'Шасси', intake: 'Захват', turret: 'Турель и шутер', lift: 'Задний блок', electronics: 'Электроника' };
  for (const part of PARTS) {
    const el = document.createElement('div');
    el.className = 'tag'; el.dataset.part = part;
    el.innerHTML = `<i class="tag__dot"></i><i class="tag__line"></i><span class="tag__label mono" data-i18n="part.${part}">${RU_NAMES[part]}</span>`;
    tagsEl.appendChild(el);
    tags[part] = { el, vis: 0 };
  }
  const tagOpacity = Object.fromEntries(PARTS.map((p) => [p, 0]));

  // ------------------------------------------------------------ dimension overlay
  function buildDims() {
    overlay.innerHTML = '';
    dimEls = (stage.dimAnchors || []).map((d) => {
      const g = document.createElementNS(dimNS, 'g');
      const mk = (name, cls) => { const e = document.createElementNS(dimNS, name); if (cls) e.setAttribute('class', cls); g.appendChild(e); return e; };
      const ext1 = mk('line', 'dim dim--ext'), ext2 = mk('line', 'dim dim--ext'), line = mk('line', 'dim');
      const t1 = mk('line', 'tick'), t2 = mk('line', 'tick');
      const text = mk('text'); text.setAttribute('text-anchor', 'middle');
      text.setAttribute('style', 'paint-order:stroke;stroke:#0d0d0e;stroke-width:7px;stroke-linejoin:round');
      overlay.appendChild(g);
      return { d, g, ext1, ext2, line, t1, t2, text };
    });
  }

  function updateDims(cur) {
    if (!dimEls.length) return;
    const op = clamp(cur.dims);
    overlay.style.opacity = op;
    if (op < 0.01) return;
    const unit = lang === 'ru' ? 'мм' : 'mm';
    for (const e of dimEls) {
      const p = e.d.pts.map((o) => stage.project(o));
      const set = (l, a, b) => { l.setAttribute('x1', a.x); l.setAttribute('y1', a.y); l.setAttribute('x2', b.x); l.setAttribute('y2', b.y); };
      set(e.ext1, p[0], p[2]); set(e.ext2, p[1], p[3]); set(e.line, p[2], p[3]);
      const dx = p[3].x - p[2].x, dy = p[3].y - p[2].y, L = Math.hypot(dx, dy) || 1;
      const nx = -dy / L * 7, ny = dx / L * 7, ux = dx / L * 7, uy = dy / L * 7;
      const tk = (l, q) => { l.setAttribute('x1', q.x - nx - ux); l.setAttribute('y1', q.y - ny - uy); l.setAttribute('x2', q.x + nx + ux); l.setAttribute('y2', q.y + ny + uy); };
      tk(e.t1, p[2]); tk(e.t2, p[3]);
      const mx = (p[2].x + p[3].x) / 2, my = (p[2].y + p[3].y) / 2;
      e.text.setAttribute('x', mx); e.text.setAttribute('y', my - 9);
      e.text.textContent = `${e.d.label}  ${Math.round(e.d.value)} ${unit}`;
      e.g.style.display = L < 40 ? 'none' : '';
    }
  }

  // ------------------------------------------------------------ key frames
  function anchorLocal(name) {
    const a = stage.anchorBase && stage.anchorBase[name];
    if (!a) return { tx: 0, ty: BASE.ty, tz: 0 };
    return { tx: a.x + stage.shift.x, ty: a.y + stage.shift.y, tz: a.z + stage.shift.z };
  }
  const F = (o) => ({ ...BASE, ...o });

  function frames() {
    const m = mobile;
    const sx = (v) => (m ? 0 : v);              // horizontal lens shift only on wide screens
    const sy = (up) => (m ? up : 0);            // on phones the robot sits in the upper part instead
    const Z = (d, p) => (m ? p : d);
    // keep the whole robot in frame, only lean the camera towards the subsystem being described
    const station = (part, o) => {
      const a = anchorLocal(part), k = 0.5;
      return F({ tx: a.tx * k, ty: lerp(BASE.ty, a.ty, k), tz: a.tz * k, ...o });
    };

    const S = {
      hero:   F({ ry: 125, zoom: Z(1.12, 1.3), elev: 20, dims: 0, shiftX: sx(0.04), shiftY: m ? 0.1 : -0.05 }),
      about:  F({ ry: 205, zoom: Z(1.28, 1.4), elev: 14, shiftX: sx(0.24), shiftY: sy(0.2), exposure: 1.1, fade: m ? 0.5 : 1 }),
      chassis: station('chassis', { ry: 330, zoom: Z(1.08, 1.3), elev: 8, shiftX: sx(0.2), shiftY: sy(0.02) }),
      intake:  station('intake', { ry: 420, zoom: Z(1.04, 1.3), elev: 14, shiftX: sx(0.2), shiftY: sy(0.02) }),
      turret:  station('turret', { ry: 490, zoom: Z(1.0, 1.3), elev: 24, shiftX: sx(0.2), shiftY: sy(0.02) }),
      lift:    station('lift', { ry: 610, zoom: Z(1.08, 1.3), elev: 12, shiftX: sx(0.2), shiftY: sy(0.02) }),
      electronics: station('electronics', { ry: 700, zoom: Z(1.04, 1.3), elev: 10, shiftX: sx(0.2), shiftY: sy(0.02) }),
      whole:   F({ ry: 790, zoom: Z(1.3, 1.5), elev: 16, shiftX: sx(0.22), shiftY: sy(0.18) }),
      blown:   F({ ry: 840, zoom: Z(1.8, 2.35), elev: 15, shiftX: sx(0.2), shiftY: sy(0.0), explode: 1, ty: 0.38 }),
      cutA:    F({ ry: 880, zoom: Z(1.8, 2.35), elev: 20, shiftX: sx(0.2), shiftY: sy(0.0), explode: 1, ty: 0.38, cut: 0.78, exposure: 1.1 }),
      cutB:    F({ ry: 935, zoom: Z(1.8, 2.35), elev: 26, shiftX: sx(0.2), shiftY: sy(0.0), explode: 1, ty: 0.38, cut: 0.0, exposure: 1.1 }),
      back:    F({ ry: 965, zoom: Z(1.8, 2.35), elev: 16, shiftX: sx(0.2), shiftY: sy(0.0), explode: 1, ty: 0.38, cut: 0.78 }),
      blown2:  F({ ry: 990, zoom: Z(1.8, 2.35), elev: 14, shiftX: sx(0.2), shiftY: sy(0.0), explode: 1, ty: 0.38 }),
      specs:   F({ ry: 1040, zoom: Z(1.22, 1.9), elev: 22, shiftX: sx(-0.2), shiftY: sy(0.34), dims: 1 }),
      specs2:  F({ ry: 1110, zoom: Z(1.22, 1.9), elev: 22, shiftX: sx(-0.2), shiftY: sy(0.34), dims: 1 }),
      match:   F({ ry: 1150, zoom: Z(2.25, 2.3), elev: 14, shiftX: sx(-0.27), shiftY: m ? -0.3 : -0.27, exposure: 1.1 }),
      process: F({ ry: 1190, zoom: Z(1.95, 2.0), elev: 15, shiftX: sx(-0.27), shiftY: m ? 0.24 : -0.2, exposure: 1.05, fade: m ? 0.2 : 1 }),
      values:  F({ ry: 1230, zoom: Z(2.0, 2.2), elev: 12, shiftX: sx(0.36), shiftY: m ? 0.3 : -0.05, exposure: 0.95, fade: m ? 0.25 : 0.14 }),
      join:    F({ ry: 1285, zoom: Z(1.24, 2.1), elev: 12, shiftX: sx(0.25), shiftY: m ? -0.36 : 0 }),
      end:     F({ ry: 1530, zoom: Z(1.3, 1.4), elev: 10, shiftX: 0, shiftY: m ? 0.12 : 0.17 }),
    };
    return S;
  }

  // positions (scrollY) of each key frame
  function layout() {
    mobile = innerWidth < 760;
    const S = frames();
    const H = innerHeight;
    const top = (el) => el.getBoundingClientRect().top + scrollY;
    const h = (el) => el.offsetHeight;
    const list = [];
    const at = (y, state) => list.push({ y, state });

    at(0, S.hero);
    at(top(sec.hero) + H * 0.9, S.hero);
    at(top(sec.about) + h(sec.about) * 0.35, S.about);

    // tour: pinned for (height - H); five stations, each with a hold in the middle
    const tT = top(sec.tour), range = h(sec.tour) - H;
    const names = ['chassis', 'intake', 'turret', 'lift', 'electronics'];
    names.forEach((n, i) => {
      const a = tT + (i / 5) * range, b = tT + ((i + 1) / 5) * range;
      at(lerp(a, b, 0.3), S[n]);
      at(lerp(a, b, 0.72), S[n]);
    });
    // explode: assembled -> apart (layer by layer) -> section plane sweeps down and back -> together again
    const eT = top(sec.explode), eR = h(sec.explode) - H;
    at(eT - H * 0.1, S.whole);
    at(eT + eR * 0.06, S.whole);
    at(eT + eR * 0.26, S.blown);
    at(eT + eR * 0.40, S.cutA);
    at(eT + eR * 0.70, S.cutB);
    at(eT + eR * 0.80, S.back);
    at(eT + eR * 0.88, S.blown2);
    at(eT + eR + H * 0.55, { ...S.specs, explode: 0 });
    // specs
    at(top(sec.specs) + h(sec.specs) * 0.6, S.specs2);
    // match
    const mT = top(sec.match), mR = h(sec.match) - H;
    at(mT + mR * 0.15, S.match);
    at(mT + mR + H * 0.2, S.match);
    // process / values / join / footer
    at(top(sec.process) + h(sec.process) * 0.5, S.process);
    at(top(sec.values) - H * 0.15, S.values);
    at(top(sec.values) + h(sec.values) * 0.7, S.values);
    at(top(sec.join) + h(sec.join) * 0.4, S.join);
    at(top(sec.foot) - H * 0.2, S.end);
    at(document.documentElement.scrollHeight, S.end);

    list.sort((a, b) => a.y - b.y);
    stops = list;
  }

  function evaluate(y) {
    let i = 0;
    while (i < stops.length - 2 && y > stops[i + 1].y) i++;
    const a = stops[i], b = stops[i + 1];
    const t = b.y === a.y ? 1 : smooth(clamp((y - a.y) / (b.y - a.y)));
    for (const k in target) target[k] = lerp(a.state[k], b.state[k], t);
  }

  // ------------------------------------------------------------ scroll-linked UI
  const words = [];
  function splitStatement() {
    const el = $('#statement');
    if (!el) return;
    const text = el.textContent;
    el.setAttribute('aria-label', text);
    el.innerHTML = text.split(/\s+/).map((w) => `<span class="w" aria-hidden="true">${w}</span>`).join(' ');
    words.length = 0; words.push(...$$('.w', el));
  }

  function update(y, cur) {
    evaluate(y);
    const H = innerHeight;
    const top = (el) => el.getBoundingClientRect().top + y;

    // hero word parallax
    const hw = $('.hero__word');
    if (hw && y < H * 1.2) { hw.style.transform = `translate3d(0, ${y * 0.22}px, 0)`; }

    // about words
    if (words.length) {
      const p = clamp((y - (top(sec.about) - H * 0.55)) / (H * 0.85));
      const n = words.length;
      words.forEach((w, i) => { w.style.opacity = (0.16 + 0.84 * clamp((p * (n + 6) - i) / 5)).toFixed(3); });
    }

    // tour stations
    const tT = top(sec.tour), range = sec.tour.offsetHeight - H;
    const tp = clamp((y - tT) / range);
    const inTour = y > tT - H * 0.5 && y < tT + range + H * 0.5;
    const idx = Math.min(4, Math.floor(tp * 5 + 1e-6));
    const local = tp * 5 - idx;
    stations.forEach((el, i) => {
      let o = 0;
      if (inTour && i === idx) o = clamp(local / 0.14) * clamp((1 - local) / 0.14);
      if (tp <= 0 && i === 0) o = clamp((y - (tT - H * 0.35)) / (H * 0.35));
      if (tp >= 1 && i === 4) o = clamp(1 - (y - (tT + range)) / (H * 0.35));
      el.style.opacity = o.toFixed(3);
      el.style.transform = `translate3d(0, ${((1 - o) * 26).toFixed(1)}px, 0)`;
      el.style.pointerEvents = o > 0.6 ? 'auto' : 'none';
    });
    tourRailItems.forEach((li, i) => {
      li.classList.toggle('is-active', i === idx && inTour);
      li.style.setProperty('--p', i < idx ? 1 : i === idx ? clamp(local) : 0);
    });
    const tourAlpha = clamp((y - (tT - H * 0.5)) / (H * 0.4)) * clamp(1 - (y - (tT + range + H * 0.1)) / (H * 0.4));
    $('.tour__head').style.opacity = tourAlpha.toFixed(3);
    $('.tour__rail').style.opacity = tourAlpha.toFixed(3);

    // tag opacity targets
    const names = PARTS;
    const eT = top(sec.explode), eR = sec.explode.offsetHeight - H;
    const ep = (y - eT) / eR;
    const scanOn = clamp((0.8 - cur.cut) / 0.04) * clamp((cur.cut + 0.05) / 0.05);
    const explodeTags = clamp((cur.explode - 0.7) / 0.3) * (ep < 1.0 ? 1 : clamp(1 - (ep - 1.0) * 3)) * (1 - scanOn);
    names.forEach((n, i) => {
      let o = 0;
      if (inTour && i === idx) o = clamp(local / 0.22) * clamp((1 - local) / 0.22);
      o = Math.max(o, explodeTags);
      tagOpacity[n] = o;
    });

    // explode copy
    const ec = $('.explode__copy');
    ec.style.opacity = clamp((y - (eT - H * 0.2)) / (H * 0.3)) * clamp(1 - (ep - 0.88) / 0.1) + '';
    const sc = $('.explode__scan'); if (sc) { sc.style.opacity = scanOn.toFixed(3); }
    const ecp = $('.explode__copy .lead'); if (ecp) ecp.style.opacity = (1 - scanOn * 0.85).toFixed(3);

    // match clock
    const mT = top(sec.match), mR = sec.match.offsetHeight - H;
    const mp = clamp((y - mT) / mR);
    const t = clamp((mp - 0.06) / 0.86) * 158;
    const mm = Math.floor(t / 60), ss = Math.floor(t % 60);
    clockTime.textContent = `${mm}:${String(ss).padStart(2, '0')}`;
    segs[0].style.transform = `scaleX(${clamp(t / 30)})`;
    segs[1].style.transform = `scaleX(${clamp((t - 30) / 8)})`;
    segs[2].style.transform = `scaleX(${clamp((t - 38) / 120)})`;
    phases.forEach((p, i) => p.classList.toggle('is-active', (t < 30 ? 0 : t < 38 ? 1 : 2) === i));

    // process steps
    let best = -1, bd = 1e9;
    steps.forEach((s, i) => {
      const r = s.getBoundingClientRect(); const d = Math.abs(r.top + r.height / 2 - H * 0.52);
      if (d < bd) { bd = d; best = i; }
    });
    steps.forEach((s, i) => s.classList.toggle('is-active', i === best && bd < H * 0.5));

    // values
    valueItems.forEach((li) => li.classList.toggle('is-on', li.getBoundingClientRect().top < H * 0.72));

    // nav + rail
    const max = document.documentElement.scrollHeight - H;
    $('#railFill').style.transform = `scaleY(${clamp(y / max)})`;
    let current = 'hero';
    for (const [k, el] of Object.entries(sec)) { if (el && top(el) - H * 0.4 <= y) current = k; }
    $$('.nav__links a').forEach((a) => a.classList.toggle('is-active', a.getAttribute('href') === '#' + current));

    // HUD: current turntable angle
    $('#hudAngle').textContent = String(((Math.round(cur.ry) % 360) + 360) % 360).padStart(3, '0');
    const hud = $('.hud');
    if (hud) hud.style.opacity = clamp((y - H * 0.5) / (H * 0.3)).toFixed(3);
  }

  // callouts are positioned every frame from the 3D anchors
  function updateTags(cur) {
    const { w, h } = stage.size;
    for (const part of PARTS) {
      const t = tags[part];
      const o = tagOpacity[part] * clamp(cur.fade * 1.4);
      if (o < 0.01) { if (t.vis) { t.el.style.opacity = 0; t.vis = 0; } continue; }
      const a = stage.anchors[part]; if (!a) continue;
      const p = stage.project(a);
      const dir = p.x > w * 0.58 ? -1 : 1;
      t.el.dataset.dir = dir;
      t.el.style.setProperty('--len', (mobile ? 36 : 70) + 'px');
      t.el.style.opacity = o.toFixed(3);
      t.el.style.transform = `translate3d(${(dir === 1 ? p.x - 5.5 : p.x - 5.5 - 0)}px, ${p.y - 18}px, 0)`;
      if (dir === -1) t.el.style.transform = `translate3d(${p.x + 5.5}px, ${p.y - 18}px, 0) translateX(-100%)`;
      t.vis = 1;
    }
  }

  function setLang(l) { lang = l; }

  return { target, layout, update, updateTags, updateDims, buildDims, splitStatement, setLang, get mobile() { return mobile; } };
}
