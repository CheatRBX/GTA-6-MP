import { Stage, pickTier } from './scene.js';
import { createStory } from './story.js';
import { EN } from './i18n.js';
import { TEAM } from './config.js';

const { gsap, ScrollTrigger, SplitText } = window;
gsap.registerPlugin(ScrollTrigger, SplitText);
gsap.ticker.lagSmoothing(0);

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const STILL = params.has('still');                       // test mode: no continuous loop
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const html = document.documentElement;

// ---------------------------------------------------------------- team data from config.js
function applyTeam() {
  const num = TEAM.number ? ` #${TEAM.number}` : '';
  $('#teamTag').textContent = `FTC${num}`;
  $('#footMeta').textContent = [TEAM.name, 'FIRST Tech Challenge' + num, TEAM.location].filter(Boolean).join(' · ');
  const arrow = '<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M1 7h11M8 2.5 12.5 7 8 11.5"/></svg>';
  const cta = $('#joinCta');
  const labels = { instagram: 'Instagram', youtube: 'YouTube', telegram: 'Telegram', github: 'GitHub', website: 'Website' };
  const parts = [];
  if (TEAM.email) parts.push(`<a class="btn btn--primary mono" href="mailto:${TEAM.email}"><span data-i18n="join.email">Написать нам</span>${arrow}</a>`);
  for (const [k, url] of Object.entries(TEAM.links || {})) if (url) parts.push(`<a class="btn btn--ghost mono" href="${url}" target="_blank" rel="noopener noreferrer"><span>${labels[k] || k}</span>${arrow}</a>`);
  if (parts.length) { cta.innerHTML = parts.join(''); cta.hidden = false; }
}

// ---------------------------------------------------------------- language
const RU = new Map();
let lang = 'ru';
function captureRu() { $$('[data-i18n]').forEach((el) => RU.set(el, el.innerHTML)); }
function applyLang(l, story) {
  lang = l;
  html.lang = l; html.dataset.lang = l;
  $$('[data-i18n]').forEach((el) => {
    if (!RU.has(el)) RU.set(el, el.innerHTML);
    el.innerHTML = l === 'ru' ? RU.get(el) : (EN[el.dataset.i18n] ?? RU.get(el));
  });
  $$('.lang button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.lang === l)));
  document.title = l === 'ru' ? 'KIBO — команда FIRST Tech Challenge' : 'KIBO — FIRST Tech Challenge team';
  $('.sr-only').textContent = document.title;
  try { localStorage.setItem('kibo-lang', l); } catch (e) { /* private mode */ }
  story?.setLang(l);
  story?.splitStatement();
}

// ---------------------------------------------------------------- reveals
let revealCtx = null;
function buildReveals() {
  revealCtx?.revert();
  if (reduced) return;
  revealCtx = gsap.context(() => {
    $$('.h2').forEach((el) => {
      SplitText.create(el, {
        type: 'lines', mask: 'lines', autoSplit: true,
        onSplit(self) {
          return gsap.from(self.lines, { yPercent: 108, duration: 1.15, ease: 'expo.out', stagger: 0.09, scrollTrigger: { trigger: el, start: 'top 90%', once: true } });
        },
      });
    });
    $$('.label, .lead, .facts > div, .row, .specs__note, .values__note, .join__cta, .explode__copy .hint').forEach((el) => {
      gsap.from(el, { opacity: 0, y: 26, duration: 1, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 92%', once: true } });
    });
    // counters in the specs table
    $$('[data-count]').forEach((el) => {
      const end = +el.dataset.count; const o = { v: 0 };
      gsap.to(o, { v: end, duration: 1.8, ease: 'power2.out', scrollTrigger: { trigger: el, start: 'top 92%', once: true },
        onUpdate: () => { el.textContent = Math.round(o.v).toLocaleString(lang === 'ru' ? 'ru-RU' : 'en-US'); } });
    });
  });
}

function intro() {
  if (reduced) return;
  const tl = gsap.timeline({ defaults: { ease: 'expo.out' } });
  tl.from('.hero__word span', { yPercent: 36, opacity: 0, duration: 1.8, stagger: 0.09 }, 0)
    .from('.nav', { opacity: 0, y: -14, duration: 1.1 }, 0.3)
    .from('.eyebrow, .hero__tag, .hero__sub', { opacity: 0, y: 28, duration: 1.3, stagger: 0.1 }, 0.55)
    .from('.hero__scroll', { opacity: 0, duration: 1.2 }, 1.2);
}

// ---------------------------------------------------------------- boot
async function boot() {
  applyTeam();
  captureRu();

  const canvas = $('#stage');
  const loaderBar = $('#loaderBar'), loaderPct = $('#loaderPct'), loader = $('#loader');
  const setProgress = (p) => { loaderBar.style.transform = `scaleX(${p.toFixed(3)})`; loaderPct.textContent = Math.round(p * 100) + '%'; };

  let stage;
  try {
    stage = new Stage(canvas, pickTier());
    await stage.init((p) => setProgress(p * 0.92));
  } catch (err) {
    console.warn('WebGL stage unavailable, showing the page without 3D.', err);
    html.classList.add('no-webgl');
    loader.classList.add('is-done');
    applyLang(initialLang());
    buildReveals();
    return;
  }

  const story = createStory({ stage });
  // tags were created by the story: capture their Russian text too
  $$('.tag [data-i18n]').forEach((el) => RU.set(el, el.innerHTML));
  applyLang(initialLang(), story);
  story.buildDims();

  await (document.fonts?.ready ?? Promise.resolve());
  story.layout();
  const lenis = (!reduced && !STILL && window.Lenis) ? new window.Lenis({ lerp: 0.085, smoothWheel: true, wheelMultiplier: 0.95 }) : null;
  lenis?.on('scroll', ScrollTrigger.update);
  let y = scrollY;
  story.update(y, stage.cur); stage.snap(story.target);

  const pointer = { x: 0, y: 0 };
  addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    pointer.x = (e.clientX / innerWidth) * 2 - 1; pointer.y = (e.clientY / innerHeight) * 2 - 1;
  }, { passive: true });

  // render path (also used by the test hook)
  let last = performance.now();
  function frame(now, forcedDt) {
    const dt = forcedDt ?? Math.min(0.05, (now - last) / 1000); last = now;
    lenis?.raf(now);
    y = lenis ? lenis.scroll : scrollY;
    story.update(y, stage.cur);
    stage.update(dt, story.target, pointer);
    story.updateTags(stage.cur);
    story.updateDims(stage.cur);
    govern(dt);
  }

  // frame-time governor: lower the quality a step if the device cannot hold ~30 fps
  let acc = 0, n = 0, warm = 0, steps = 0;
  const forced = params.has('q');
  function govern(dt) {
    if (forced || STILL || document.hidden) return;
    if (warm++ < 120) return;
    acc += dt; n++;
    if (n >= 90) {
      if (acc / n > 1 / 28 && steps < 4 && stage.degrade()) { steps++; warm = 60; }
      acc = 0; n = 0;
    }
  }

  // reveal the page
  stage.resize();
  frame(performance.now(), 0.016);
  setProgress(1);
  await new Promise((r) => setTimeout(r, reduced ? 0 : 350));
  canvas.classList.add('is-ready');
  setTimeout(() => canvas.classList.add('is-live'), 1500);
  loader.classList.add('is-done');
  buildReveals();
  intro();

  if (!STILL) {
    const loop = (now) => { frame(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  // language switch
  $$('.lang button').forEach((b) => b.addEventListener('click', () => { applyLang(b.dataset.lang, story); buildReveals(); }));

  // in-page links
  $$('a[href^="#"]').forEach((a) => a.addEventListener('click', (e) => {
    const id = a.getAttribute('href');
    const el = id.length > 1 ? $(id) : null;
    if (!el) return;
    e.preventDefault();
    if (lenis) lenis.scrollTo(el, { duration: 1.8, easing: (t) => 1 - Math.pow(1 - t, 4) });
    else el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth' });
  }));

  // resize
  let rt;
  addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => { stage.resize(); story.layout(); ScrollTrigger.refresh(); }, 160);
  });
  addEventListener('load', () => { story.layout(); ScrollTrigger.refresh(); });

  // test / debug hook
  window.__kibo = {
    stage, story, lenis,
    renderAt(yy, extra) {
      scrollTo(0, yy);
      story.update(yy, stage.cur);
      if (extra) Object.assign(story.target, extra);
      stage.snap(story.target);
      for (let i = 0; i < 2; i++) frame(performance.now(), 0.016);
    },
  };
  window.__kiboReady = true;
}

function initialLang() {
  const q = params.get('lang');
  if (q === 'ru' || q === 'en') return q;
  let l = null;
  try { l = localStorage.getItem('kibo-lang'); } catch (e) { /* ignore */ }
  if (l === 'ru' || l === 'en') return l;
  return (navigator.language || 'ru').toLowerCase().startsWith('ru') ? 'ru' : 'en';
}

boot();
