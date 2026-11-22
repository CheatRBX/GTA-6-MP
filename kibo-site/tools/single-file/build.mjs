// Packs the whole site into ONE html file: styles, scripts, fonts, images and the robot model are all inside,
// so it opens straight from disk (double-click) or from any static host, with no other files next to it.
//
//   cd tools/single-file && npm install && node build.mjs            -> ../../dist/kibo-site.html
//   node build.mjs path/to/out.html
//
// The model is stored as base64 text (about a third bigger than the .glb) and decoded in the browser;
// desktops use kibo-robot.glb, phones kibo-robot-lite.glb, the same choice as the normal site.
import { build } from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.resolve(process.argv[2] || path.join(site, 'dist/kibo-site.html'));
const read = (p) => fs.readFileSync(path.join(site, p));
const MIME = { '.woff2': 'font/woff2', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml' };
const dataUri = (p) => `data:${MIME[path.extname(p)]};base64,${read(p).toString('base64')}`;

// text placed inside <script> must not end the element early
const inlineScript = (js, attrs = '') => {
  if (/<!--/.test(js)) throw new Error('script contains "<!--", which changes how HTML parses it');
  return `<script${attrs}>${js.replace(/<\/(script)/gi, '<\\/$1')}</script>`;
};

// one html edit that must match exactly once, so a change in index.html can't be skipped silently
const swap = (html, find, replace) => {
  const n = html.split(find).length - 1;
  if (n !== 1) throw new Error(`expected exactly one match for ${JSON.stringify(find)}, found ${n}`);
  return html.replace(find, () => replace);
};

// 1. the app: js/main.js and every module it imports (three.js + addons resolved like the importmap does)
const importmap = {
  name: 'importmap',
  setup(b) {
    b.onResolve({ filter: /^three$/ }, () => ({ path: path.join(site, 'vendor/three/build/three.module.js') }));
    b.onResolve({ filter: /^three\/addons\// }, (a) => ({ path: path.join(site, 'vendor/three/addons', a.path.slice('three/addons/'.length)) }));
  },
};
const bundle = await build({
  entryPoints: [path.join(site, 'js/main.js')], bundle: true, format: 'esm', target: 'es2022',
  minify: true, legalComments: 'inline', write: false, plugins: [importmap], logLevel: 'warning',
});
const app = bundle.outputFiles[0].text;

// 2. styles, with fonts and the poster image as data URIs
const css = [read('vendor/lenis/lenis.css').toString(), read('css/style.css').toString()].join('\n')
  .replace(/url\("\.\.\/(assets\/[^"]+)"\)/g, (m, p) => `url("${dataUri(p)}")`);
if (/url\("(?!data:)/.test(css)) throw new Error('a stylesheet still points to an outside file');

// 3. the robot model, decoded on demand by scene.js through window.KIBO_MODELS
const modelTag = (id, file) => `<script type="application/octet-stream" id="${id}">${read(file).toString('base64')}</script>`;
const modelSource = `
  // The robot model is inside this file (base64 at the end of the page) and is decoded only when needed.
  (function () {
    function source(id) {
      return function () {
        return new Promise(function (resolve) {
          var s = document.getElementById(id).textContent, bytes;
          if (Uint8Array.fromBase64) bytes = Uint8Array.fromBase64(s);
          else { var b = atob(s); bytes = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) bytes[i] = b.charCodeAt(i); }
          resolve(bytes.buffer);
        });
      };
    }
    window.KIBO_MODELS = { hi: source('kibo-model-hi'), lite: source('kibo-model-lite') };
  })();
`;

// 4. index.html with every outside reference replaced by its content
let html = read('index.html').toString();
html = swap(html, '<link rel="icon" href="assets/img/favicon.svg" type="image/svg+xml">',
  `<link rel="icon" href="${dataUri('assets/img/favicon.svg')}" type="image/svg+xml">`);
html = html.replace(/\s*<meta property="og:image"[^>]*>/, '');
html = html.replace(/\s*<link rel="preload"[^>]*>/g, '');
html = swap(html, '<link rel="stylesheet" href="vendor/lenis/lenis.css">', '');
html = swap(html, '<link rel="stylesheet" href="css/style.css">', `<style>\n${css}\n</style>`);
html = html.replace(/\s*<script type="importmap">[\s\S]*?<\/script>/, '');
for (const lib of ['vendor/gsap/gsap.min.js', 'vendor/gsap/ScrollTrigger.min.js', 'vendor/gsap/SplitText.min.js', 'vendor/lenis/lenis.min.js']) {
  html = swap(html, `<script src="${lib}"></script>`, inlineScript(read(lib).toString()));
}
html = swap(html, '<script type="module" src="js/main.js"></script>', inlineScript(modelSource) + '\n  ' + inlineScript(app, ' type="module"'));
html = swap(html, '</body>', [
  modelTag('kibo-model-hi', 'assets/models/kibo-robot.glb'),
  modelTag('kibo-model-lite', 'assets/models/kibo-robot-lite.glb'),
  '</body>'].join('\n'));

const markup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '');   // code may build links itself
const left = markup.match(/(?:src|href)="(?!data:|#|https?:|mailto:)[^"]+"/g);
if (left) throw new Error('outside references left: ' + left.join(', '));

fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, html);
console.log(`${path.relative(process.cwd(), out)}  ${(Buffer.byteLength(html) / 1e6).toFixed(1)} MB (app ${(app.length / 1e6).toFixed(2)} MB)`);
