// 3D stage: the real robot CAD on a studio turntable.
// Realism comes from physically based materials, a studio environment made of soft boxes,
// real shadows, screen-space ambient occlusion and a restrained neutral tone map. No bloom, no coloured lights.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const PARTS = ['chassis', 'intake', 'turret', 'lift', 'electronics'];

// The robot is split into ~18 sub-groups at build time (each node carries its centroid in extras);
// the exploded view pushes every group away from the centre, layer by layer.

// Points on the robot that callouts attach to: centres of real Fusion components (model space, metres)
export const ANCHORS = {
  chassis:     { at: [0.1, -0.181, 0.126], group: 'frame' },       // goBILDA Yellow Jacket drive motor (side seen at the station)
  intake:      { at: [-0.095, -0.111, -0.035], group: 'intake' },  // 90 mm tubing rollers
  turret:      { at: [0.003, 0.03, -0.004], group: 'shooter' },    // Non CR shooter (flywheel)
  lift:        { at: [0.17, 0.05, 0.003], group: 'rear' },         // Flower Box
  electronics: { at: [0.042, -0.06, 0.192], group: 'elecL' },      // REV Expansion Hub face (side seen at the station)
};

const TIERS = {
  high:   { model: 'hi',   ao: true,  aoSamples: 16, shadow: 2048, dpr: 2,   msaa: 4 },
  medium: { model: 'hi',   ao: true,  aoSamples: 8,  shadow: 1024, dpr: 1.5, msaa: 2 },
  low:    { model: 'lite', ao: false, aoSamples: 0,  shadow: 1024, dpr: 1.5, msaa: 4 },
};

const rad = (d) => (d * Math.PI) / 180;
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function pickTier() {
  const q = new URLSearchParams(location.search).get('q');
  if (q && TIERS[q]) return q;
  const coarse = matchMedia('(pointer: coarse)').matches || Math.min(innerWidth, innerHeight) < 600;
  if (coarse) return 'low';
  const mem = navigator.deviceMemory || 8;
  const cores = navigator.hardwareConcurrency || 8;
  return mem >= 8 && cores >= 8 ? 'high' : 'medium';
}

export class Stage {
  constructor(canvas, tierName) {
    this.canvas = canvas;
    this.tierName = tierName;
    this.tier = { ...TIERS[tierName] };
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: this.tier.msaa > 0, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.NeutralToneMapping;   // Khronos PBR Neutral: keeps real material colours
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.localClippingEnabled = true;        // for the section (cut-away) effect
    this.clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 10);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(27, 1, 0.05, 20);
    this.rig = new THREE.Group();          // turntable rotation
    this.scene.add(this.rig);

    // smoothed state (what is drawn) — S in main.js is the target
    this.cur = { ry: 125, zoom: 1.4, elev: 12, fov: 27, tx: 0, ty: 0.22, tz: 0, shiftX: 0, shiftY: 0, explode: 0, exposure: 1.2, dims: 0, fade: 1, plinth: 1, cut: 10 };
    this.smooth = 5.2;
    this.pointer = { x: 0, y: 0 };
    this.sx = 0; this.sy = 0;
    this.time = 0;
    this.anchors = {};
    this.regions = {};
    this.groups = [];
    this.modelBox = new THREE.Box3();
    this.size = { w: 1, h: 1 };
    this._v = new THREE.Vector3();
    this._t = new THREE.Vector3();
  }

  async init(onProgress) {
    this._buildEnvironment();
    this._buildLights();
    this._buildPlinth();
    await this._loadModel(this.tier.model, onProgress);
    this.resize();
    this._applyTier();
    await this.renderer.compileAsync(this.scene, this.camera).catch(() => {});
  }

  // ---------------------------------------------------------------- environment
  _buildEnvironment() {
    // A dark studio with a few soft boxes. Reflections on metal read as real photography,
    // and nothing is coloured: warm key, neutral strip, cool-neutral rim.
    const env = new THREE.Scene();
    const room = new THREE.Mesh(new THREE.BoxGeometry(12, 8, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.016, 0.0155, 0.015), side: THREE.BackSide }));
    env.add(room);
    const box = (w, h, pos, look, intensity, color) => {
      const m = new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), side: THREE.DoubleSide });
      const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), m);
      p.position.set(...pos); p.lookAt(...look); env.add(p);
    };
    box(4.6, 3.0, [-1.6, 3.7, 1.6], [0, 0, 0], 6.5, 0xfff0e0);    // key: big overhead softbox, warm white
    box(0.9, 5.2, [-5.2, 0.8, 0.4], [0, 0.4, 0], 2.0, 0xf2f2f0);  // left strip, neutral
    box(0.8, 5.0, [5.0, 1.0, -2.6], [0, 0.4, 0], 6.5, 0xe9eef5);  // right rim strip, a touch cool
    box(5.0, 2.2, [0.5, 1.2, -5.6], [0, 0.6, 0], 1.0, 0xfff6ea);  // back wall panel
    box(5.0, 2.6, [0.6, 1.4, 5.4], [0, 0.3, 0], 1.3, 0xfff4e8);   // soft front fill (behind the camera): lets flat dark plates show a gradient
    box(8.0, 6.0, [0, -3.6, 0], [0, 0, 0], 0.22, 0xdad5cc);        // floor bounce
    const pm = new THREE.PMREMGenerator(this.renderer);
    this.envMap = pm.fromScene(env, 0.025).texture;
    pm.dispose();
    env.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    this.scene.environment = this.envMap;
    this.scene.environmentIntensity = 1.2;
  }

  _buildLights() {
    const key = new THREE.DirectionalLight(0xfff1de, 2.8);
    key.position.set(-0.9, 2.2, 1.2);
    key.castShadow = true;
    const sc = key.shadow.camera;
    sc.left = -0.62; sc.right = 0.62; sc.top = 0.62; sc.bottom = -0.62; sc.near = 0.5; sc.far = 5;
    key.shadow.bias = -0.00035; key.shadow.normalBias = 0.0016; key.shadow.radius = 3.5;
    this.scene.add(key, key.target);
    this.key = key;

    const rim = new THREE.DirectionalLight(0xf3f5f8, 1.5);
    rim.position.set(1.6, 1.3, -2.0);
    this.scene.add(rim);
    const amb = new THREE.HemisphereLight(0xfff3e4, 0x2a2724, 0.9);   // soft shadow-side fill, keeps the yellow parts yellow
    this.scene.add(amb);
    const fill = new THREE.DirectionalLight(0xfff6ec, 0.35);
    fill.position.set(1.4, 0.5, 1.8);
    this.scene.add(fill);
  }

  _buildPlinth() {
    // A round machined plate: the robot's turntable, with an angle scale on its rim.
    const S = 2048, R = S / 2;
    const c = document.createElement('canvas'); c.width = c.height = S;
    const g = c.getContext('2d');
    const base = g.createRadialGradient(R, R, 0, R, R, R);
    base.addColorStop(0, '#222120'); base.addColorStop(0.72, '#191817'); base.addColorStop(1, '#121211');
    g.fillStyle = base; g.beginPath(); g.arc(R, R, R, 0, Math.PI * 2); g.fill();
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < 700; i++) { // turned (lathe) finish: fine concentric lines
      g.strokeStyle = `rgba(255,255,255,${(rnd() * 0.045).toFixed(3)})`; g.lineWidth = 0.6 + rnd() * 1.6;
      g.beginPath(); g.arc(R, R, rnd() * R * 0.93, 0, Math.PI * 2); g.stroke();
    }
    g.strokeStyle = 'rgba(236,232,225,0.12)'; g.lineWidth = 2;
    for (const r of [0.74, 0.86]) { g.beginPath(); g.arc(R, R, R * r, 0, Math.PI * 2); g.stroke(); }
    for (let d = 0; d < 360; d += 2) { // degree scale
      const a = rad(d - 90);
      const long = d % 30 === 0, mid = d % 10 === 0;
      const r1 = R * 0.965, r2 = r1 - R * (long ? 0.075 : mid ? 0.045 : 0.022);
      g.strokeStyle = long ? 'rgba(224,173,58,0.9)' : 'rgba(236,232,225,0.34)'; g.lineWidth = long ? 4 : 2;
      g.beginPath(); g.moveTo(R + Math.cos(a) * r1, R + Math.sin(a) * r1); g.lineTo(R + Math.cos(a) * r2, R + Math.sin(a) * r2); g.stroke();
      if (long) {
        g.save(); g.translate(R + Math.cos(a) * R * 0.875, R + Math.sin(a) * R * 0.875); g.rotate(a + Math.PI / 2);
        g.fillStyle = 'rgba(236,232,225,0.5)'; g.font = '500 30px "Plex Mono", monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(String(d), 0, 0); g.restore();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    const topMat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.46, metalness: 0.6, envMapIntensity: 0.8 });
    const sideMat = new THREE.MeshStandardMaterial({ color: 0x121211, roughness: 0.4, metalness: 0.8 });
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.37, 0.37, 0.03, 128, 1), [sideMat, topMat, sideMat]);
    disc.position.y = -0.0151; disc.receiveShadow = true;
    // CylinderGeometry cap UVs start at +X; rotate so the scale's 0° faces the viewer in the hero pose
    disc.rotation.y = rad(0);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(0.37, 0.0035, 12, 160), new THREE.MeshStandardMaterial({ color: 0x8d877d, roughness: 0.38, metalness: 1 }));
    rim.rotation.x = Math.PI / 2; rim.position.y = -0.0005;
    this.plinth = new THREE.Group(); this.plinth.add(disc, rim);
    // scan plane for the section effect: a faint amber sheet and a hairline ring, nothing glowing
    this.scan = new THREE.Group();
    const sheet = new THREE.Mesh(new THREE.CircleGeometry(0.64, 96), new THREE.MeshBasicMaterial({ color: 0xe0ad3a, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false, premultipliedAlpha: true }));
    sheet.rotation.x = -Math.PI / 2;
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.636, 0.64, 160), new THREE.MeshBasicMaterial({ color: 0xe0ad3a, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false, premultipliedAlpha: true }));
    ring.rotation.x = -Math.PI / 2;
    sheet.renderOrder = 6; ring.renderOrder = 6;
    this.scan.add(sheet, ring); this.scan.visible = false;
    this.scene.add(this.scan);
    this.scene.add(this.plinth);
  }

  // ---------------------------------------------------------------- model
  async _loadModel(which, onProgress) {
    // A page can override the sources: window.KIBO_MODELS = { hi, lite, plain }, each either a URL
    // or a function (onProgress) => Promise<ArrayBuffer> for hosts that can't serve .glb files.
    const cfg = window.KIBO_MODELS || {};
    const main = which === 'lite' ? (cfg.lite || 'assets/models/kibo-robot-lite.glb') : (cfg.hi || 'assets/models/kibo-robot.glb');
    // optional copy without meshopt compression, for where WebAssembly is missing or blocked by a CSP
    const plain = cfg.plain || 'assets/models/kibo-robot-plain.glb';
    const meshopt = MeshoptDecoder.supported !== false && await Promise.resolve(MeshoptDecoder.ready).then(() => true, () => false);
    const sources = meshopt ? [main, plain] : [plain];
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const progress = (e) => { if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total); };
    const load = (src) => (typeof src === 'function'
      ? src(onProgress).then((bytes) => loader.parseAsync(bytes, ''))
      : loader.loadAsync(src, progress));
    // if every source fails, report the first error: it is about the file that was actually wanted
    const tryLoad = (i) => load(sources[i]).catch((err) => {
      if (i + 1 < sources.length) return tryLoad(i + 1).catch(() => { throw err; });
      throw err;
    });
    this._mount((await tryLoad(0)).scene);
  }

  _mount(root) {
    if (this.model) this.rig.remove(this.model);
    const model = new THREE.Group();
    model.add(root);
    // centre the robot on the turntable, resting on the floor
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    const centre = box.getCenter(new THREE.Vector3());
    root.position.set(-centre.x, -box.min.y, -centre.z);
    this.shift = new THREE.Vector3(-centre.x, -box.min.y, -centre.z);
    this.modelBox = new THREE.Box3(box.min.clone().add(this.shift), box.max.clone().add(this.shift));
    this.model = model;
    this.rig.add(model);

    // exploded-view groups (see tools/model/build_robot.mjs): one per real Fusion sub-assembly,
    // each with its own path: lift first, then sideways, so parts never travel through each other
    this.groups = [];
    root.children.forEach((n) => {
      const x = n.userData;
      if (!x || !x.c) return;
      // NB: the node already carries a translation/scale (mesh quantisation); explode offsets are ADDED to it
      this.groups.push({ node: n, base: n.position.clone(), name: n.name, region: x.region, c: new THREE.Vector3(...x.c),
        dir: new THREE.Vector3(...(x.dir || [0, 0, 0])), lift: x.lift || 0, delay: x.delay || 0 });
    });
    const planes = [this.clip];
    const aniso = this.renderer.capabilities.getMaxAnisotropy();   // keeps the carbon weave crisp at grazing angles
    root.traverse((o) => {
      if (!o.isMesh) return;
      const m = o.material;
      o.frustumCulled = false;
      if (m.name && m.name.startsWith('decal_')) {           // stickers: a thin layer 0.12 mm above the part
        o.castShadow = false; o.receiveShadow = true; o.renderOrder = 3;
        m.transparent = true; m.depthWrite = false; m.premultipliedAlpha = true;
        m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -4;
      } else if (m.transparent) {                             // clear polycarbonate
        o.castShadow = false; o.receiveShadow = false; o.renderOrder = 5;
        m.depthWrite = false; m.opacity = 0.12; m.color.multiplyScalar(0.75); m.envMapIntensity = 0.45; m.premultipliedAlpha = true;
      } else {
        o.castShadow = true; o.receiveShadow = true; m.envMapIntensity = m.metalness > 0.5 ? 0.62 : 1.0;
        m.shadowSide = THREE.FrontSide;
      }
      for (const t of [m.map, m.normalMap]) if (t) { t.anisotropy = aniso; t.needsUpdate = true; }
      m.clippingPlanes = planes; m.clipShadows = true;
    });

    // callout anchors live inside their sub-assembly so they follow it when the robot comes apart
    this.anchors = {}; this.anchorBase = {};
    for (const name of PARTS) {
      const { at, group } = ANCHORS[name];
      const pos = new THREE.Vector3(...at);
      let holder = this.groups.find((g) => g.name === group);
      if (!holder) { let bd = 1e9; for (const g of this.groups) { const d = g.c.distanceToSquared(pos); if (d < bd) { bd = d; holder = g; } } }
      const a = new THREE.Object3D();
      const parent = holder ? holder.node : root;
      root.updateMatrixWorld(true);
      a.position.copy(parent.worldToLocal(root.localToWorld(pos.clone())));
      parent.add(a);
      this.anchors[name] = a;
      this.anchorBase[name] = pos.clone();          // model-space position, used by the camera key frames
    }
    this.dimAnchors = this._buildDimAnchors(root);
  }

  // six reference points per dimension: two on the model, two on the dimension line, plus mid label
  _buildDimAnchors(root) {
    const b = this.modelBox;
    const mn = b.min.clone().sub(this.shift), mx = b.max.clone().sub(this.shift); // back to root-local
    const off = 0.065;
    const mk = (p) => { const o = new THREE.Object3D(); o.position.copy(p); root.add(o); return o; };
    const dims = [];
    // length (x) along the floor edge on the +z side
    dims.push({ label: 'L', value: (mx.x - mn.x) * 1000,
      pts: [mk(new THREE.Vector3(mn.x, mn.y, mx.z)), mk(new THREE.Vector3(mx.x, mn.y, mx.z)), mk(new THREE.Vector3(mn.x, mn.y, mx.z + off)), mk(new THREE.Vector3(mx.x, mn.y, mx.z + off))] });
    // height (y) on the back-left vertical edge
    dims.push({ label: 'H', value: (mx.y - mn.y) * 1000,
      pts: [mk(new THREE.Vector3(mx.x, mn.y, mn.z)), mk(new THREE.Vector3(mx.x, mx.y, mn.z)), mk(new THREE.Vector3(mx.x + off, mn.y, mn.z - off * 0.0)), mk(new THREE.Vector3(mx.x + off, mx.y, mn.z - off * 0.0))] });
    // width (z) along the floor edge on the front (-x) side
    dims.push({ label: 'W', value: (mx.z - mn.z) * 1000,
      pts: [mk(new THREE.Vector3(mn.x, mn.y, mn.z)), mk(new THREE.Vector3(mn.x, mn.y, mx.z)), mk(new THREE.Vector3(mn.x - off, mn.y, mn.z)), mk(new THREE.Vector3(mn.x - off, mn.y, mx.z))] });
    return dims;
  }

  // ---------------------------------------------------------------- quality
  _applyTier() {
    const t = this.tier;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, t.dpr));
    this.renderer.shadowMap.enabled = t.shadow > 0;
    if (t.shadow > 0) {
      if (this.key.shadow.mapSize.x !== t.shadow) { this.key.shadow.mapSize.set(t.shadow, t.shadow); this.key.shadow.map?.dispose(); this.key.shadow.map = null; }
    }
    if (t.ao) this._buildComposer(); else this._disposeComposer();
    this._setSize();
  }

  _buildComposer() {
    if (this.composer) { this.gtao.updateGtaoMaterial({ samples: this.tier.aoSamples }); return; }
    const { w, h } = this.size;
    const rt = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: this.tier.msaa || 0 });
    this.composer = new EffectComposer(this.renderer, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.gtao = new GTAOPass(this.scene, this.camera, w, h);
    this.gtao.output = GTAOPass.OUTPUT.Default;
    this.gtao.blendIntensity = 1.0;
    this.gtao.updateGtaoMaterial({ radius: 0.05, distanceExponent: 1.4, thickness: 1.2, scale: 1.1, samples: this.tier.aoSamples, distanceFallOff: 1, screenSpaceRadius: false });
    this.gtao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, rings: 2, samples: 12 });
    this.composer.addPass(this.gtao);
    this.composer.addPass(new OutputPass());
  }

  _disposeComposer() {
    if (!this.composer) return;
    this.composer.dispose(); this.gtao?.dispose?.();
    this.composer = null; this.gtao = null;
  }

  // Called by the frame-time governor: step down one level when the device cannot keep up.
  degrade() {
    const t = this.tier;
    if (t.ao && t.aoSamples > 8) { t.aoSamples = 8; t.dpr = Math.min(t.dpr, 1.5); }
    else if (t.ao) { t.ao = false; }
    else if (t.dpr > 1) { t.dpr = 1; }
    else if (t.shadow > 0) { t.shadow = 0; }
    else return false;
    this._applyTier();
    return true;
  }

  // ---------------------------------------------------------------- frame
  resize() { this._setSize(); }

  _setSize() {
    const w = Math.max(1, this.canvas.clientWidth || innerWidth), h = Math.max(1, this.canvas.clientHeight || innerHeight);
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.composer?.setPixelRatio(this.renderer.getPixelRatio());
    this.composer?.setSize(w, h);
  }

  snap(S) { Object.assign(this.cur, S); }

  // world -> screen (CSS px), for callouts
  project(obj) {
    const v = this._v;
    obj.getWorldPosition(v);
    v.project(this.camera);
    return { x: (v.x * 0.5 + 0.5) * this.size.w, y: (-v.y * 0.5 + 0.5) * this.size.h, z: v.z };
  }

  update(dt, S, pointer) {
    this.time += dt;
    const c = this.cur;
    const k = this.reduced ? 1 : 1 - Math.exp(-dt * this.smooth);
    for (const key in S) c[key] += (S[key] - c[key]) * k;
    const pk = 1 - Math.exp(-dt * 3);
    this.sx += (pointer.x - this.sx) * pk;
    this.sy += (pointer.y - this.sy) * pk;

    const idle = this.reduced ? 0 : this.time;
    // turntable: scroll angle + a slow breathing sway so the robot is never frozen
    this.rig.rotation.y = rad(c.ry) + Math.sin(idle * 0.31) * 0.035 + this.sx * 0.12;
    this.plinth.visible = c.plinth > 0.01;
    this.plinth.position.y = 0;
    this.renderer.toneMappingExposure = c.exposure;
    this.canvas.style.setProperty('--fade', c.fade.toFixed(3));

    // exploded view: lift first, then sideways; groups are staggered (shooter first, chassis never moves)
    for (const g of this.groups) {
      const p = clamp01((c.explode - g.delay) / (1 - g.delay));
      const pl = easeInOut(clamp01(p / 0.5));
      const pd = easeInOut(clamp01((p - 0.3) / 0.7));
      g.node.position.set(g.base.x + g.dir.x * pd, g.base.y + g.lift * pl + g.dir.y * pd, g.base.z + g.dir.z * pd);
    }
    // section view: a horizontal plane that slices the robot open
    this.clip.constant = c.cut;
    const showScan = c.cut < 0.8 && c.cut > -0.05;
    this.scan.visible = showScan;
    if (showScan) this.scan.position.y = Math.max(c.cut, 0);
    this.rig.position.y = 0.0;

    // camera: aims at a point local to the robot, keeps the robot fitted regardless of aspect ratio
    const { w, h } = this.size;
    const fov = rad(c.fov);
    const hf = 2 * Math.atan(Math.tan(fov / 2) * (w / h));
    const fit = 0.36 / Math.sin(Math.min(fov, hf) / 2);
    const dist = fit * c.zoom;
    this._t.set(c.tx, c.ty, c.tz);
    this.rig.updateMatrixWorld(true);
    this.rig.localToWorld(this._t);
    const el = rad(c.elev) + this.sy * 0.025;
    const az = this.sx * 0.05;
    this.camera.position.set(this._t.x + Math.sin(az) * Math.cos(el) * dist, this._t.y + Math.sin(el) * dist, this._t.z + Math.cos(az) * Math.cos(el) * dist);
    this.camera.fov = c.fov;
    this.camera.lookAt(this._t);
    this.camera.setViewOffset(w, h, -c.shiftX * w, c.shiftY * h, w, h);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);

    // light follows the turntable's feel: a hair of movement keeps highlights alive on metal
    this.key.position.set(-0.9 + Math.sin(idle * 0.2) * 0.08, 2.2, 1.2);
    this.key.target.position.set(0, 0.2, 0);

    if (this.composer) this.composer.render(dt); else this.renderer.render(this.scene, this.camera);
  }
}
