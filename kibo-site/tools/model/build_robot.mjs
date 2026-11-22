// SVF -> web GLB, with the real Fusion (Prism) materials, decals and carbon/brush textures.
//   node build_robot.mjs <svfDir> <out.glb> <profile: hi|lite>
import { createRequire } from 'node:module';
import fs from 'node:fs';
import zlib from 'node:zlib';
import path from 'node:path';
import sharp from 'sharp';
import { Document, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRMaterialsClearcoat, KHRMaterialsEmissiveStrength } from '@gltf-transform/extensions';
import { prune } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';

const require = createRequire(import.meta.url);
const { SvfReader } = require('forge-convert-utils');

const [,, svfDir, outPath, profile = 'hi'] = process.argv;
// Fidelity first: every part is kept, and each mesh is simplified only down to an ABSOLUTE deviation
// (0.25 mm on desktop, 0.6 mm on phones), which is below one screen pixel even with the robot full-screen.
// Desktop uses no 'Permissive', so hard edges (split normals) stay crisp.
const P = profile === 'lite'
  ? { drop: 3, errMM: 0.8, nw: 0.2, flags: ['ErrorAbsolute', 'Permissive'], texMax: 512 }   // phones: 1 px is ~1.3 mm
  : { drop: 0, errMM: 0.25, nw: 0.1, flags: ['ErrorAbsolute'], texMax: 1024 };
if (process.env.SIMP) Object.assign(P, JSON.parse(process.env.SIMP));
const FLIP_V = process.env.FLIP_V !== '0';
const DECAL_MODE = +(process.env.DECAL_MODE ?? 1);

await MeshoptSimplifier.ready;

// ------------------------------------------------------------------ read SVF
const reader = await SvfReader.FromFileSystem(path.join(svfDir, 'Design.svf'));
const scene = await reader.read({ skipPropertyDb: false });   // the property DB holds the Fusion component tree
const meta = scene.getMetadata();
const gunzip = (f) => JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(svfDir, f))).toString());
const protein = gunzip('ProteinMaterials.json.gz').materials;
const phong = gunzip('Materials.json.gz').materials;   // surface-finish textures (carbon twill, brushed, bead-blast) are described here

// world transform (what forge-convert's glTF writer does): SVF units -> metres, Y up, model centred
const upV = meta['world up vector'].XYZ, frV = meta['world front vector'].XYZ;
const left = [upV[1] * frV[2] - upV[2] * frV[1], upV[2] * frV[0] - upV[0] * frV[2], upV[0] * frV[1] - upV[1] * frV[0]];
const unit = { centimeter: 0.01, cm: 0.01, millimeter: 0.001, mm: 0.001, foot: 0.3048, ft: 0.3048, inch: 0.0254, in: 0.0254 }[meta['distance unit'].value] ?? 1;
const bmin = meta['world bounding box'].minXYZ, bmax = meta['world bounding box'].maxXYZ;
const cen = [-(bmin[0] + bmax[0]) / 2, -(bmin[1] + bmax[1]) / 2, -(bmin[2] + bmax[2]) / 2];
// column-major 4x4 helpers
const mul = (a, b) => { const o = new Array(16).fill(0); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; };
const ROOT = [left[0] * unit, up(0) * unit, frV[0] * unit, 0, left[1] * unit, up(1) * unit, frV[1] * unit, 0, left[2] * unit, up(2) * unit, frV[2] * unit, 0, 0, 0, 0, 1];
function up(i) { return upV[i]; }
const CENTER = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, cen[0], cen[1], cen[2], 1];
const WORLD = mul(ROOT, CENTER);
function localMatrix(t) {
  if (!t) return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (t.kind === 0) return t.elements.slice();
  const q = t.rotation || { x: 0, y: 0, z: 0, w: 1 }, s = t.scale || { x: 1, y: 1, z: 1 }, p = t.translation || { x: 0, y: 0, z: 0 };
  const { x, y, z, w } = q;
  const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  return [(1 - 2 * (yy + zz)) * s.x, 2 * (xy + wz) * s.x, 2 * (xz - wy) * s.x, 0,
    2 * (xy - wz) * s.y, (1 - 2 * (xx + zz)) * s.y, 2 * (yz + wx) * s.y, 0,
    2 * (xz + wy) * s.z, 2 * (yz - wx) * s.z, (1 - 2 * (xx + yy)) * s.z, 0, p.x, p.y, p.z, 1];
}
const xf = (m, x, y, z) => [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
const det3 = (m) => m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);

// ------------------------------------------------------------------ materials (Prism -> glTF PBR)
const s2l = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const val = (p, kind, name, d) => p.properties?.[kind]?.[name]?.values?.[0] ?? d;
const col = (p, name, d = [0.5, 0.5, 0.5]) => { const c = p.properties?.colors?.[name]?.values?.[0]; return c ? [c.r, c.g, c.b] : d; };
const texOf = (group, mat, slot) => {
  const c = mat.textures?.[slot]?.connections?.[0]; if (!c) return null;
  const t = group.materials[c]; const uri = t?.properties?.uris?.unifiedbitmap_Bitmap?.values?.[0]; if (!uri) return null;
  const sc = t.properties.scalars || {};
  return { uri, u: sc.texture_UScale?.values?.[0] ?? 1, v: sc.texture_VScale?.values?.[0] ?? 1, ang: sc.texture_WAngle?.values?.[0] ?? 0 };
};
const matInfo = new Map();
function infoOf(id) {
  if (matInfo.has(id)) return matInfo.get(id);
  const g = protein[String(id)]; const m = g?.materials?.[g.userassets?.[0]];
  const o = { id, kind: 'opaque', base: [0.2, 0.2, 0.2], alpha: 1, metal: 0, rough: 0.5, emissive: null, clearcoat: 0, decals: [], diffuseTex: null, normalTex: null };
  if (m) {
    const def = m.definition;
    const rough = val(m, 'scalars', 'surface_roughness', 0.4);
    o.rough = rough;
    if (def === 'PrismMetal') { o.kind = 'metal'; o.metal = 1; o.base = col(m, 'metal_f0', [0.8, 0.8, 0.8]); o.rough = Math.max(rough, 0.16); }
    else if (def === 'PrismTransparent') { o.kind = 'clear'; o.base = col(m, 'transparent_color', [0.95, 0.95, 0.95]); o.alpha = 0.16; o.rough = 0.05; }
    else if (def === 'PrismLayered') { o.kind = 'carbon'; o.base = col(m, 'layered_diffuse', [0.1, 0.1, 0.1]); o.clearcoat = 0.7; o.rough = 0.42; }
    else if (def === 'PrismOpaque') { o.base = col(m, 'opaque_albedo'); o.rough = Math.max(rough, 0.28); if (/Emissive/.test(m.categories?.[0] || '')) o.emissive = o.base; }
    else { const d = col(m, 'generic_diffuse', [0.3, 0.3, 0.3]); o.base = d; }
    const pg = phong[String(id)]; const pm = pg?.materials?.[pg.userassets?.[0]];
    if (pm) {
      const dt = texOf(pg, pm, 'generic_diffuse'); if (dt) o.diffuseTex = dt;
      const bt = texOf(pg, pm, 'generic_bump'); if (bt) o.normalTex = bt;
    }
  }
  for (const d of g?.decals || []) {
    const dm = d.material; const dmat = dm?.materials?.[dm.userassets?.[0]];
    const uri = dmat?.textures?.generic_diffuse ? dm.materials[dmat.textures.generic_diffuse.connections[0]]?.properties?.uris?.unifiedbitmap_Bitmap?.values?.[0] : null;
    if (uri) o.decals.push({ uri, uv: d.uv ?? 1 });
  }
  matInfo.set(id, o);
  return o;
}

// ------------------------------------------------------------------ collect fragments
const frags = [];
for (let i = 0, n = scene.getNodeCount(); i < n; i++) {
  const node = scene.getNode(i);
  if (node.kind !== 1) continue;
  const geo = scene.getGeometry(node.geometry);
  if (!geo || geo.kind !== 0) continue;
  const M = mul(WORLD, localMatrix(node.transform));
  const V = geo.getVertices();
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (let k = 0; k < V.length; k += 3) { const q = xf(M, V[k], V[k + 1], V[k + 2]); for (let a = 0; a < 3; a++) { if (q[a] < mn[a]) mn[a] = q[a]; if (q[a] > mx[a]) mx[a] = q[a]; } }
  frags.push({ i, gid: node.geometry, mid: node.material, geo, M, mn, mx, diag: Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) * 1000, c: [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2] });
}
console.log('fragments', frags.length);
const kept = frags.filter((f) => f.diag >= P.drop || infoOf(f.mid).decals.length);
console.log('kept', kept.length, 'dropped', frags.length - kept.length);

// model bounds -> floor at y = 0 later in the site; here just centre-based
let gmn = [1e9, 1e9, 1e9], gmx = [-1e9, -1e9, -1e9];
for (const f of kept) for (let a = 0; a < 3; a++) { gmn[a] = Math.min(gmn[a], f.mn[a]); gmx[a] = Math.max(gmx[a], f.mx[a]); }
const mid = [(gmn[0] + gmx[0]) / 2, (gmn[1] + gmx[1]) / 2, (gmn[2] + gmx[2]) / 2];

// ------------------------------------------------------------------ simplify once per geometry
const gDiag = new Map(), gScale = new Map();
for (const f of kept) {
  gDiag.set(f.gid, Math.max(gDiag.get(f.gid) || 0, f.diag));
  const sc = Math.hypot(f.M[0], f.M[1], f.M[2]);                 // metres per local unit for this instance
  gScale.set(f.gid, Math.max(gScale.get(f.gid) || 0, sc));
}
const simp = new Map();
let trisBefore = 0, trisAfter = 0;
const decalGeo = new Set(kept.filter((f) => infoOf(f.mid).decals.length).map((f) => f.gid));
for (const [gid] of gDiag) {
  const geo = scene.getGeometry(gid);
  const idx = Uint32Array.from(geo.getIndices());
  const pos = geo.getVertices(); const nrm = geo.getNormals();
  trisBefore += idx.length / 3;
  let out = idx;
  if (!decalGeo.has(gid) && idx.length > 36) {
    const err = (P.errMM / 1000) / gScale.get(gid);               // same physical tolerance for every part
    out = nrm ? MeshoptSimplifier.simplifyWithAttributes(idx, pos, 3, nrm, 3, [P.nw, P.nw, P.nw], null, 12, err, P.flags)[0]
              : MeshoptSimplifier.simplify(idx, pos, 3, 12, err, P.flags)[0];
    if (out.length < 12) out = idx;                               // never let a part vanish
  }
  trisAfter += out.length / 3;
  simp.set(gid, out);
}
console.log('unique tris', trisBefore, '->', trisAfter);

// ------------------------------------------------------------------ exploded-view groups from the real Fusion components
// Parts are never split: every top-level Fusion component moves as one piece, and each group has a path
// (lift first, then sideways) chosen so that nothing travels through a neighbouring part.
const pdb = scene.svf.properties;
const pname = (id) => { try { const p = pdb.getProperties(id); return p.name || p.Name || ''; } catch (e) { return ''; } };
const pkids = (id) => { try { return pdb.getChildren(id); } catch (e) { return []; } };
const topOf = new Map();
const rootKids = pkids(1); const designRoot = rootKids.length === 1 ? rootKids[0] : 1;
const markTop = (id, t) => { topOf.set(id, t); for (const k of pkids(id)) markTop(k, t); };
for (const t of pkids(designRoot)) markTop(t, t);
const fragList = scene.svf.fragments;
const compMap = new Map();
for (const f of kept) {
  const top = topOf.get(fragList[f.i].dbID) ?? -1;
  let c = compMap.get(top);
  if (!c) { c = { id: top, name: pname(top).replace(/(:\d+)+$/, ''), frags: [], mn: [1e9, 1e9, 1e9], mx: [-1e9, -1e9, -1e9] }; compMap.set(top, c); }
  c.frags.push(f);
  for (let a = 0; a < 3; a++) { c.mn[a] = Math.min(c.mn[a], f.mn[a] - mid[a]); c.mx[a] = Math.max(c.mx[a], f.mx[a] - mid[a]); }
}
// Order matters: upper units leave first and travel further, so nothing has to pass through a unit above it.
const GROUPS = {
  frame:    { region: 'chassis',     dir: [0, 0, 0],        lift: 0,    delay: 0.0 },
  shooter:  { region: 'turret',      dir: [0, 0, 0],        lift: 0.34, delay: 0.0 },
  camera:   { region: 'turret',      dir: [-0.09, 0, 0],    lift: 0.26, delay: 0.06 },
  turret:   { region: 'turret',      dir: [0, 0, 0],        lift: 0.18, delay: 0.12 },
  rear:     { region: 'lift',        dir: [0.2, 0, 0],      lift: 0.08, delay: 0.14 },
  elecR:    { region: 'electronics', dir: [0, 0, -0.18],    lift: 0.12, delay: 0.2 },
  elecL:    { region: 'electronics', dir: [0, 0, 0.18],     lift: 0.12, delay: 0.2 },
  intake:   { region: 'intake',      dir: [-0.2, 0.02, 0],  lift: 0,    delay: 0.24 },
  transfer: { region: 'intake',      dir: [0, 0, 0],        lift: 0.1,  delay: 0.32 },
};
const RULES = [
  [/shooter/i, 'shooter'],
  [/turret|rot gear/i, 'turret'],
  [/limelight|camera/i, 'camera'],
  [/control hub|11-1855|servo hub plate/i, 'elecR'],
  [/expansion\+?hub|battery/i, 'elecL'],
  [/hub mount/i, (c) => (c.mx[2] + c.mn[2] < 0 ? 'elecR' : 'elecL')],
  [/flower box|new pc flower|spool|boxtube/i, 'rear'],
  [/transfer|ramp mgn/i, 'transfer'],
  [/intake|flower arm|tubing roller|inner arms/i, 'intake'],
  [/140mm cf rod/i, 'camera'],                         // camera mast: the upper ends sit in the Limelight beam's connectors
  [/86mm cf rod/i, 'rear'],                            // brace at the front edge of the rear block
  [/cf rod|cf tube/i, (c) => ((c.mn[1] < -0.12 && c.mx[0] - c.mn[0] > 0.2) || c.mn[1] < -0.19 ? 'frame' : null)],  // chassis rails
  [/yellowjacket|dead wheel|\bodo|pinpoint|outer|inner|component12|plate|backplate|^wedge$|^motors$|corner node|expansion node/i, 'frame'],
];
const named = [], unnamed = [];
for (const c of compMap.values()) {
  for (const [re, g] of RULES) { if (!re.test(c.name)) continue; const v = typeof g === 'function' ? g(c) : g; if (v) { c.group = v; break; } }
  (c.group ? named : unnamed).push(c);
}
// screws, bearings, pulleys, shafts, belts: follow the nearest named component (box-to-box distance)
const boxDist = (a, b) => { let d = 0; for (let k = 0; k < 3; k++) { const gap = Math.max(0, a.mn[k] - b.mx[k], b.mn[k] - a.mx[k]); d += gap * gap; } return d; };
for (const c of unnamed) {
  let best = null, bd = 1e9;
  for (const n of named) { const d = boxDist(c, n); if (d < bd || (d === bd && best && (n.frags.length > best.frags.length))) { bd = d; best = n; } }
  c.group = best ? best.group : 'frame';
}
const groups = Object.entries(GROUPS).map(([name, g]) => ({ name, ...g, items: [], centroid: [0, 0, 0] }));
const gByName = Object.fromEntries(groups.map((g) => [g.name, g]));
for (const c of compMap.values()) for (const f of c.frags) { f.group = gByName[c.group]; f.group.items.push(f); }
for (const g of groups) {
  if (!g.items.length) continue;
  for (const f of g.items) for (let a = 0; a < 3; a++) g.centroid[a] += f.c[a] / g.items.length;
}
console.log('groups', groups.map((g) => `${g.name}:${g.items.length}`).join(' '));
if (process.env.DEBUG_GROUPS) for (const c of compMap.values()) console.log('  ', c.group.padEnd(8), c.name);

// ------------------------------------------------------------------ build document
const doc = new Document();
const buffer = doc.createBuffer();
const ccExt = doc.createExtension(KHRMaterialsClearcoat), emExt = doc.createExtension(KHRMaterialsEmissiveStrength);
const sceneG = doc.createScene('KIBO');
const texCache = new Map();
async function makeTexture(key, make, name) {
  if (texCache.has(key)) return texCache.get(key);
  const buf = await make();
  const t = doc.createTexture(name).setImage(buf).setMimeType('image/png');
  texCache.set(key, t); return t;
}
const imageBuf = (uri) => scene.getImage(uri) || scene.getImage(uri.replace(/\\/g, '/'));

const gltfMats = new Map(); // key -> {mat, texUV:{set, su, sv, ang} | null}
async function materialFor(f, wantDecal) {
  const info = infoOf(f.mid);
  const useDecal = wantDecal && info.decals.length && f.geo.getUvChannelCount() > info.decals[0].uv;
  const key = `${f.mid}|${useDecal ? 'd' : 'p'}`;
  if (gltfMats.has(key)) return gltfMats.get(key);
  const m = doc.createMaterial(`prism_${f.mid}${useDecal ? '_decal' : ''}`);
  const lin = info.base.map(s2l);
  const floor = info.metal ? 0.07 : 0.02; // real black plastic / anodising still reflects a few percent
  const mx = Math.max(...lin); const k = mx < floor ? (mx > 1e-5 ? floor / mx : 0) : 1;
  const base = k ? lin.map((c) => c * k) : [floor, floor, floor];
  m.setBaseColorFactor([base[0], base[1], base[2], info.alpha]).setMetallicFactor(info.metal).setRoughnessFactor(info.rough);
  m.setDoubleSided(true);
  if (info.kind === 'clear') m.setAlphaMode('BLEND');
  if (info.emissive) { m.setEmissiveFactor(info.emissive.map(s2l)); m.setExtension('KHR_materials_emissive_strength', emExt.createEmissiveStrength().setEmissiveStrength(1.6)); }
  if (info.clearcoat) m.setExtension('KHR_materials_clearcoat', ccExt.createClearcoat().setClearcoatFactor(info.clearcoat).setClearcoatRoughnessFactor(0.06));
  let tex = null;
  const resize = async (buf, w) => sharp(buf, { limitInputPixels: false }).resize({ width: w, height: w, fit: 'inside', withoutEnlargement: true }).png().toBuffer();
  if (useDecal) {
    // base colour with the decal image composited on top; flat outside the decal's UV area (clamped sampling)
    const d = info.decals[0]; const src = imageBuf(d.uri);
    if (src) {
      const dec = await sharp(src, { limitInputPixels: false }).resize({ width: P.texMax, height: P.texMax, fit: 'inside', withoutEnlargement: true }).ensureAlpha().png().toBuffer();
      const md = await sharp(dec).metadata();
      const bg = info.base.map((c) => Math.round(Math.min(1, c) * 255));
      const comp = await sharp({ create: { width: md.width, height: md.height, channels: 3, background: { r: bg[0], g: bg[1], b: bg[2] } } }).composite([{ input: dec }]).png().toBuffer();
      const t = await makeTexture(`decal:${f.mid}:${d.uri}`, async () => comp, `decal_${d.uri.slice(0, 8)}`);
      m.setBaseColorTexture(t); m.setBaseColorFactor([1, 1, 1, 1]);
      m.getBaseColorTextureInfo().setWrapS(33071).setWrapT(33071);
      tex = { set: d.uv, su: 1, sv: 1, ang: 0 };
      m.setMetallicFactor(0); m.setRoughnessFactor(Math.max(info.rough, 0.55));
    }
  } else if (info.diffuseTex || info.normalTex) {
    const dt = info.diffuseTex, nt = info.normalTex;
    if (dt && imageBuf(dt.uri)) {
      const t = await makeTexture('d:' + dt.uri, async () => resize(imageBuf(dt.uri), P.texMax), 'tex_' + dt.uri.slice(-24));
      m.setBaseColorTexture(t); tex = { set: 0, su: dt.u, sv: dt.v, ang: dt.ang };
    }
    // surface-finish (bump) maps: carbon twill and brushed metal are the visible ones
    if (nt && imageBuf(nt.uri) && /carbon|brush/.test(nt.uri)) {
      const t = await makeTexture('n:' + nt.uri, async () => {
        const isNormal = /norm/.test(nt.uri);
        const buf = await resize(imageBuf(nt.uri), Math.min(P.texMax, 512));
        if (isNormal) return buf;
        // height map -> tangent space normal map
        const { data, info: im } = await sharp(buf).greyscale().raw().toBuffer({ resolveWithObject: true });
        const W = im.width, H = im.height, out = Buffer.alloc(W * H * 3), s = 2.2;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const g = (xx, yy) => data[((yy + H) % H) * W + ((xx + W) % W)] / 255;
          const dx = (g(x + 1, y) - g(x - 1, y)) * s, dy = (g(x, y + 1) - g(x, y - 1)) * s;
          const l = Math.hypot(dx, dy, 1), o = (y * W + x) * 3;
          out[o] = Math.round((-dx / l * 0.5 + 0.5) * 255); out[o + 1] = Math.round((-dy / l * 0.5 + 0.5) * 255); out[o + 2] = Math.round((1 / l * 0.5 + 0.5) * 255);
        }
        return sharp(out, { raw: { width: W, height: H, channels: 3 } }).png().toBuffer();
      }, 'nrm_' + nt.uri.slice(-24));
      m.setNormalTexture(t); m.setNormalScale(0.6); tex = tex || { set: 0, su: nt.u, sv: nt.v, ang: nt.ang };
    }
  }
  const out = { mat: m, tex };
  gltfMats.set(key, out);
  return out;
}

// bake world transform + merge into (group, material) buckets
const buckets = new Map();
let instTris = 0;
for (const f of kept) {
  const info = infoOf(f.mid);
  const { mat, tex } = await materialFor(f, false);
  const idx = simp.get(f.gid); if (!idx) continue;
  const key = `${f.group.name}|${mat.getName()}`;
  let b = buckets.get(key);
  if (!b) { b = { group: f.group, mat, tex, pos: [], nrm: [], uv: [], idx: [], v: 0 }; buckets.set(key, b); }
  const P0 = f.geo.getVertices(), N0 = f.geo.getNormals();
  const uvSet = tex ? Math.min(tex.set, f.geo.getUvChannelCount() - 1) : -1;
  const U0 = uvSet >= 0 ? f.geo.getUvs(uvSet) : null;
  const M = f.M, flip = det3(M) < 0;
  const c00 = M[5] * M[10] - M[6] * M[9], c01 = M[6] * M[8] - M[4] * M[10], c02 = M[4] * M[9] - M[5] * M[8];
  const c10 = M[2] * M[9] - M[1] * M[10], c11 = M[0] * M[10] - M[2] * M[8], c12 = M[1] * M[8] - M[0] * M[9];
  const c20 = M[1] * M[6] - M[2] * M[5], c21 = M[2] * M[4] - M[0] * M[6], c22 = M[0] * M[5] - M[1] * M[4];
  const remap = new Map(); let local = 0; const outIdx = new Array(idx.length);
  const ca = tex ? Math.cos(tex.ang) : 1, sa = tex ? Math.sin(tex.ang) : 0;
  for (let t = 0; t < idx.length; t++) {
    const v = idx[t]; let r = remap.get(v);
    if (r === undefined) {
      r = local++; remap.set(v, r);
      const q = xf(M, P0[v * 3], P0[v * 3 + 1], P0[v * 3 + 2]); b.pos.push(q[0], q[1], q[2]);
      if (N0) { const nx = N0[v * 3], ny = N0[v * 3 + 1], nz = N0[v * 3 + 2]; const X = c00 * nx + c01 * ny + c02 * nz, Y = c10 * nx + c11 * ny + c12 * nz, Z = c20 * nx + c21 * ny + c22 * nz; const L = Math.hypot(X, Y, Z) || 1; b.nrm.push(X / L, Y / L, Z / L); } else b.nrm.push(0, 1, 0);
      if (U0) {
        let u = U0[v * 2], w = U0[v * 2 + 1];
        if (!(Math.abs(u) < 1e3) || !(Math.abs(w) < 1e3)) { u = 0; w = 0; }
        if (tex.set === 0 && (tex.su !== 1 || tex.sv !== 1 || tex.ang)) { const ru = u * ca - w * sa, rv = u * sa + w * ca; u = ru * tex.su; w = rv * tex.sv; }
        if (tex.set > 0) { const m = DECAL_MODE; const uu = [u, u, w, 1 - w, w, 1 - w, 1 - u, 1 - u][m], vv = [1 - w, w, u, u, 1 - u, 1 - u, w, 1 - w][m]; b.uv.push(uu, vv); }
        else b.uv.push(u, FLIP_V ? 1 - w : w);
      }
    }
    outIdx[t] = r;
  }
  for (let t = 0; t < outIdx.length; t += 3) { if (flip) b.idx.push(b.v + outIdx[t], b.v + outIdx[t + 2], b.v + outIdx[t + 1]); else b.idx.push(b.v + outIdx[t], b.v + outIdx[t + 1], b.v + outIdx[t + 2]); }
  if (process.env.DEBUG_DECAL && tex && tex.set > 0) {
    let cx = 0, cy = 0, cz = 0, nx = 0, ny = 0, nz = 0, cnt = 0;
    for (const [v] of remap) { const u = U0[v * 2], w = U0[v * 2 + 1]; if (!(Math.abs(u) < 1e3 && u > 0.02 && u < 0.98 && w > 0.02 && w < 0.98)) continue; const q = xf(M, P0[v * 3], P0[v * 3 + 1], P0[v * 3 + 2]); cx += q[0]; cy += q[1]; cz += q[2]; const a = N0[v * 3], bb = N0[v * 3 + 1], c = N0[v * 3 + 2]; nx += c00 * a + c01 * bb + c02 * c; ny += c10 * a + c11 * bb + c12 * c; nz += c20 * a + c21 * bb + c22 * c; cnt++; }
    if (cnt) { const L = Math.hypot(nx, ny, nz); console.log('DECAL frag', f.i, 'mat', f.mid, 'img', info.decals[0].uri.slice(0, 8), 'centre', [cx / cnt - mid[0], cy / cnt - mid[1], cz / cnt - mid[2]].map((v) => +v.toFixed(3)), 'normal', [nx / L, ny / L, nz / L].map((v) => +v.toFixed(2)), 'valid verts', cnt); }
  }
  b.v += local; instTris += outIdx.length / 3;
}
// decals: a thin layer on exactly the triangles the decal covers (several decals per part are allowed)
const decalMats = new Map();
async function decalMaterial(uri) {
  if (decalMats.has(uri)) return decalMats.get(uri);
  const src = imageBuf(uri); if (!src) return null;
  const img = await sharp(src, { limitInputPixels: false }).resize({ width: P.texMax, height: P.texMax, fit: 'inside', withoutEnlargement: true }).ensureAlpha().png().toBuffer();
  const md = await sharp(img).metadata();
  const pad = 4; // transparent frame: clamped sampling outside the decal returns "nothing"
  const png = await sharp({ create: { width: md.width + 2 * pad, height: md.height + 2 * pad, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } }).composite([{ input: img, left: pad, top: pad }]).png().toBuffer();
  const t = await makeTexture('decal:' + uri, async () => png, 'decal_' + uri.slice(0, 8));
  const m = doc.createMaterial('decal_' + uri.slice(0, 8));
  m.setBaseColorTexture(t).setBaseColorFactor([1, 1, 1, 1]).setAlphaMode('BLEND').setMetallicFactor(0).setRoughnessFactor(0.5);
  m.getBaseColorTextureInfo().setWrapS(33071).setWrapT(33071);
  const out = { mat: m, su: md.width / (md.width + 2 * pad), sv: md.height / (md.height + 2 * pad), ou: pad / (md.width + 2 * pad), ov: pad / (md.height + 2 * pad) };
  decalMats.set(uri, out); return out;
}
let decalTris = 0;
for (const f of kept) {
  const info = infoOf(f.mid);
  if (!info.decals.length) continue;
  const idx = simp.get(f.gid); if (!idx) continue;
  const P0 = f.geo.getVertices(), N0 = f.geo.getNormals(); const M = f.M, flip = det3(M) < 0;
  const c00 = M[5] * M[10] - M[6] * M[9], c01 = M[6] * M[8] - M[4] * M[10], c02 = M[4] * M[9] - M[5] * M[8];
  const c10 = M[2] * M[9] - M[1] * M[10], c11 = M[0] * M[10] - M[2] * M[8], c12 = M[1] * M[8] - M[0] * M[9];
  const c20 = M[1] * M[6] - M[2] * M[5], c21 = M[2] * M[4] - M[0] * M[6], c22 = M[0] * M[5] - M[1] * M[4];
  for (const d of info.decals) {
    if (f.geo.getUvChannelCount() <= d.uv) continue;
    const U = f.geo.getUvs(d.uv);
    const ok = (v) => Math.abs(U[v * 2]) < 1e3 && Math.abs(U[v * 2 + 1]) < 1e3;   // not the 'no decal here' sentinel
    const touches = (t) => { let u0 = 9, u1 = -9, v0 = 9, v1 = -9; for (let k = 0; k < 3; k++) { const v = idx[t + k]; u0 = Math.min(u0, U[v * 2]); u1 = Math.max(u1, U[v * 2]); v0 = Math.min(v0, U[v * 2 + 1]); v1 = Math.max(v1, U[v * 2 + 1]); } return u1 > 0 && u0 < 1 && v1 > 0 && v0 < 1; };
    const tris = []; for (let t = 0; t < idx.length; t += 3) if (ok(idx[t]) && ok(idx[t + 1]) && ok(idx[t + 2]) && touches(t)) tris.push(t);
    if (!tris.length) continue;
    const dm = await decalMaterial(d.uri); if (!dm) continue; const mat = dm.mat;
    const key = `${f.group.name}|decal_${d.uri}`;
    let b = buckets.get(key);
    if (!b) { b = { group: f.group, mat, tex: { set: d.uv }, pos: [], nrm: [], uv: [], idx: [], v: 0 }; buckets.set(key, b); }
    const remap = new Map(); let local = 0;
    const vert = (v) => {
      let r = remap.get(v); if (r !== undefined) return r;
      r = local++; remap.set(v, r);
      const nx = N0[v * 3], ny = N0[v * 3 + 1], nz = N0[v * 3 + 2];
      let X = c00 * nx + c01 * ny + c02 * nz, Y = c10 * nx + c11 * ny + c12 * nz, Z = c20 * nx + c21 * ny + c22 * nz; const L = Math.hypot(X, Y, Z) || 1; X /= L; Y /= L; Z /= L;
      const q = xf(M, P0[v * 3], P0[v * 3 + 1], P0[v * 3 + 2]);
      const o = 0; // coplanar: the renderer wins the depth test with polygonOffset (normals of some CAD faces point inwards)
      b.pos.push(q[0] + X * o, q[1] + Y * o, q[2] + Z * o); b.nrm.push(X, Y, Z);
      b.uv.push(U[v * 2] * dm.su + dm.ou, U[v * 2 + 1] * dm.sv + dm.ov);
      return r;
    };
    for (const t of tris) { const i0 = vert(idx[t]), i1 = vert(idx[t + 1]), i2 = vert(idx[t + 2]); if (flip) b.idx.push(b.v + i0, b.v + i2, b.v + i1); else b.idx.push(b.v + i0, b.v + i1, b.v + i2); }
    if (process.env.DEBUG_DECAL2) { let u0 = 9, u1 = -9, v0 = 9, v1 = -9; for (const t of tris) for (let k = 0; k < 3; k++) { const v = idx[t + k]; u0 = Math.min(u0, U[v*2]); u1 = Math.max(u1, U[v*2]); v0 = Math.min(v0, U[v*2+1]); v1 = Math.max(v1, U[v*2+1]); } console.log('DECAL2 frag', f.i, 'mat', f.mid, d.uri.slice(0, 8), 'uvset', d.uv, 'tris', tris.length, '/', idx.length / 3, 'uv', [u0, u1, v0, v1].map((x) => x.toFixed(2)).join(' '), 'flip', flip, 'group', f.group.name); }
    b.v += local; decalTris += tris.length;
  }
}
console.log('baked instanced triangles', instTris, 'decal triangles', decalTris, 'buckets', buckets.size);

const meshes = new Map();
for (const b of buckets.values()) {
  const gname = b.group.name;
  if (!meshes.has(gname)) {
    const mesh = doc.createMesh(gname);
    const node = doc.createNode(gname).setMesh(mesh);
    node.setExtras({ region: b.group.region, dir: b.group.dir, lift: b.group.lift, delay: b.group.delay, c: b.group.centroid.map((v, i) => +(v - mid[i]).toFixed(4)) });
    sceneG.addChild(node);
    meshes.set(gname, mesh);
  }
  const prim = doc.createPrimitive();
  // shift so the model's centre is the origin (matches what the site expects)
  const pos = new Float32Array(b.pos.length); for (let i = 0; i < pos.length; i += 3) { pos[i] = b.pos[i] - mid[0]; pos[i + 1] = b.pos[i + 1] - mid[1]; pos[i + 2] = b.pos[i + 2] - mid[2]; }
  prim.setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(pos).setBuffer(buffer));
  prim.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(new Float32Array(b.nrm)).setBuffer(buffer));
  if (b.uv.length) prim.setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(new Float32Array(b.uv)).setBuffer(buffer));
  prim.setIndices(doc.createAccessor().setType('SCALAR').setArray(new Uint32Array(b.idx)).setBuffer(buffer));
  prim.setMaterial(b.mat);
  meshes.get(gname).addPrimitive(prim);
}
await doc.transform(prune());
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
await io.write(outPath, doc);
fs.writeFileSync(outPath.replace(/\.glb$/, '.groups.json'), JSON.stringify(groups.filter((g) => g.items.length).map((g) => ({ name: g.name, region: g.region, c: g.centroid.map((v, i) => +(v - mid[i]).toFixed(4)), dir: g.dir, lift: g.lift, delay: g.delay, parts: g.items.length })), null, 1));
console.log('wrote', outPath, (fs.statSync(outPath).size / 1e6).toFixed(1), 'MB; centre offset', mid.map((v) => +v.toFixed(4)));
