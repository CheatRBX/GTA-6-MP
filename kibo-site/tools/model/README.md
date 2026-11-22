# Rebuilding the robot model

`assets/models/kibo-robot.glb` (desktop) and `kibo-robot-lite.glb` (phones) come from the team's Fusion design.
Run these when the CAD changes:

```bash
cd tools/model
npm install
python3 fetch_svf.py "https://<hub>.autodesk360.com/g/shares/SH..." ./svf      # download the viewable
node build_robot.mjs ./svf ./out/hi.glb   hi                                   # geometry + real Fusion materials + decals
node build_robot.mjs ./svf ./out/lite.glb lite
node stage2.mjs ./out/hi.glb   ../../assets/models/kibo-robot.glb              # quantise + meshopt + WebP textures
node stage2.mjs ./out/lite.glb ../../assets/models/kibo-robot-lite.glb
cp ./out/hi.groups.json ../../assets/models/groups.json
# optional: a copy without meshopt for hosts that block WebAssembly (the site tries it automatically)
node stage2plain.mjs ./out/lite.glb ../../assets/models/kibo-robot-plain.glb
```

What `build_robot.mjs` does

* reads the SVF directly: geometry, every UV set, the Fusion "Prism" materials and the decal definitions;
* keeps **every** part and simplifies each mesh only down to an absolute deviation of 0.25 mm (desktop) or 0.8 mm
  (phones), so the shape matches the Autodesk viewer below one screen pixel;
* maps Prism materials to glTF PBR: metals, plastics, clear polycarbonate, carbon fibre (clear-coat + twill texture),
  brushed metal (bump → normal map); all CAD materials are double-sided like in the Autodesk viewer;
* puts decals (REV Control/Expansion Hub, goBILDA Pinpoint, Axon Mini, BRUSHLESS, Limelight …) on a separate layer that
  covers exactly the decal's triangles, with a transparent frame so nothing repeats outside the artwork;
* groups the real Fusion components into sub-assemblies for the exploded view (`GROUPS` / `RULES`): each group has a
  lift and a sideways move, ordered so that no part travels through another.

Hosts that won't serve `.glb` files can hand the site the bytes instead: define
`window.KIBO_MODELS = { hi, lite, plain }` before `js/main.js` loads, each entry a URL or a function
`(onProgress) => Promise<ArrayBuffer>` (for example, base64 text parts joined back together). Where WebAssembly is
missing or blocked by a CSP, the site skips the meshopt files and loads `plain` straight away.

Every glTF node keeps its own translation/scale after `stage2.mjs` (mesh quantisation stores the de-quantisation there);
the site adds explode offsets to it instead of overwriting it.

Checked against the share link: same camera in the Autodesk viewer and in the GLB (see the comparison helpers used during
the build), all 5 029 fragments inside their Autodesk bounds.
