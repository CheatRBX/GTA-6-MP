# KIBO · FIRST Tech Challenge team site

A long scroll-driven site around the real CAD model of the team's robot.
Plain HTML/CSS/ES modules, no build step, nothing loaded from a CDN.

## Run

The page loads modules and a 3D model, so it needs a web server (opening `index.html` from disk will not work):

```bash
cd kibo-site
python3 -m http.server 8080      # or: npx serve .
# open http://localhost:8080
```

Deploy by uploading the folder to any static host (GitHub Pages, Netlify, Cloudflare Pages …).

## Fill in the team details

Edit `js/config.js`: team number, city, e-mail and social links. Empty fields are simply not shown.
All text is in `index.html` (Russian) and `js/i18n.js` (English). The RU/EN switch picks the browser language first.

## Debug / test parameters

* `?q=high|medium|low` forces a quality tier (default: chosen from the device, then adjusted by a frame-time governor)
* `?lang=ru|en`, `?still` renders on demand only (used for screenshots)

## What is where

```
index.html           content (RU), structure
css/style.css        design tokens, layout, responsive rules
js/main.js           boot, loader, language, reveals, render loop, quality governor
js/scene.js          Three.js stage: studio lighting, turntable, model, exploded view, section plane, AO
js/story.js          the scroll story: camera / rotation key frames, callouts, clock, dimension lines
assets/models        kibo-robot.glb (14 MB: all 5 029 parts, 0.25 mm tolerance), kibo-robot-lite.glb (phones, 4.8 MB), groups.json
tools/model          how the model is built from the Fusion share link
vendor               three.js, GSAP, Lenis (local copies)
```

Scroll effects: the robot turns on a turntable with the page, five callout stations, an exploded view (the real Fusion
sub-assemblies lift off in order: shooter, camera mast, turret, transfer, rear block, hubs, intake), a section plane that
slices the robot from top to bottom, dimension lines measured on the model, a match clock (0:00 → 2:38) and a process list. `prefers-reduced-motion` turns off smoothing and reveals.
