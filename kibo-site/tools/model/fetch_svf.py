#!/usr/bin/env python3
"""Download the viewable (SVF) of a public Autodesk Fusion share link, for build_robot.mjs.

    python3 fetch_svf.py "https://<hub>.autodesk360.com/g/shares/SH..." ./svf

It uses the same anonymous, view-only token the share page itself uses for its 3D viewer.
Only use it for designs you own or are allowed to reuse.
"""
import json, os, re, sys, urllib.parse, urllib.request, zipfile, concurrent.futures as cf

url, out = sys.argv[1], sys.argv[2]
m = re.match(r"(https://[^/]+)/g/shares/(SH[0-9A-Za-z]+)", url)
if not m:
    sys.exit("expected a link like https://<hub>.autodesk360.com/g/shares/SH...")
host, share = m.groups()

def get(u, data=None, headers=None):
    req = urllib.request.Request(u, data=data, headers=headers or {})
    return urllib.request.urlopen(req, timeout=120).read()

meta = json.loads(get(f"{host}/shares/metadata/{share}", headers={"Accept": "application/json"}))
urn = meta["success"]["body"]["viewing"]["views"]["viewableUrn"]
token = json.loads(get(f"{host}/shares/sign/{share}?oauth2=true", data=b"{}", headers={"Content-Type": "application/json"}))["accessToken"]
manifest = json.loads(get(f"{host}/shares/derivatives/{share}/manifest"))

def find_svf(node):
    if node.get("mime") == "application/autodesk-svf":
        return node["urn"]
    for c in node.get("children", []):
        r = find_svf(c)
        if r:
            return r
svf_urn = find_svf(manifest)
if not svf_urn:
    sys.exit("no SVF in the manifest")
root = svf_urn.rsplit("/", 1)[0] + "/"
base = "https://developer.api.autodesk.com/derivativeservice/v2/derivatives/"
auth = {"Authorization": "Bearer " + token}
def fetch(rel):
    return get(base + urllib.parse.quote(root + rel.replace("\\", "/"), safe=""), headers=auth)

os.makedirs(out, exist_ok=True)
svf = fetch(svf_urn.rsplit("/", 1)[1])
open(os.path.join(out, "Design.svf"), "wb").write(svf)
with zipfile.ZipFile(os.path.join(out, "Design.svf")) as z:
    assets = json.loads(z.read("manifest.json"))["assets"]
print(len(assets), "assets")

def one(a):
    uri = a["URI"]
    if uri.startswith("embed:"):
        return uri, True
    try:
        data = fetch(uri)
    except Exception as e:  # noqa: BLE001
        return uri, str(e)
    p = os.path.join(out, uri.replace("\\", "/").replace(":", ""))
    os.makedirs(os.path.dirname(p) or out, exist_ok=True)
    open(p, "wb").write(data)
    return uri, True

with cf.ThreadPoolExecutor(8) as ex:
    bad = [(u, r) for u, r in ex.map(one, assets) if r is not True]
print("done,", len(bad), "failed", bad[:5])
