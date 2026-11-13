"""
Send every photo in a folder to the AgriQuest server and summarise the results.

  python batch_test.py "C:\\path\\to\\photos" --url https://YOUR-APP.onrender.com --expect healthy

--expect   the true class of ALL photos in the folder (healthy, or a label such as "Early Blight").
           Leave it out to just list what the model says.
--key      the API key, only if you set API_KEY on Render.
--sheet    also save a contact sheet of photos with their heatmaps (needs: pip install pillow).
Needs only Python 3; no other packages (Pillow only for --sheet).
Output: results.csv and, with --sheet, heatmaps.jpg, next to where you run it.
"""
import argparse, base64, csv, io, json, mimetypes, os, sys, time, uuid
import urllib.request, urllib.error
from collections import Counter

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def post(url, path, key):
    b = uuid.uuid4().hex
    data = open(path, "rb").read()
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
            f'Content-Type: {mimetypes.guess_type(path)[0] or "image/jpeg"}\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/predict", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    if key: req.add_header("X-API-Key", key)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--url", required=True)
    ap.add_argument("--expect", default=None); ap.add_argument("--key", default=None); ap.add_argument("--sheet", action="store_true"); ap.add_argument("--delay", type=float, default=1.0, help="seconds between photos")
    a = ap.parse_args()
    files = sorted(os.path.join(a.folder, f) for f in os.listdir(a.folder) if os.path.splitext(f)[1].lower() in EXT)
    if not files: sys.exit("No photos found in that folder.")
    exp = a.expect.lower().replace("leaf", "").strip() if a.expect else None
    try: urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=120).read()      # wake the free Render server
    except Exception as e: sys.exit(f"Cannot reach the server: {e}")
    rows, sheet = [], []
    for i, p in enumerate(files, 1):
        for attempt in range(6):                      # 502/503/429 = server waking up or busy: wait and retry
            try: r = post(a.url, p, a.key); break
            except Exception as e:
                r = None; err = str(e); time.sleep(10 + 10 * attempt)
        time.sleep(a.delay)
        if r is None:
            print(f"[{i}/{len(files)}] {os.path.basename(p)}  FAILED: {err}"); rows.append([os.path.basename(p), "ERROR", "", "", "", err]); continue
        ok = "" if exp is None else (exp in r["label"].lower().replace("leaf", "").strip())
        rows.append([os.path.basename(p), r["label"], r["confidence"], r["lowConfidence"], ok, " | ".join(f'{t["label"]} {t["confidence"]:.2f}' for t in r["topResults"])])
        print(f"[{i}/{len(files)}] {os.path.basename(p):40s} {r['label']:24s} {r['confidenceText']:>4s} {'' if ok == '' else ('OK' if ok else 'WRONG')}")
        if a.sheet and r.get("heatmapPng"): sheet.append((p, r["heatmapPng"]))
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "predicted", "confidence", "low_confidence", "correct", "top3"]); w.writerows(rows)
    good = [r for r in rows if r[1] != "ERROR"]
    print(f"\nPhotos scored: {len(good)} of {len(rows)}")
    if exp is not None and good:
        n_ok = sum(1 for r in good if r[4] is True)
        print(f"Correct ({a.expect}): {n_ok} / {len(good)} = {100 * n_ok / len(good):.1f}%")
    print("Predicted as:", dict(Counter(r[1] for r in good).most_common()))
    low = sum(1 for r in good if r[3] is True); print(f"Low-confidence (retake) flags: {low}")
    if a.sheet and sheet:
        from PIL import Image
        T = 220; cols = 6; rowsn = (len(sheet) + cols - 1) // cols
        S = Image.new("RGB", (cols * T, rowsn * T), "white")
        for k, (p, b64) in enumerate(sheet):
            im = Image.open(p).convert("RGBA"); hm = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
            im.alpha_composite(hm.resize(im.size)); im = im.convert("RGB"); im.thumbnail((T, T))
            S.paste(im, ((k % cols) * T, (k // cols) * T))
        S.save("heatmaps.jpg", quality=88); print("Saved heatmaps.jpg")
    print("Saved results.csv")

main()
