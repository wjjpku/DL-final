import os, sys, time, urllib.request

URL = "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-win_amd64.whl"
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "wheels", "torch-2.9.1+cu128-cp313-cp313-win_amd64.whl")
os.makedirs(os.path.dirname(DST), exist_ok=True)


def total_size():
    for _ in range(5):
        try:
            req = urllib.request.Request(URL, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers.get("Content-Length", 0))
        except Exception as e:
            print("HEAD failed:", e, flush=True); time.sleep(3)
    return 0


total = total_size()
print("total bytes:", total, f"({total/1e9:.2f} GB)", flush=True)
attempt = 0
while True:
    have = os.path.getsize(DST) if os.path.exists(DST) else 0
    if total and have >= total:
        print("DOWNLOAD COMPLETE", have, flush=True); break
    attempt += 1
    if attempt > 80:
        print("GIVING UP after 80 attempts", flush=True); sys.exit(1)
    req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(DST, "ab") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        now = os.path.getsize(DST) if os.path.exists(DST) else 0
        print(f"attempt {attempt}: broke at {now/1e9:.2f}GB / {total/1e9:.2f}GB : {type(e).__name__} {e}", flush=True)
        time.sleep(2)
print("final size:", os.path.getsize(DST), flush=True)
