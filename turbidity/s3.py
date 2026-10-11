"""Sentinel-3 OLCI L2 WFR の濁り推定(TSM_NN)を区画ごとに出す。**時計にはまだ出さない(裏で記録して Sentinel-2 と比べる)**。

- 区画・「同じ時季のいつもと比べる」段階の決め方は pipeline.py と同じ(level_of をそのまま使う)
- 値は TSM_NN(懸濁物の推定 g/m³)の区画内の中央値。品質フラグで陸・雲・照り返しなどを除く
- 300m なので、2km の区画に晴れていれば約50画素入る。岸の区画は陸の混じりで弾かれやすい
- 取得には Copernicus Data Space のアカウントが要る(環境変数 CDSE_USER / CDSE_PASSWORD。GitHub の Secrets)

  python s3.py backfill 2023-01-01 2023-01-31 [出力CSV]   # 過去分(NT版を優先)。月ごとに分けて Actions で回す
  python s3.py daily                                       # 直近2日ぶんを obs_s3.csv に足して latest_s3.json を書く
  python s3.py latest [YYYY-MM-DD]
"""
import csv, datetime as dt, json, os, sys, time
import numpy as np, requests, netCDF4
import pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(HERE, "obs_s3.csv")
LATEST = os.path.join(HERE, "latest_s3.json")
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CAT = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
BAD = ["INVALID", "LAND", "CLOUD", "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "SNOW_ICE", "SUSPECT", "HISOLZEN",
       "SATURATED", "HIGHGLINT", "AC_FAIL", "WHITECAPS"]
MIN_PX = 5

def retry(f, tries=6):
    for i in range(tries):
        try: return f()
        except Exception as e:
            if i == tries - 1: raise
            print(f"  再試行 {i + 1}: {e}", flush=True); time.sleep(10 * (i + 1))

class Session:
    """トークンは10分で切れるので8分で取り直す。ログインの窓口が500を返すことがあった(2026-10-11)ので繰り返す"""
    def __init__(self): self.t = 0; self.s = requests.Session()
    def get(self, url, **kw):
        if time.time() - self.t > 480:
            def tok():
                r = requests.post(TOKEN_URL, data={"grant_type": "password", "client_id": "cdse-public",
                                                   "username": os.environ["CDSE_USER"], "password": os.environ["CDSE_PASSWORD"]}, timeout=60)
                r.raise_for_status(); return r.json()["access_token"]
            self.s.headers["Authorization"] = f"Bearer {retry(tok)}"; self.t = time.time()
        return self.s.get(url, **kw)

def search(start, end):
    w, s, e, n = p.REGION
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    f = (f"Collection/Name eq 'SENTINEL-3' and contains(Name,'OL_2_WFR') and "
         f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
         f"ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start le {end}T23:59:59.999Z")
    out = []
    while True:   # 1回は最大100件
        def q():
            r = requests.get(CAT, params={"$filter": f, "$orderby": "ContentDate/Start asc", "$top": 100, "$skip": len(out)}, timeout=120)
            r.raise_for_status(); return r.json()["value"]
        v = retry(q); out += v
        if len(v) < 100: break
    # 同じ撮影は NT(作り直し版)を優先し、無ければ NR(速報版)
    by = {}
    for it in out:
        k = it["Name"][:47]
        if k not in by or "_NT_" in it["Name"]: by[k] = it
    return sorted(by.values(), key=lambda x: x["ContentDate"]["Start"])

def fetch(sess, pid, pname, fname):
    url = f"{DL}({pid})/Nodes({pname})/Nodes({fname})/$value"
    def go():
        r = sess.get(url, allow_redirects=False, timeout=300)
        while r.status_code in (301, 302, 303, 307):
            r = sess.get(r.headers["Location"], allow_redirects=False, timeout=300)
        r.raise_for_status(); return r.content
    out = f"/tmp/s3_{fname}"
    open(out, "wb").write(retry(go)); return out

def var(path, name):
    with netCDF4.Dataset(path) as d:
        v = d.variables[name]; v.set_auto_maskandscale(True); a = v[:]
        attrs = {k: v.getncattr(k) for k in v.ncattrs()}
    return (np.ma.filled(a.astype(float), np.nan) if a.dtype.kind == "f" else np.ma.filled(a, 0)), attrs

def process(sess, it):
    """1枚 → {(iy,ix): (TSM の中央値 g/m³, 画素数)}"""
    name, pid = it["Name"], it["Id"]
    lat, _ = var(fetch(sess, pid, name, "geo_coordinates.nc"), "latitude")
    lon, _ = var(f"/tmp/s3_geo_coordinates.nc", "longitude")
    wq, wa = var(fetch(sess, pid, name, "wqsf.nc"), "WQSF")
    tsm, _ = var(fetch(sess, pid, name, "tsm_nn.nc"), "TSM_NN")   # log10(g/m³)
    bad = 0
    for nm, m in zip(wa["flag_meanings"].split(), [int(m) for m in np.atleast_1d(wa["flag_masks"])]):
        if nm in BAD: bad |= m
    R = p.REGION
    ok = (lon >= R[0]) & (lon < R[2]) & (lat >= R[1]) & (lat < R[3]) & ((wq.astype(np.uint64) & np.uint64(bad)) == 0) & np.isfinite(tsm)
    iy, ix = p.cell_of(lat[ok], lon[ok]); key = iy * 100000 + ix
    v = np.power(10, tsm[ok]); o = np.argsort(key); key, v = key[o], v[o]
    uk, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    return {(int(k // 100000), int(k % 100000)): (float(np.median(v[i:i + c])), int(c))
            for k, i, c in zip(uk, idx, cnt) if c >= MIN_PX}

def load_obs(path=OBS):
    d = {}
    if os.path.exists(path):
        for r in csv.reader(open(path)):
            if r and r[0] != "date": d[(r[0], int(r[1]), int(r[2]))] = (int(r[3]) / 100, int(r[4]))
    return d

def save_obs(d, path=OBS):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "iy", "ix", "tsm_x100", "n"])
        for (date, iy, ix), (v, n) in sorted(d.items()): w.writerow([date, iy, ix, round(v * 100), n])

def ingest(start, end, path=OBS):
    obs = load_obs(path); sess = Session()
    items = search(start, end); print(f"{len(items)} 枚", flush=True)
    for k, it in enumerate(items):
        date = it["ContentDate"]["Start"][:10]
        try: res = process(sess, it)
        except Exception as e:
            print("  失敗", it["Name"], e, flush=True); continue
        for (iy, ix), (v, n) in res.items():
            old = obs.get((date, iy, ix))
            if old is None or n > old[1]: obs[(date, iy, ix)] = (v, n)   # 同じ日の2機目・重なりは画素の多い方
        print(f"  {k + 1}/{len(items)} {it['Name'][:47]} 区画{len(res)}", flush=True)
        if (k + 1) % 20 == 0: save_obs(obs, path)
    save_obs(obs, path); return obs

def write_latest(obs, today):
    """pipeline.write_latest と同じ形。時計はまだ読まない"""
    old = p.LATEST; p.LATEST = LATEST
    try: p.write_latest(obs, today)
    finally: p.LATEST = old
    d = json.load(open(LATEST)); d["source"] = "Contains modified Copernicus Sentinel data (Sentinel-3 OLCI, TSM_NN)"
    json.dump(d, open(LATEST, "w"), ensure_ascii=False, separators=(",", ":"))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "backfill":
        ingest(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else OBS)
    elif cmd == "daily":
        t = dt.date.today()
        write_latest(ingest((t - dt.timedelta(days=2)).isoformat(), t.isoformat()), t)
    elif cmd == "latest":
        t = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today()
        write_latest(load_obs(), t)
