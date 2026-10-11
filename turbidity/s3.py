"""Sentinel-3 OLCI L2 WFR の濁り推定(TSM_NN)を区画ごとに出し、**Sentinel-2 の補いとして latest.json に混ぜる**(2026-10-11〜)。

【補いだけ・「やや濁り」まで】(ユーザー判断 2026-10-11「A」) 過去3年で Sentinel-2 と段階がほぼ一致しない(κ=0.04・
S2 が強の日のうち S3 も強は15%)。一方で測れる日は5倍あり(船形の近くで年36日 → 177日)、10/09 の船形を
強(216%)と出せたのは S3 だけだった。どちらが正しいかは決められないので、**S3 は S2 で測れていない所だけを埋め、
段階は最大1に抑える**(→ merge)。

- 区画・「同じ時季のいつもと比べる」段階の決め方は pipeline.py と同じ(level_of をそのまま使う)
- 値は TSM_NN(懸濁物の推定 g/m³)の区画内の中央値。品質フラグで陸・雲・照り返しなどを除く
- 300m なので、2km の区画に晴れていれば約50画素入る。岸の区画は陸の混じりで弾かれやすい
- 取得には Copernicus Data Space のアカウントが要る(環境変数 CDSE_USER / CDSE_PASSWORD。GitHub の Secrets)

  python s3.py backfill 2023-01-01 2023-01-31 [出力CSV]   # 過去分(NT版を優先)。月ごとに分けて Actions で回す
  python s3.py daily                                       # 直近2日ぶんを obs/s3_YYYY.csv に足して latest_s3.json を書き、latest.json に混ぜる
  python s3.py latest [YYYY-MM-DD]                         # 記録から latest_s3.json と latest.json を書き直す
"""
import csv, datetime as dt, json, os, sys, time
import numpy as np, requests, netCDF4
import pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
S3_STORE = ("s3", "tsm_x100", 100)
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
    # 範囲すべてを囲む長方形で探す(1枚が約1,200km四方なので、ほとんど同じ画像に入る)
    w, s, e, n = (min(r[1] for r in p.REGIONS), min(r[2] for r in p.REGIONS),
                  max(r[3] for r in p.REGIONS), max(r[4] for r in p.REGIONS))
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
    ok = p.in_region(lat, lon) & ((wq.astype(np.uint64) & np.uint64(bad)) == 0) & np.isfinite(tsm)
    iy, ix = p.cell_of(lat[ok], lon[ok]); key = iy * 100000 + ix
    v = np.power(10, tsm[ok]); o = np.argsort(key); key, v = key[o], v[o]
    uk, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    return {(int(k // 100000), int(k % 100000)): (float(np.median(v[i:i + c])), int(c))
            for k, i, c in zip(uk, idx, cnt) if c >= MIN_PX}

def load_obs(path=None): return p.load_store(S3_STORE, path)
def save_obs(d, path=None): p.save_store(d, S3_STORE, path)

def ingest(start, end, path=None):
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
    """pipeline.write_latest と同じ形(S3 だけの段階。比べるために残す。時計は読まない)"""
    old = p.LATEST; p.LATEST = LATEST
    try: p.write_latest(obs, today)
    finally: p.LATEST = old
    d = json.load(open(LATEST)); d["source"] = "Contains modified Copernicus Sentinel data (Sentinel-3 OLCI, TSM_NN)"
    json.dump(d, open(LATEST, "w"), ensure_ascii=False, separators=(",", ":"))

S3_MAX_LEVEL = 1
# 時計の WatchTurbidity.searchCells(自分から2区画先まで)と同じ値。変える時は両方
WATCH_SEARCH_CELLS = 2

def merge(c2, c3):
    """**S2 で測れていない所だけ S3 で埋める**。S3 の段階は最大1(やや濁り)に抑え、「いつもどおり」も載せる。
    S3 の区画は、周り4区画(時計が見る5×5が重なりうる範囲)に7日以内の S2 が1つも無い時だけ足す。
    → 時計の5×5の中に S2 と S3 が混ざることは無く、**S2 が見える場所の印は S2 だけの時と同じ**。
    (S3 の「やや」の区画だけを足すと、それが常に「いちばん新しい画像」になり、船形でやや濁りが
     出る日が 9% → 47% に増えた。2023〜2025年の実測)
    返り値は (区画, S3 を使った区画の名前)"""
    out, used = dict(c2), []
    R = 2 * WATCH_SEARCH_CELLS
    have2 = [tuple(map(int, k.split("_"))) for k in c2]
    for k, (d3, lv3, pct3) in c3.items():
        y, x = map(int, k.split("_"))
        if any(abs(y - sy) <= R and abs(x - sx) <= R for sy, sx in have2): continue
        out[k] = [d3, min(lv3, S3_MAX_LEVEL), pct3]; used.append(k)
    return out, sorted(used)

def write_merged(obs3, today):
    """latest.json を S2 の記録から作り直し、S3 を混ぜる(何度回しても同じ結果になる)"""
    c2 = p.latest_cells(p.load_obs(), today)
    cells, used = merge(c2, p.latest_cells(obs3, today))
    # "s3" は S3 で埋めた区画(時計は読まない。後から数えるため)。cells の形は変えない(古い時計が読めなくなる)
    p.write_latest(None, today, cells=cells, extra={"s3": used})
    print(f"latest.json に S3 を混ぜた: S2 {len(c2)}区画 + S3 {len(used)}区画(うちやや濁り {sum(1 for k in used if cells[k][1] == 1)})")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "backfill":
        ingest(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "daily":
        t = dt.date.today()
        obs = ingest((t - dt.timedelta(days=2)).isoformat(), t.isoformat())
        write_latest(obs, t); write_merged(obs, t)
    elif cmd == "latest":
        t = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today()
        obs = load_obs(); write_latest(obs, t); write_merged(obs, t)
