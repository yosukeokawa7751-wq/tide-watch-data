"""海岸沿いの区画ごとに、Sentinel-2 の画像から「いつもより濁っているか」を出す。

- 区画は緯度 DLAT × 経度 DLON（約2km四方）。時計は現在地から同じ式で区画の番号を出す
- 濁りの量は **赤（B04）− 短波赤外（B11）** の水面反射率の中央値。雲・影・陸を SCL で除き、晴れた水面だけを使う。
  短波赤外は水に吸収されて水中からはほぼ返らないので、そこに出る分は水面の照り返し・もや。赤だけだと多くの日で
  赤の値の大半がこれで（2026-10-08 館山湾の真ん中: 赤0.0257 / 短波赤外0.0200）、照り返しの多い日を「濁り」と読んでいた。
  引いた後も 10/10 の船形の周りは強のまま（いつもの1.5〜2.8倍）。Sentinel-3 の濁り推定との「場所の並び」の順位相関も
  0.25→0.31（日ごとは多くの日で0.4〜0.8）に上がった（2026-10-11）
- 「いつもより」は、同じ区画の過去の観測のうち、年をまたいで同じ時季（±WINDOW_DAYS）のものと比べる。
  夏はプランクトンや照り返しで赤が秋の数倍になるため（館山湾の中央: 6月0.092 / 10月0.013）、値そのものでは決めない
- 段階: 2=強（同じ時季の上位1割かつ中央値の1.5倍以上） 1=やや強（上位4分の1かつ1.25倍以上） 0=なし

使い方:
  python pipeline.py backfill 2023-01-01 2026-10-10   # 過去ぶんを obs.csv に足す
  python pipeline.py daily                             # 直近3日ぶんを足して latest.json を書く
  python pipeline.py latest [YYYY-MM-DD]               # obs.csv から latest.json だけ書き直す（日付は「今日」の代わり）
"""
import csv, datetime as dt, json, math, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(HERE, "obs.csv")
LATEST = os.path.join(HERE, "latest.json")

# 房総・東京湾・三浦・相模湾。広げる時は LON0/LAT0 を変えない（区画の番号が変わる）
LAT0, LON0 = 30.0, 128.0
DLAT, DLON = 0.02, 0.025
REGION = (139.0, 34.85, 140.9, 35.75)   # 西, 南, 東, 北
RES = 40.0                               # 読む細かさ(m)。COG の縮小版をそのまま使う
MIN_CLEAR_PX = 40                        # 40m×40m×40 ≒ 0.064km² 以上の晴れた水面
MIN_CLEAR_FRAC = 0.6                     # 水面・雲・影の合計のうち晴れた水面の割合
WINDOW_DAYS = 30
MIN_SAMPLES = 8
MAX_AGE_DAYS = 7
STAC = "https://earth-search.aws.element84.com/v1/search"
SCL_WATER = 6
SCL_OBSCURED = (3, 7, 8, 9, 10)          # 影・不明・雲(中/高)・薄雲

def cell_of(lat, lon):
    return np.floor((lat - LAT0) / DLAT).astype(int), np.floor((lon - LON0) / DLON).astype(int)

def stac_items(start, end, max_cloud=60):
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(REGION),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    url, out = STAC, []
    while url:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        for i in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r: d = json.load(r)
                break
            except Exception as e:
                if i == 3: raise
                time.sleep(5 * (i + 1))
        out += d.get("features", [])
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return out

def read(href):
    """RES の細かさで読む（COG の縮小版が使われるので、全画素を落とさずに済む）"""
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as s:
            shape = (int(round(s.height * s.res[1] / RES)), int(round(s.width * s.res[0] / RES)))
            return s.read(1, out_shape=shape, resampling=Resampling.nearest), s.transform, s.crs

_KEYS = {}
_KEYS_LOCK = threading.Lock()

def cell_keys(tr, crs, shape):
    """画素 → 区画の番号(範囲外は -1)。**タイルごとに1回だけ計算する**(全画素の座標変換が1枚の処理時間の大半だった)"""
    k = (str(crs), tr.c, tr.f, shape)
    with _KEYS_LOCK:
        if k in _KEYS: return _KEYS[k]
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    X = tr.c + (xs + 0.5) * RES; Y = tr.f - (ys + 0.5) * RES
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(X, Y)
    inr = (lon >= REGION[0]) & (lon < REGION[2]) & (lat >= REGION[1]) & (lat < REGION[3])
    iy, ix = cell_of(lat, lon)
    key = np.where(inr, iy * 100000 + ix, -1)
    with _KEYS_LOCK: _KEYS[k] = key
    return key

def process(item):
    """1枚の画像 → {(iy,ix): (赤の中央値, 晴れた水面の画素数)}"""
    a = item["assets"]
    scl, tr, crs = read(a["scl"]["href"])
    red, _, _ = read(a["red"]["href"])
    sw, _, _ = read(a["swir16"]["href"])
    assert red.shape == scl.shape == sw.shape, (red.shape, scl.shape, sw.shape)
    key = cell_keys(tr, crs, scl.shape)
    inr = key >= 0
    water = (scl == SCL_WATER) & (red > 0) & inr
    obsc = np.isin(scl, SCL_OBSCURED) & inr
    out = {}
    if not water.any(): return out
    # 照り返し・もやの分(短波赤外)を引く。earth-search の COG は 0.1 の底上げを済ませてある。引いた後は0以下もある(全体の16%)
    kw, rv = key[water], (red[water].astype(float) - sw[water]) * 1e-4
    ko = key[obsc]
    oc = dict(zip(*np.unique(ko, return_counts=True)))
    order = np.argsort(kw); kw, rv = kw[order], rv[order]
    uk, idx, cnt = np.unique(kw, return_index=True, return_counts=True)
    for k, i, n in zip(uk, idx, cnt):
        if n < MIN_CLEAR_PX or n / (n + oc.get(k, 0)) < MIN_CLEAR_FRAC: continue
        out[(int(k // 100000), int(k % 100000))] = (float(np.median(rv[i:i + n])), int(n))
    return out

def load_obs():
    d = {}
    if os.path.exists(OBS):
        for r in csv.reader(open(OBS)):
            if r and r[0] != "date": d[(r[0], int(r[1]), int(r[2]))] = (int(r[3]) / 1e5, int(r[4]))
    return d

def save_obs(d):
    with open(OBS, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "iy", "ix", "red_minus_swir_x1e5", "n"])
        for (date, iy, ix), (v, n) in sorted(d.items()): w.writerow([date, iy, ix, round(v * 1e5), n])

def ingest(start, end):
    obs = load_obs()
    items = sorted(stac_items(start, end), key=lambda f: f["properties"]["datetime"])
    print(f"{len(items)} 枚", flush=True)
    def work(it):
        try: return it, process(it), None
        except Exception as e: return it, None, e
    # 【4枚ずつ並べて読む】待ち時間の大半は通信。
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, (it, res, err) in enumerate(ex.map(work, items)):
            date = it["properties"]["datetime"][:10]
            if err is not None:
                print("  失敗", it["id"], err, flush=True); continue
            for (iy, ix), (v, n) in res.items():
                old = obs.get((date, iy, ix))
                if old is None or n > old[1]: obs[(date, iy, ix)] = (v, n)   # 隣のタイルと重なる区画は晴れた画素の多い方
            print(f"  {k+1}/{len(items)} {it['id']} 区画{len(res)}", flush=True)
            if (k + 1) % 50 == 0: save_obs(obs)
    save_obs(obs)
    return obs

def doy_dist(a, b):
    d = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(d, 365 - d)

def level_of(obs_by_cell, cell, date):
    """その日の観測を、同じ区画の同じ時季の過去と比べる → (段階, 中央値比, 標本数)"""
    v = obs_by_cell[cell][date]
    d0 = dt.date.fromisoformat(date)
    ref = [x for ds, x in obs_by_cell[cell].items()
           if ds != date and doy_dist(dt.date.fromisoformat(ds), d0) <= WINDOW_DAYS
           and abs((dt.date.fromisoformat(ds) - d0).days) > 3]
    if len(ref) < MIN_SAMPLES: return None, None, len(ref)
    med, p75, p90 = np.median(ref), np.percentile(ref, 75), np.percentile(ref, 90)
    ratio = v / med if med > 0 else None
    lv = 2 if v >= p90 and ratio and ratio >= 1.5 else 1 if v >= p75 and ratio and ratio >= 1.25 else 0
    return lv, ratio, len(ref)

def by_cell(obs):
    bc = {}
    for (date, iy, ix), (v, n) in obs.items(): bc.setdefault((iy, ix), {})[date] = v
    return bc

def write_latest(obs, today):
    bc = by_cell(obs)
    cells = {}
    for cell, dd in bc.items():
        recent = [d for d in dd if 0 <= (today - dt.date.fromisoformat(d)).days <= MAX_AGE_DAYS]
        if not recent: continue
        d = max(recent)
        lv, ratio, n = level_of(bc, cell, d)
        # 同じ時季の中央値が0以下(澄んだ沖で照り返しを引いた後)は「いつもの何%」が出せない。段階も0なので載せない
        if lv is None or ratio is None: continue
        cells[f"{cell[0]}_{cell[1]}"] = [d, lv, round(ratio * 100)]
    out = {"v": 1, "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "grid": {"lat0": LAT0, "lon0": LON0, "dlat": DLAT, "dlon": DLON},
           "maxAgeDays": MAX_AGE_DAYS,
           "source": "Contains modified Copernicus Sentinel data",
           "cells": dict(sorted(cells.items()))}
    json.dump(out, open(LATEST, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"latest.json 区画{len(cells)}（強{sum(1 for c in cells.values() if c[1]==2)}・やや強{sum(1 for c in cells.values() if c[1]==1)}）")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "backfill":
        ingest(sys.argv[2], sys.argv[3])
    elif cmd == "daily":
        t = dt.date.today()
        obs = ingest((t - dt.timedelta(days=3)).isoformat(), t.isoformat())
        write_latest(obs, t)
    elif cmd == "latest":
        t = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today()
        write_latest(load_obs(), t)
