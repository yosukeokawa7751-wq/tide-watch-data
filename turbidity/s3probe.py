"""Sentinel-3 OLCI L2 WFR の試験。館山湾などの区画の赤(Oa08・665nm)と濁りの推定(TSM_NN)を出し、
1枚あたりのデータ量と時間を測る。GitHub Actions の中で CDSE_USER / CDSE_PASSWORD を使って動かす。

  python s3probe.py 2026-10-08 2026-10-10
"""
import json, os, sys, time, datetime as dt
import numpy as np, requests, netCDF4

REGION = (139.0, 34.85, 140.9, 35.75)
LAT0, LON0, DLAT, DLON = 30.0, 128.0, 0.02, 0.025
CELLS = {"船形": (251, 473), "館山湾中央": (250, 471), "館山湾南": (249, 471), "船形北": (252, 472),
         "東京湾中央": (272, 474), "相模湾外": (240, 460)}
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CAT = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
# 水の反射率に使ってはいけない画素(フラグ名は製品の属性から引く。無い名前は無視)
BAD = ["INVALID", "LAND", "CLOUD", "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "SNOW_ICE", "SUSPECT", "HISOLZEN",
       "SATURATED", "HIGHGLINT", "AC_FAIL", "RWNEG_O8", "WHITECAPS"]

def token():
    r = requests.post(TOKEN_URL, data={"grant_type": "password", "client_id": "cdse-public",
                                       "username": os.environ["CDSE_USER"], "password": os.environ["CDSE_PASSWORD"]}, timeout=60)
    r.raise_for_status(); return r.json()["access_token"]

def search(start, end):
    w, s, e, n = REGION
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    f = (f"Collection/Name eq 'SENTINEL-3' and contains(Name,'OL_2_WFR') and "
         f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
         f"ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start le {end}T23:59:59.999Z")
    r = requests.get(CAT, params={"$filter": f, "$orderby": "ContentDate/Start asc", "$top": 100}, timeout=120)
    r.raise_for_status(); return r.json()["value"]

def fetch(sess, pid, pname, fname, out):
    url = f"{DL}({pid})/Nodes({pname})/Nodes({fname})/$value"
    t0 = time.time()
    r = sess.get(url, allow_redirects=False, timeout=300)
    while r.status_code in (301, 302, 303, 307):
        r = sess.get(r.headers["Location"], allow_redirects=False, timeout=300)
    r.raise_for_status()
    open(out, "wb").write(r.content)
    return len(r.content), time.time() - t0

def var(path, name):
    with netCDF4.Dataset(path) as d:
        v = d.variables[name]; v.set_auto_maskandscale(True)
        a = v[:]
        attrs = {k: v.getncattr(k) for k in v.ncattrs()}
    return np.ma.filled(a.astype(float), np.nan) if a.dtype.kind == "f" else np.ma.filled(a, 0), attrs

def main(start, end):
    tok = token()
    sess = requests.Session(); sess.headers["Authorization"] = f"Bearer {tok}"
    items = search(start, end)
    print(f"{len(items)} 件", flush=True)
    res = []
    for it in items:
        name, pid = it["Name"], it["Id"]
        tline = "NR" if "_NR_" in name else "NT" if "_NT_" in name else "?"
        lag = (dt.datetime.fromisoformat(it["PublicationDate"].replace("Z", "+00:00"))
               - dt.datetime.fromisoformat(it["ContentDate"]["Start"].replace("Z", "+00:00"))).total_seconds() / 3600
        got, nbytes, secs = {}, 0, 0
        try:
            for f in ["geo_coordinates.nc", "wqsf.nc", "Oa08_reflectance.nc", "tsm_nn.nc"]:
                b, s = fetch(sess, pid, name, f, f"/tmp/{f}"); nbytes += b; secs += s; got[f] = f"/tmp/{f}"
        except Exception as e:
            print("  失敗", name, e, flush=True); continue
        lat, _ = var(got["geo_coordinates.nc"], "latitude"); lon, _ = var(got["geo_coordinates.nc"], "longitude")
        wq, wa = var(got["wqsf.nc"], "WQSF")
        names = wa["flag_meanings"].split(); masks = [int(m) for m in np.atleast_1d(wa["flag_masks"])]
        badmask = 0
        for nm, m in zip(names, masks):
            if nm in BAD: badmask |= m
        o8, _ = var(got["Oa08_reflectance.nc"], "Oa08_reflectance")
        tsm, _ = var(got["tsm_nn.nc"], "TSM_NN")
        inr = (lon >= REGION[0]) & (lon < REGION[2]) & (lat >= REGION[1]) & (lat < REGION[3])
        iy = np.floor((lat - LAT0) / DLAT).astype(int); ix = np.floor((lon - LON0) / DLON).astype(int)
        ok = inr & ((wq.astype(np.uint64) & np.uint64(badmask)) == 0) & np.isfinite(o8)
        row = {"name": name, "timeline": tline, "lag_h": round(lag, 2), "MB": round(nbytes / 1e6, 1), "dl_s": round(secs, 1),
               "region_px": int(inr.sum()), "clear_px": int(ok.sum()), "cells": {}}
        for label, (cy, cx) in CELLS.items():
            sel = inr & (iy == cy) & (ix == cx)
            good = sel & ok
            if good.sum() >= 3:
                row["cells"][label] = {"n": int(good.sum()), "of": int(sel.sum()),
                                       "Oa08": round(float(np.nanmedian(o8[good])), 4),
                                       "TSM": round(float(np.nanmedian(np.power(10, tsm[good]))), 2)}   # TSM_NN は log10(g/m³) で入っている
            else:
                row["cells"][label] = {"n": int(good.sum()), "of": int(sel.sum())}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        res.append(row)
    json.dump(res, open("s3probe.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
