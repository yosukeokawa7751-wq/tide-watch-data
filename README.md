# tide-watch-data

Apple Watch の潮位アプリ「タイド」が読むデータ置き場。

## turbidity/

海岸沿いの区画（緯度0.02°×経度0.025°、約2km四方）ごとに、Sentinel-2 の画像から
**同じ時季のいつもの値と比べて濁っているか**を出す。範囲は宮城南部〜福島〜茨城・房総・東京湾・三浦・相模湾・静岡・愛知・三重（`pipeline.py` の `REGIONS`）。

- `pipeline.py` — 解析の本体（GitHub Actions が日本時間13〜18時の毎時と23時に回す）
- `obs/s2_YYYY.csv` / `obs/s3_YYYY.csv` — 区画ごとの観測（Sentinel-2 は赤−短波赤外の水面反射率・Sentinel-3 は濁りの推定 TSM の中央値）。基準の計算に使う。年ごとに分けてある
- `latest.json` — 時計が読むファイル。直近7日以内の画像がある区画だけ

段階: 2=強（同じ時季の上位1割かつ中央値の1.5倍以上） 1=やや強（上位4分の1かつ1.25倍以上） 0=いつもどおり

Contains modified Copernicus Sentinel data (2023–2026), processed by the author.
Images obtained from the Earth Search STAC (Element 84) / Sentinel-2 COGs on AWS Open Data.
