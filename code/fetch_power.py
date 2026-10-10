"""Pull 10 years (2014-2023) of hourly 2 m temperature, relative humidity and surface pressure from NASA POWER (MERRA-2 based,
0.5 x 0.625 degree) for every POWER grid cell that hosts a projected site, plus the four Texas station points. Resumable."""
import os, json, time, glob, urllib.request, urllib.error
import numpy as np, pandas as pd

BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; OUT = EXT + "/climate_power"; os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(f"{EXT}/sites_all_runs.csv")
if "cell_era" not in S.columns: S["cell_era"] = S.cell
S["pclat"] = np.round(np.round(S.lat / 0.5) * 0.5, 3); S["pclon"] = np.round(np.round(S.lon / 0.625) * 0.625, 4)
S["cell"] = ["P%.4f_%.3f" % (a, b) for a, b in zip(S.pclon, S.pclat)]
S.to_csv(f"{EXT}/sites_all_runs.csv", index=False)
cells = S[["cell", "pclon", "pclat"]].drop_duplicates().rename(columns={"pclon": "lon", "pclat": "lat"}).reset_index(drop=True)
tx = pd.DataFrame({"cell": ["TX_ElPaso", "TX_Lubbock", "TX_Dallas", "TX_Houston"], "lon": [-106.3112, -101.8761, -96.8358, -95.0832], "lat": [31.7791, 33.5281, 32.8384, 29.4718]})
cells = pd.concat([tx, cells], ignore_index=True); cells.to_csv(f"{EXT}/climate_cells_power.csv", index=False)
print("site records", len(S), "cells", len(cells), flush=True)
t0 = time.time(); fails = 0
for i, r in cells.iterrows():
    f = f"{OUT}/{r.cell}.npz"
    if os.path.exists(f): continue
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,PS&community=RE&longitude={r.lon:.4f}&latitude={r.lat:.4f}"
           f"&start=20140101&end=20231231&format=JSON")
    js = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script (data-center water study)"})
            with urllib.request.urlopen(req, timeout=600) as resp: js = json.loads(resp.read().decode("utf8")); break
        except Exception as e:
            print(f"{r.cell}: {type(e).__name__} {str(e)[:100]} -> retry in 30s", flush=True); time.sleep(30)
    if js is None or "properties" not in js:
        fails += 1; continue
    p = js["properties"]["parameter"]; keys = sorted(p["T2M"].keys())
    T = np.array([p["T2M"][k] for k in keys], dtype=np.float32); RH = np.array([p["RH2M"][k] for k in keys], dtype=np.float32); P = np.array([p["PS"][k] for k in keys], dtype=np.float32) * 10.0
    for a in (T, RH, P): a[a <= -990] = np.nan
    el = js["geometry"]["coordinates"][2] if len(js["geometry"]["coordinates"]) > 2 else np.nan
    np.savez_compressed(f, T=T, RH=RH, P=P, elevation=np.float32(el), lat=np.float32(r.lat), lon=np.float32(r.lon))
    if not os.path.exists(f"{OUT}/_time.npy"):
        np.save(f"{OUT}/_time.npy", np.array([f"{k[:4]}-{k[4:6]}-{k[6:8]}T{k[8:10]}:00" for k in keys]))
    if i % 25 == 0: print(f"{i+1}/{len(cells)} cells, {time.time()-t0:.0f}s, fails {fails}", flush=True)
    time.sleep(1.0)
print("DONE cells saved:", len(glob.glob(f"{OUT}/*.npz")), "fails:", fails, flush=True)
