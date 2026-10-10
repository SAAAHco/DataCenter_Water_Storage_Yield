"""Pull 10 years of hourly ERA5-based weather for every climate cell that hosts a projected site, plus the four Texas stations.
Source: Open-Meteo Historical Weather API (CC BY 4.0). Resumable; three parallel workers; per-batch existence check."""
import os, sys, json, time, glob, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

BASEW = "F:/2025/Publications/Water_DataCenters"
OUT = BASEW + "/data_ext/climate"; os.makedirs(OUT, exist_ok=True)
START, END = "2014-01-01", "2023-12-31"
HOURS = 87648  # 3,652 days
cells = pd.read_csv(f"{BASEW}/data_ext/climate_cells.csv")
lock = threading.Lock(); state = dict(fails=0, saved=0)


def fetch(batch):
    lat = ",".join(f"{v:.4f}" for v in batch.cell_lat); lon = ",".join(f"{v:.4f}" for v in batch.cell_lon)
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
           f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure&timezone=UTC")
    req = urllib.request.Request(url, headers={"User-Agent": "research-script (data-center water study)"})
    with urllib.request.urlopen(req, timeout=400) as r:
        return json.loads(r.read().decode("utf8"))


def work(batch):
    batch = batch[[not os.path.exists(f"{OUT}/{c}.npz") for c in batch.cell]]
    if len(batch) == 0: return
    js = None
    for attempt in range(6):
        try:
            js = fetch(batch); break
        except urllib.error.HTTPError as e:
            sl = 300 * (attempt + 1) if e.code in (429, 503) else 20
            print(f"HTTP {e.code} -> sleep {sl}s", flush=True); time.sleep(sl)
        except Exception as e:
            print(f"error {type(e).__name__}: {str(e)[:120]} -> sleep 30s", flush=True); time.sleep(30)
    if js is None:
        with lock: state["fails"] += len(batch)
        return
    if isinstance(js, dict): js = [js]
    for cellname, j in zip(batch.cell, js):
        if "hourly" not in j:
            print("no hourly for", cellname, str(j)[:160], flush=True)
            with lock: state["fails"] += 1
            continue
        h = j["hourly"]
        T = np.array(h["temperature_2m"], dtype=np.float32); RH = np.array(h["relative_humidity_2m"], dtype=np.float32); P = np.array(h["surface_pressure"], dtype=np.float32)
        if len(T) != HOURS: print("length", len(T), "for", cellname, flush=True)
        np.savez_compressed(f"{OUT}/{cellname}.npz", T=T, RH=RH, P=P, elevation=np.float32(j.get("elevation", np.nan)), lat=np.float32(j["latitude"]), lon=np.float32(j["longitude"]))
        if not os.path.exists(f"{OUT}/_time.npy"): np.save(f"{OUT}/_time.npy", np.array(h["time"]))
        with lock: state["saved"] += 1
    time.sleep(4.0)


todo = cells[[not os.path.exists(f"{OUT}/{c}.npz") for c in cells.cell]].reset_index(drop=True)
print("cells", len(cells), "to fetch", len(todo), flush=True)
batches = [todo.iloc[i:i + 5] for i in range(0, len(todo), 5)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(work, b) for b in batches]
    for k, f in enumerate(futs):
        f.result()
        if k % 20 == 0: print(f"batch {k}/{len(batches)}: saved {state['saved']}, fails {state['fails']}, {time.time()-t0:.0f}s", flush=True)
print("DONE cells saved:", len(glob.glob(f"{OUT}/*.npz")), "fails:", state["fails"], flush=True)
