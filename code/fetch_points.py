"""NASA POWER hourly 2014-2023 (LST) for the ISD validation stations and the measured-facility cells. Resumable."""
import os, json, time, urllib.request
import numpy as np, pandas as pd
EXT = "F:/2025/Publications/Water_DataCenters/data_ext"; OUT = EXT + "/climate_power"
st = pd.read_csv(f"{EXT}/isd/stations_selected.csv", dtype=str)
pts = [(f"S-{r.icao}", float(r.lon), float(r.lat)) for _, r in st.iterrows()]
pts += [("V-Phoenix", -112.07, 33.45), ("V-TheDalles", -121.18, 45.60), ("V-WestDesMoines", -93.71, 41.57), ("V-Leesburg", -77.56, 39.11), ("V-MountPleasant", -87.89, 42.71)]
fails = 0; t0 = time.time()
for name, lon, lat in pts:
    f = f"{OUT}/{name}.npz"
    if os.path.exists(f): continue
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,PS&community=RE&longitude={lon:.4f}&latitude={lat:.4f}&start=20140101&end=20231231&format=JSON"
    js = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script (data-center water study)"})
            with urllib.request.urlopen(req, timeout=600) as resp: js = json.loads(resp.read().decode("utf8")); break
        except Exception as e:
            print(name, type(e).__name__, str(e)[:80], "retry", flush=True); time.sleep(20)
    if js is None or "properties" not in js: fails += 1; continue
    p = js["properties"]["parameter"]; keys = sorted(p["T2M"].keys())
    T = np.array([p["T2M"][k] for k in keys], dtype=np.float32); RH = np.array([p["RH2M"][k] for k in keys], dtype=np.float32); P = np.array([p["PS"][k] for k in keys], dtype=np.float32) * 10.0
    for a in (T, RH, P): a[a <= -990] = np.nan
    el = js["geometry"]["coordinates"][2] if len(js["geometry"]["coordinates"]) > 2 else np.nan
    np.savez_compressed(f, T=T, RH=RH, P=P, elevation=np.float32(el), lat=np.float32(lat), lon=np.float32(lon))
    print(name, "saved", round(time.time() - t0), "s", flush=True); time.sleep(1.0)
print("DONE fails", fails, flush=True)
