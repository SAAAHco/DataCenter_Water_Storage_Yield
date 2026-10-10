"""Run the cooling model on the station (S-*) and measured-facility (V-*) POWER cells fetched by fetch_points.py."""
import glob, os, sys, time, json, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_hourly as CH
CLIM = "F:/2025/Publications/Water_DataCenters/data_ext/climate_power"; OUT = "F:/2025/Publications/Water_DataCenters/data_ext/model_power"
rows = []; t0 = time.time()
for f in sorted(glob.glob(f"{CLIM}/S-*.npz") + glob.glob(f"{CLIM}/V-*.npz")):
    name = os.path.basename(f)[:-4]
    if os.path.exists(f"{OUT}/{name}.npz"): continue
    rows.append(CH.process(f, OUT)); print(name, round(time.time() - t0), "s", flush=True)
if rows: pd.DataFrame(rows).to_csv(f"{OUT}/_summary_points.csv", index=False)
print("done", len(rows))
