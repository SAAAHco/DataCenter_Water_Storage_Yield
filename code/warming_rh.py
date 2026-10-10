"""Warming sensitivity at constant relative humidity (+2 C; the main text uses constant specific humidity), water-priority design:
annual water, peak-day factor and storage at a mean-day allocation per cell. Writes results_v3/warming_constRH.csv."""
import os, sys, glob, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M, compute_hourly as CH
EXT = "F:/2025/Publications/Water_DataCenters/data_ext"; CLIM = EXT + "/climate_power"; R3 = EXT + "/results_v3"; NDAYS = 3652
cells = sorted(pd.read_csv(f"{R3}/sites_infra_v3.csv", usecols=["cell"]).cell.unique()); rows = []; t0 = time.time()
def sequent_peak(W, R):
    C = np.cumsum(W - R); low = np.minimum.accumulate(np.minimum(C, 0.0)); return float(np.max(C - low))
for i, c in enumerate(cells):
    z = np.load(f"{CLIM}/{c}.npz"); T, RH, P = z["T"].astype(float), z["RH"].astype(float), z["P"].astype(float)
    for a in (T, RH, P):
        if not np.isfinite(a).all(): a[:] = pd.Series(a).interpolate(limit_direction="both").values
    PUE, WUE, _ = CH.run_cell((T + 2.0).astype(np.float32), RH.astype(np.float32), P.astype(np.float32), 0.0)   # +2 C, RH unchanged
    mode, pue, wue = M.select(PUE.astype(float), WUE.astype(float), "water_priority"); W = wue[:NDAYS * 24].reshape(-1, 24).sum(axis=1); mu = W.mean()
    rows.append(dict(cell=c, WUE_rh2=float(wue.mean()), pk_rh2=float(W.max() / mu), S1_rh2=sequent_peak(W, mu) / mu, free_rh2=float(np.mean(mode == 0))))
    if i % 50 == 0: print(i, round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).to_csv(f"{R3}/warming_constRH.csv", index=False); print("done", round(time.time() - t0), "s")
