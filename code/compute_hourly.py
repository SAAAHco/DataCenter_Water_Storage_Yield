"""Run the four-mode cooling model on every climate cell (10 years hourly) for warming offsets 0, +1, +2 C.
Stores per cell: hourly PUE (4 modes) and WUE (2 wet modes) as float16, plus daily totals for the two bounding designs.
Usage: python compute_hourly.py [--test path.npz]"""
import os, sys, glob, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M

BASEW = "F:/2025/Publications/Water_DataCenters"
CLIM = BASEW + "/data_ext/climate_power"; OUT = BASEW + "/data_ext/model_power"; os.makedirs(OUT, exist_ok=True)
DT = [0.0, 1.0, 2.0]


def run_cell(T, RH, P, dT):
    """Bin the hourly states (0.1 C, 1 % RH, 2 hPa), evaluate unique states, map back."""
    T = T.astype(float) + dT; RH = M.rh_const_w(T - dT, RH.astype(float), dT) if dT else RH.astype(float)
    P = P.astype(float) * 100.0  # hPa -> Pa
    key = np.stack([np.round(T, 1), np.round(RH, 0), np.round(P / 200.0) * 200.0], axis=1)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    r = M.run_modes(u[:, 0], u[:, 1], u[:, 2])
    return r['PUE'][:, inv].astype(np.float16), r['WUE'][:, inv].astype(np.float16), len(u)


def daily(arr, hours_shift, n_hours_day=24):
    a = np.roll(arr, hours_shift)[:(len(arr) // 24) * 24].reshape(-1, 24)
    return a.sum(axis=1)


def process(cellfile, out_dir):
    name = os.path.basename(cellfile)[:-4]
    z = np.load(cellfile); T, RH, P = z['T'], z['RH'], z['P']; lon = float(z['lon'])
    bad = ~(np.isfinite(T) & np.isfinite(RH) & np.isfinite(P))
    if bad.any():
        T = pd.Series(T).interpolate(limit_direction='both').values; RH = pd.Series(RH).interpolate(limit_direction='both').values
        P = pd.Series(P).interpolate(limit_direction='both').values
    shift = 0  # NASA POWER hourly series are in local solar time (LST); no shift
    res = {}; summ = dict(cell=name, lon=lon, lat=float(z['lat']), elev=float(z['elevation']), n_bad=int(bad.sum()))
    for dT in DT:
        PUE, WUE, nu = run_cell(T, RH, P, dT)
        res[f'PUE_{int(dT)}'] = PUE; res[f'WUE_{int(dT)}'] = WUE
        for design in ('water_priority', 'water_free'):
            mode, pue, wue = M.select(PUE.astype(float), WUE.astype(float), design)
            tag = f"{design}_{int(dT)}"
            res[f'dW_{tag}'] = daily(wue, shift).astype(np.float32); res[f'dE_{tag}'] = daily(pue - 1.0, shift).astype(np.float32)
            summ[f'WUE_{tag}'] = float(np.mean(wue)); summ[f'PUE_{tag}'] = float(np.mean(pue))
            for m in range(4): summ[f'occ{m}_{tag}'] = float(np.mean(mode == m))
            dW = res[f'dW_{tag}']; summ[f'pkday_{tag}'] = float(dW.max() / dW.mean()) if dW.mean() > 0 else np.nan
            summ[f'p99day_{tag}'] = float(np.quantile(dW, 0.99) / dW.mean()) if dW.mean() > 0 else np.nan
        summ[f'nunique_{int(dT)}'] = nu
    summ['Tmax'] = float(np.max(T)); summ['Tmean'] = float(np.mean(T))
    np.savez_compressed(f"{out_dir}/{name}.npz", **res)
    return summ


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        t0 = time.time(); fp = os.path.abspath(sys.argv[2]); s = process(fp, os.path.dirname(fp)); print(json.dumps(s, indent=1)); print("seconds", round(time.time() - t0, 1)); sys.exit()
    files = sorted(glob.glob(f"{CLIM}/*.npz")); rows = []; t0 = time.time()
    done_file = f"{OUT}/_summary.csv"
    done = set(pd.read_csv(done_file).cell) if os.path.exists(done_file) else set()
    rows = [pd.read_csv(done_file)] if done else []
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-4]
        if name in done: continue
        try:
            s = process(f, OUT); rows.append(pd.DataFrame([s]))
        except Exception as e:
            print("FAILED", name, type(e).__name__, e, flush=True); continue
        if i % 25 == 0:
            pd.concat(rows, ignore_index=True).to_csv(done_file, index=False)
            print(f"{i+1}/{len(files)} cells, {time.time()-t0:.0f}s", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(done_file, index=False)
    print("DONE", len(files), "cells in", round(time.time() - t0), "s", flush=True)
