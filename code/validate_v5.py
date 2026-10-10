"""Validation for the revision: (1) NOAA ISD-lite station observations at 46 airports versus the NASA POWER cell containing each
station, both run through the cooling model (2014-2023); (2) measured-facility cases reported by Han et al. (2026) against the
modeled water-priority and hybrid designs at the facility cells. Writes data_ext/results_v5/validation_*."""
import os, sys, gzip, glob, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M, compute_hourly as CH, revmodel as R
EXT = "F:/2025/Publications/Water_DataCenters/data_ext"; RES = EXT + "/results_v5"; MOD = EXT + "/model_power"; CLIM = EXT + "/climate_power"; os.makedirs(RES, exist_ok=True)
NDAYS = 3652; T_THR = 29.4; FR = [1.0, 1.2, 1.5]
month_of_day = pd.date_range("2014-01-01", periods=NDAYS, freq="D").month.values
def sequent_peak(W, Rr):
    C = np.cumsum(W - Rr); low = np.minimum.accumulate(np.minimum(C, 0.0)); return float(np.max(C - low))
def select_hybrid(PUE, WUE, T, thr):
    free = ~np.isnan(PUE[0]); evap = ~np.isnan(PUE[1]); hot = T > thr
    mode = np.where(free, 0, np.where(hot, np.where(evap, 1, 2), 3)); idx = np.arange(PUE.shape[1]); return mode, PUE[mode, idx], WUE[mode, idx]
def metrics(PUE, WUE, T):
    out = {}
    for des in ("water_priority", "hybrid"):
        if des == "hybrid": mode, pue, wue = select_hybrid(PUE, WUE, T, T_THR)
        else: mode, pue, wue = M.select(PUE, WUE, des)
        W = wue[:NDAYS * 24].reshape(-1, 24).sum(axis=1); mu = W.mean(); tag = "wp" if des == "water_priority" else "hy"
        out[f"WUE_{tag}"] = float(wue.mean()); out[f"PUE_{tag}"] = float(pue.mean()); out[f"pk_{tag}"] = float(W.max() / mu) if mu > 0 else np.nan
        out[f"p99_{tag}"] = float(np.quantile(W, 0.99) / mu) if mu > 0 else np.nan; out[f"S1_{tag}"] = sequent_peak(W, mu) / mu if mu > 0 else np.nan
        out[f"monthpk_{tag}"] = float(pd.Series(W).groupby(month_of_day).mean().max() / mu) if mu > 0 else np.nan
        out[f"free_{tag}"] = float(np.mean(mode == 0)); out[f"chill_{tag}"] = float(np.mean(mode == 2)); out[f"waterhours_{tag}"] = float(np.mean(wue > 0) * 8760)
    _, pwf, _ = M.select(PUE, WUE, "water_free"); out["PUE_wf"] = float(pwf.mean())
    _, pwp, _ = M.select(PUE, WUE, "water_priority"); e = pwp - 1.0; hrs = np.arange(len(e)) % 24; top = np.argsort(e)[-50:]
    out["cooltop50_hour"] = int(np.argmax(np.bincount(hrs[top], minlength=24))); prof = np.array([e[hrs == h_].mean() for h_ in range(24)]); out["cool_diurnal_amp_pct"] = float((prof.max() - prof.min()) / prof.mean() * 100); out["cool_meanprof_hour"] = int(np.argmax(prof))
    return out

# ---------------------------------------------------------------- 1. ISD stations
st = pd.read_csv(f"{EXT}/isd/stations_selected.csv", dtype=str); rows = []; t0 = time.time()
hours = pd.date_range("2014-01-01", "2023-12-31 23:00", freq="h")
for _, s in st.iterrows():
    parts = []
    for y in range(2014, 2024):
        f = f"{EXT}/isd/lite/{s.usaf}-{s.wban}-{y}.gz"
        if not os.path.exists(f): continue
        d = pd.read_csv(gzip.open(f, "rt"), sep=r"\s+", header=None, usecols=[0, 1, 2, 3, 4, 5, 6], names=["y", "m", "d", "h", "T", "Td", "slp"], na_values=[-9999])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True); d["t"] = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h), errors="coerce")
    d = d.dropna(subset=["t"]).set_index("t").sort_index(); d = d[~d.index.duplicated()]
    T = (d["T"] / 10.0).reindex(hours); Td = (d["Td"] / 10.0).reindex(hours); cov = float(T.notna().mean())
    T = T.interpolate(limit_direction="both").values; Td = np.minimum(Td.interpolate(limit_direction="both").values, T)
    RH = np.clip(100.0 * R.psat(Td) / R.psat(T), 0.5, 100.0); P = np.full(len(T), float(R.p_std(float(s.elev))))
    shift = int(round(float(s.lon) / 15.0))      # ISD-lite is UTC: shift to local solar time, as the POWER series are
    T = np.roll(T, shift); RH = np.roll(RH, shift)
    PUE, WUE, nu = CH.run_cell(T.astype(np.float32), RH.astype(np.float32), (P / 100.0).astype(np.float32), 0.0)
    m_st = metrics(PUE.astype(float), WUE.astype(float), T)
    z = np.load(f"{MOD}/S-{s.icao}.npz"); zc = np.load(f"{CLIM}/S-{s.icao}.npz"); Tc = zc["T"].astype(float)
    if not np.isfinite(Tc).all(): Tc = pd.Series(Tc).interpolate(limit_direction="both").values
    m_po = metrics(z["PUE_0"].astype(float), z["WUE_0"].astype(float), Tc)
    r = dict(icao=s.icao, name=s.name, lat=float(s.lat), lon=float(s.lon), elev=float(s.elev), coverage=cov, Tmean_station=float(np.mean(T)), Tmean_power=float(np.mean(Tc)))
    r.update({k + "_station": v for k, v in m_st.items()}); r.update({k + "_power": v for k, v in m_po.items()}); rows.append(r); print(s.icao, "cov", round(cov, 3), round(time.time() - t0), "s", flush=True)
v = pd.DataFrame(rows); v.to_csv(f"{RES}/validation_isd_stations.csv", index=False)
summ = []
for k in ("WUE_wp", "PUE_wp", "pk_wp", "p99_wp", "S1_wp", "free_wp", "chill_wp", "PUE_wf", "WUE_hy", "pk_hy", "monthpk_hy", "waterhours_hy", "cooltop50_hour", "cool_diurnal_amp_pct", "cool_meanprof_hour", "Tmean"):
    a = v[k + "_station"].values; b = v[k + "_power"].values; m = np.isfinite(a) & np.isfinite(b)
    summ.append(dict(metric=k, n=int(m.sum()), mean_station=float(a[m].mean()), mean_power=float(b[m].mean()), bias_power_minus_station=float((b[m] - a[m]).mean()), mae=float(np.abs(b[m] - a[m]).mean()), r=float(np.corrcoef(a[m], b[m])[0, 1]), rel_bias_pct=float(100 * (b[m] - a[m]).mean() / a[m].mean())))
pd.DataFrame(summ).to_csv(f"{RES}/validation_isd_summary.csv", index=False); print(pd.DataFrame(summ).round(3).to_string())

# ---------------------------------------------------------------- 2. measured-facility cases (Han et al. 2026)
cases = {"V-Phoenix": "Phoenix, AZ colocation facility with cooling towers (2019): peak-month/average-month WUE 1.46", "V-TheDalles": "The Dalles, OR pressure zone dominated by a hyperscale facility with cooling towers: daily peaking factor 2.21 (zone)",
         "V-WestDesMoines": "Iowa hyperscale campus with evaporative assist: monthly peaking factor 4.30 (2022-2025), daily at least 6.45; annual WUE 0.19 L/kWh (2022)",
         "V-Leesburg": "Leesburg, VA campus: planned daily peaking factor about 8 or higher; fleet annual WUE 0.14 L/kWh", "V-MountPleasant": "Wisconsin AI campus: planned peaking factor above 30; up to 480 hours of water use per year"}
fac = []
for name, desc in cases.items():
    z = np.load(f"{MOD}/{name}.npz"); zc = np.load(f"{CLIM}/{name}.npz"); Tc = zc["T"].astype(float)
    if not np.isfinite(Tc).all(): Tc = pd.Series(Tc).interpolate(limit_direction="both").values
    m = metrics(z["PUE_0"].astype(float), z["WUE_0"].astype(float), Tc); m.update(case=name, reported=desc); fac.append(m)
fac = pd.DataFrame(fac); fac.to_csv(f"{RES}/validation_facilities.csv", index=False)
print(fac[["case", "WUE_wp", "pk_wp", "monthpk_wp", "WUE_hy", "pk_hy", "monthpk_hy", "waterhours_hy"]].round(3).to_string()); print("done", round(time.time() - t0), "s")
