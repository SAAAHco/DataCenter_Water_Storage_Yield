"""v5: dry condenser mode with a conventional 6 K range and the COP read at the leaving condenser-water temperature (base),
plus variants stored as single-row arrays: PUEdry_opt (24 K range, COP at the entering temperature; the v4 base),
PUEdry_lo (24 K range, COP at the leaving temperature) and PUEwet_lo (the wet chiller mode, row 2, read as a conventional-range
chiller: COP at the leaving temperature of a 6 K range instead of at the entering temperature). Recomputes the water-free daily series, summary fields and the
dry-cooler capacity the base mode needs relative to the free-cooling design. Usage: python update_dry_v5.py <climate_dir> <model_dir> [--test cell]"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M, compute_hourly as CH
R = M.R
CLIM, MOD = sys.argv[1], sys.argv[2]; TEST = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--test" else None
DT = [0.0, 1.0, 2.0]; R_DRY = 6.0


def dry_only(T2, RH2, p, T1max=63.0, cop_T='entering', R_min=3.0, R_dry=None):
    prm = dict(R.DEFAULT); pr = R.Props(prm['f_cpL'], prm['f_hfg'], prm['f_cpa'])
    T2 = np.asarray(T2, float); RH2 = np.clip(np.asarray(RH2, float), 0.5, 100.0); n = len(T2); one = np.ones(n); p = np.broadcast_to(np.asarray(p, float), T2.shape).astype(float)
    T4d = np.maximum(prm['T4_design'], T2 + prm['a_dry']); R0 = float(prm['dT_chiller']) if R_dry is None else float(R_dry)
    Rd = np.full(n, R0) if T1max is None else np.maximum(np.minimum(R0, T1max - T4d), R_min)
    T1d = T4d + Rd; Tcop = {'entering': T4d, 'leaving': T1d, 'mid': T4d + 0.5 * Rd}[cop_T]; copd = R.cop_fun(Tcop, T1d, prm)
    dr = R.tower_states(T1d, T4d, T2, RH2, one, p, prm, pr, False)
    aux = dr['VA5'] * prm['dp_fan'] / 1000.0 + (dr['mw'] / 1000.0) * prm['dp_pump'] / 1000.0
    fr = R.tower_states(np.array([prm['T_ite_return']]), np.array([prm['T_ite_supply']]), np.array([12.0]), np.array([50.0]), np.array([1.0]), np.array([101325.0]), prm, pr, False)
    return dict(PUE3=1 + aux + 1.0 / copd, VA5_dry=dr['VA5'], VA5_free12=float(fr['VA5'][0]))


def wet_leaving_delta(T2, RH2, p):
    """Change of 1/COP of the wet chiller mode when it is read as a conventional-range chiller: COP at the leaving condenser-water temperature of a 6 K range (T4 + R_DRY) instead of at the entering temperature."""
    prm = dict(R.DEFAULT); pr = R.Props(prm['f_cpL'], prm['f_hfg'], prm['f_cpa']); p = np.broadcast_to(np.asarray(p, float), np.asarray(T2).shape).astype(float)
    Twb = R.wet_bulb(np.asarray(T2, float), np.clip(np.asarray(RH2, float), 0.5, 100.0), p, pr)
    T4c = np.maximum(prm['T4_design'], Twb + prm['a_ev']); T1c = T4c + R_DRY; T1p = T4c + prm['dT_chiller']
    return 1.0 / R.cop_fun(T1c, T1c, prm) - 1.0 / R.cop_fun(T4c, T1p, prm)


def rows_for(T, RH, P, dT):
    T = T.astype(float) + dT; RH = M.rh_const_w(T - dT, RH.astype(float), dT) if dT else RH.astype(float); P = P.astype(float) * 100.0
    key = np.stack([np.round(T, 1), np.round(RH, 0), np.round(P / 200.0) * 200.0], axis=1)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    base = dry_only(u[:, 0], u[:, 1], u[:, 2], R_dry=R_DRY, cop_T='leaving'); opt = dry_only(u[:, 0], u[:, 1], u[:, 2], cop_T='entering'); lo = dry_only(u[:, 0], u[:, 1], u[:, 2], cop_T='leaving')
    dwet = wet_leaving_delta(u[:, 0], u[:, 1], u[:, 2])
    return base['PUE3'][inv], opt['PUE3'][inv], lo['PUE3'][inv], dwet[inv], (float(np.nanmax(base['VA5_dry'] / base['VA5_free12'])), float(np.nanmax(opt['VA5_dry'] / opt['VA5_free12'])), float(np.nanmax(lo['VA5_dry'] / lo['VA5_free12'])))


def process(f):
    name = os.path.basename(f)[:-4]; z = dict(np.load(f)); zc = np.load(f"{CLIM}/{name}.npz"); T, RH, P = zc['T'], zc['RH'], zc['P']
    bad = ~(np.isfinite(T) & np.isfinite(RH) & np.isfinite(P))
    if bad.any():
        T = pd.Series(T).interpolate(limit_direction='both').values; RH = pd.Series(RH).interpolate(limit_direction='both').values; P = pd.Series(P).interpolate(limit_direction='both').values
    summ = dict(cell=name)
    for dT in DT:
        k = int(dT); PUE = z[f'PUE_{k}'].astype(float); WUE = z[f'WUE_{k}'].astype(float)
        d3, dopt, dlo, dwet, ratio = rows_for(T, RH, P, dT)
        PUE[3] = d3; z[f'PUE_{k}'] = PUE.astype(np.float16); z[f'PUEdry_opt_{k}'] = dopt.astype(np.float16); z[f'PUEdry_lo_{k}'] = dlo.astype(np.float16); z[f'PUEwet_lo_{k}'] = (PUE[2] + dwet).astype(np.float16)
        z.pop(f'PUEdry_hl_{k}', None)
        mode, pue, wue = M.select(PUE, WUE, 'water_free'); tag = f'water_free_{k}'
        z[f'dE_{tag}'] = CH.daily(pue - 1.0, 0).astype(np.float32); summ[f'PUE_{tag}'] = float(np.mean(pue))
        for m in range(4): summ[f'occ{m}_{tag}'] = float(np.mean(mode == m))
        if k == 0:
            summ['dry_cooler_extra'] = max(0.0, ratio[0] - 1.0); summ['dry_airflow_ratio_max'] = ratio[0]; summ['dry_cooler_extra_opt'] = max(0.0, ratio[1] - 1.0); summ['dry_airflow_ratio_max_opt'] = ratio[1]; summ['dry_cooler_extra_lo'] = max(0.0, ratio[2] - 1.0); summ['dry_airflow_ratio_max_lo'] = ratio[2]
    np.savez_compressed(f, **z)
    return summ


if __name__ == '__main__':
    files = [f"{MOD}/{TEST}.npz"] if TEST else sorted(glob.glob(f"{MOD}/*.npz")); t0 = time.time(); rows = []
    for i, f in enumerate(files):
        rows.append(process(f))
        if i % 25 == 0: print(f"{i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    upd = pd.DataFrame(rows).set_index('cell')
    if TEST: print(upd.T); sys.exit()
    sf = f"{MOD}/_summary.csv"
    if os.path.exists(sf):
        s = pd.read_csv(sf).set_index('cell')
        for c in upd.columns: s.loc[upd.index.intersection(s.index), c] = upd.loc[upd.index.intersection(s.index), c]
        s.reset_index().to_csv(sf, index=False)
    upd.reset_index().to_csv(f"{MOD}/_dry_update_v5.csv", index=False); print('DONE', len(files), 'files', round(time.time() - t0), 's', flush=True)
