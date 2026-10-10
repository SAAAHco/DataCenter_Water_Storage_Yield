"""Infrastructure capacity analysis, version 3 (revision after review): demand on withdrawals (cycles of concentration), a hybrid
design that evaporates only above a dry-bulb threshold, dry-day calendars and a forecast (reserve) rule, utility supply from the
USGS public-supply reanalysis with maximum-day headroom, peak coincidence on UTC by EIA-930 region against the actual 2022-2023
peak hours, an extended least-cost grid with a capacity charge and a Monte Carlo sensitivity. Version 5: dry mode with a conventional 6 K condenser range and the COP at the leaving temperature (bounds as variants), exact peak-day factor in the cost grid, cycles-of-concentration siting and exposure rows, grid-window shares. Version 4: dry mode inside a 63 C chiller envelope with priced dry-backup capital, summer/annual/winter peak hours, 60 gpcd screen with USGS 2015 state per-capita fallback. Writes data_ext/results_v4.
Usage: python infra_v4.py [--partial N]"""
import os, sys, json, glob, time, re
import numpy as np, pandas as pd, shapely
from shapely.strtree import STRtree
from pyogrio.raw import read as raw_read
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M

BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; MOD = EXT + "/model_power"; CLIM = EXT + "/climate_power"; RES = EXT + "/results_v5"
os.makedirs(RES, exist_ok=True)
PARTIAL = int(sys.argv[sys.argv.index("--partial") + 1]) if "--partial" in sys.argv else 0
IT_KW = 36000.0; IT_MW = 36.0; L_PER_MG = 3.785411784e6; L_PER_ML = 1e6; NDAYS = 3652; HOURS = 87648
COC = 5.0; MU = COC / (COC - 1.0)                    # makeup = evaporation x CoC/(CoC-1): 1.25 at five cycles (IM3: consumption fraction 0.8)
MU_SENS = {"coc3": 1.5, "coc6": 1.2}
T_THR = 29.4; T_THR_SENS = [25.0, 35.0]             # hybrid design: evaporative assist above 29.4 C (85 F) dry bulb
FRACS = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
FC = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]                 # cost grid; the cell's own peak factor is appended so that conventional service is a grid point
SDAYS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]
COST = dict(cap_usd_per_mgd=25e6, crf=0.0651, tank_usd_per_m3=200.0, pond_usd_per_m3=55.0, elec_usd_per_kwh=0.0813 * 1.5,
            cap_charge_usd_per_mw_yr=120e3, dry_capex_usd_per_kw_it=25.0, dry_cooler_usd_per_kw_th=150.0)
GPCD_MIN = 60.0                                     # reanalysis systems below this implied per-capita production (purchased water) use the state per-capita value
MDD_MONTH = 1.5; MDD_ANNUAL = 2.25                   # 22 CCR 64554: maximum day = 1.5 x maximum-month average day; 2.25 x annual average day
HEADROOM = [("mdd", 0.10), ("mdd", 0.25), ("mdd", 0.50), ("avg", 0.25)]
t0 = time.time()
def log(*a): print(*a, round(time.time() - t0), "s", flush=True)

# ---------------------------------------------------------------- inputs
sites = pd.read_csv(f"{EXT}/sites_all_runs.csv")
summ = pd.read_csv(f"{MOD}/_summary.csv").set_index("cell")
sites = sites[sites.cell.isin(summ.index)].reset_index(drop=True)
sites["run"] = sites.growth + "_" + sites.gravity.astype(int).astype(str); sites["state"] = sites.region.str.replace("_", " ").str.title()
cnty = pd.read_csv(f"{BASEW}/report_data/IM3_all_projected_sites_with_county.csv", dtype={"GEOID": str}).rename(columns={"growth_scenario": "growth"})
cnty["gravity"] = cnty.gravity.astype(int)
sites = sites.merge(cnty[["id", "growth", "gravity", "GEOID", "NAMELSAD", "STUSPS", "normalized_locational_cost"]], on=["id", "growth", "gravity"], how="left")
sites["xk"] = sites.x.round(0); sites["yk"] = sites.y.round(0)
# balancing authority / EIA-930 region of each site (county -> dominant utility -> BA), state majority as fallback, Connecticut -> NE
cb = pd.read_csv(f"{EXT}/eia861/county_ba_region.csv").dropna(subset=["region"]); cb["key"] = cb.state.str.upper() + "|" + cb.county.str.upper().str.strip()
norm = lambda s: re.sub(r"\s+(COUNTY|PARISH|BOROUGH|CENSUS AREA|CITY AND BOROUGH|MUNICIPALITY)$", "", str(s).upper()).strip()
sites["key"] = sites.STUSPS.astype(str).str.upper() + "|" + sites.NAMELSAD.map(norm)
kmap = cb.set_index("key").region.to_dict(); bamap = cb.set_index("key").ba.to_dict()
st_major = cb.groupby("state").region.agg(lambda v: v.mode().iloc[0]).to_dict(); st_major["CT"] = "NE"
sites["eia_region"] = [kmap.get(k, st_major.get(s, "MIDA")) for k, s in zip(sites.key, sites.STUSPS)]
sites["ba"] = [bamap.get(k, "") for k in sites.key]
cells = sorted(sites.cell.unique())
if PARTIAL: cells = sorted(sites[sites.run == "higher_50"].cell.unique())[:PARTIAL]; sites = sites[sites.cell.isin(cells)].reset_index(drop=True)
log("site records", len(sites), "cells", len(cells), "| region fallbacks:", int((~sites.key.isin(kmap)).sum()))
tm = np.load(f"{CLIM}/_time.npy", allow_pickle=True)
LON = {c: float(np.load(f"{CLIM}/{c}.npz")["lon"]) for c in cells}
def utc_shift(c): return int(round(-LON[c] / 15.0))  # LST index -> UTC index: roll forward by -lon/15 hours (west of Greenwich)

# ---------------------------------------------------------------- 1. hourly designs and daily series per cell
def select_hybrid(PUE, WUE, T, thr):
    free = ~np.isnan(PUE[0]); evap = ~np.isnan(PUE[1]); hot = T > thr
    mode = np.where(free, 0, np.where(hot, np.where(evap, 1, 2), 3)); idx = np.arange(PUE.shape[1])
    return mode, PUE[mode, idx], WUE[mode, idx]
def daily(arr): return arr[:NDAYS * 24].reshape(-1, 24).sum(axis=1)
D = {}; HP = {}; MONTHLY = {}
month_of_day = pd.date_range("2014-01-01", periods=NDAYS, freq="D").month.values; year_of_day = pd.date_range("2014-01-01", periods=NDAYS, freq="D").year.values
for i, c in enumerate(cells):
    z = np.load(f"{MOD}/{c}.npz"); PUE = z["PUE_0"].astype(float); WUE = z["WUE_0"].astype(float)
    zc = np.load(f"{CLIM}/{c}.npz"); T = zc["T"].astype(float)
    if not np.isfinite(T).all(): T = pd.Series(T).interpolate(limit_direction="both").values
    d = {"dW_wp": z["dW_water_priority_0"][:NDAYS].astype(float), "dE_wp": z["dE_water_priority_0"][:NDAYS].astype(float), "dE_wf": z["dE_water_free_0"][:NDAYS].astype(float),
         "dW_wp_2": z["dW_water_priority_2"][:NDAYS].astype(float), "dW_wp_1": z["dW_water_priority_1"][:NDAYS].astype(float)}
    _, p_wp, _ = M.select(PUE, WUE, "water_priority"); _, p_wf, _ = M.select(PUE, WUE, "water_free")
    free = ~np.isnan(PUE[0]); p_opt = np.where(free, PUE[0], z["PUEdry_opt_0"].astype(float)); p_lo = np.where(free, PUE[0], z["PUEdry_lo_0"].astype(float))
    m_wp, _, _ = M.select(PUE, WUE, "water_priority"); p_wp_cons = np.where(m_wp == 2, z["PUEwet_lo_0"].astype(float), p_wp)   # wet chiller COP read at its leaving temperature (consistent convention)
    d["dE_wf_opt"] = daily(p_opt - 1.0); d["dE_wf_lo"] = daily(p_lo - 1.0); d["dE_wp_cons"] = daily(p_wp_cons - 1.0); d["dry_cooler_extra"] = float(summ.loc[c, "dry_cooler_extra"]); d["dry_cooler_extra_opt"] = float(summ.loc[c, "dry_cooler_extra_opt"]); d["dry_cooler_extra_lo"] = float(summ.loc[c, "dry_cooler_extra_lo"])
    mh, p_hy, w_hy = select_hybrid(PUE, WUE, T, T_THR)
    d["dW_hy"] = daily(w_hy); d["dE_hy"] = daily(p_hy - 1.0); d["hy_occ"] = [float(np.mean(mh == m)) for m in range(4)]; d["hy_WUE"] = float(w_hy.mean()); d["hy_PUE"] = float(p_hy.mean())
    d["hy_water_hours"] = float(np.mean(w_hy > 0) * 8760.0)
    for thr in T_THR_SENS:
        _, _, w2 = select_hybrid(PUE, WUE, T, thr); d[f"dW_hy{int(thr)}"] = daily(w2); d[f"hy{int(thr)}_WUE"] = float(w2.mean()); d[f"hy{int(thr)}_water_hours"] = float(np.mean(w2 > 0) * 8760.0)
    d["Tday_max"] = T[:NDAYS * 24].reshape(-1, 24).max(axis=1)
    HP[c] = {"wp": (p_wp - 1.0).astype(np.float32), "wf": (p_wf - 1.0).astype(np.float32), "hy": (p_hy - 1.0).astype(np.float32), "wf_opt": (p_opt - 1.0).astype(np.float32), "wf_lo": (p_lo - 1.0).astype(np.float32), "wp_cons": (p_wp_cons - 1.0).astype(np.float32)}
    D[c] = d
    if i % 50 == 0: log("hourly designs", i, "/", len(cells))
log("designs done")

# ---------------------------------------------------------------- 2. storage-yield (sequent peak) and the daily simulation
def sequent_peak(W, R):
    C = np.cumsum(W - R); low = np.minimum.accumulate(np.minimum(C, 0.0)); return float(np.max(C - low))
def simulate_multi(W, R, S_list, pen, reserve_K=None):
    """Daily balance for several tank sizes at once. inflow R, demand W; a day the tank cannot cover is a dry-backup day (water 0,
    energy penalty pen[d]); dry days refill the tank by R. reserve_K: the tank is used only on the K highest-demand days of each
    year (forecast rule); other deficit days run dry even if the tank holds water."""
    S = np.asarray(S_list, float); L = S.copy(); n = len(W); short = np.zeros(len(S)); E = np.zeros(len(S)); dry = np.zeros((len(S), n), bool)
    tank_ok = np.ones(n, bool)
    if reserve_K is not None:
        tank_ok[:] = False
        for y in np.unique(year_of_day):
            idx = np.where(year_of_day == y)[0]; top = idx[np.argsort(W[idx])[::-1][:int(reserve_K)]]; tank_ok[top] = True
    for d in range(n):
        if not tank_ok[d] and W[d] > R:
            short += 1; E += pen[d]; L = np.minimum(S, L + R); dry[:, d] = True; continue
        ok = L + R >= W[d]
        L = np.where(ok, np.minimum(S, L + R - W[d]), np.minimum(S, L + R)); short += ~ok; E += pen[d] * (~ok); dry[~ok, d] = True
    return short / n, E * 365.0 / n, dry

rows = []; curves = []; SIM = {}; DRY = {}
for i, c in enumerate(cells):
    d = D[c]; pen = d["dE_wf"] - d["dE_wp"]; pen_opt = d["dE_wf_opt"] - d["dE_wp"]; pen_lo = d["dE_wf_lo"] - d["dE_wp"]; pen_cons = d["dE_wf"] - d["dE_wp_cons"]
    for design, key in (("water_priority", "dW_wp"), ("hybrid", "dW_hy"), ("hybrid25", "dW_hy25"), ("hybrid35", "dW_hy35"), ("water_priority_plus2", "dW_wp_2"), ("water_priority_plus1", "dW_wp_1")):
        W = d[key]; mu = W.mean()
        if mu <= 0:
            rows.append(dict(cell=c, design=design, mean_L_per_kW_day=0.0, peak_day_factor=np.nan, p99_day_factor=np.nan, annual_L_per_kW=0.0, month_peak_factor=np.nan, water_days_per_year=0.0, **{f'S_days_f{f}': np.nan for f in FRACS})); continue
        r = dict(cell=c, design=design, mean_L_per_kW_day=mu, peak_day_factor=W.max() / mu, p99_day_factor=np.quantile(W, 0.99) / mu, annual_L_per_kW=mu * 365.0)
        ym = pd.Series(W).groupby(month_of_day).mean(); r["month_peak_factor"] = float(ym.max() / mu)
        yr_max = [W[year_of_day == y].max() for y in np.unique(year_of_day)]; r["interannual_cv_peak"] = float(np.std(yr_max) / np.mean(yr_max))
        r["water_days_per_year"] = float(np.mean(W > 0) * 365.0)
        if design.startswith("hybrid"): r["water_hours_per_year"] = d["hy_water_hours"] if design == "hybrid" else d[f"hy{design[6:]}_water_hours"]
        if design == "water_priority":
            hot = [int(np.where(year_of_day == y)[0][np.argmax(d["Tday_max"][year_of_day == y])]) for y in np.unique(year_of_day)]
            r["coincident_factor"] = float(np.mean([W[h] for h in hot]) / mu); r["peak_on_hottest_day_share"] = float(np.mean([W[h] >= 0.9 * W[year_of_day == year_of_day[h]].max() for h in hot]))
        for f in FRACS:
            S = sequent_peak(W, f * mu); r[f"S_days_f{f}"] = S / mu
            if design in ("water_priority", "hybrid"): curves.append(dict(cell=c, design=design, f=f, S_days=S / mu, S_ML_100MW_withdrawal=S * MU * 100000.0 / L_PER_ML))
        if design == "water_priority":
            fl = FC + [float(r["peak_day_factor"]) * (1.0 + 1e-9)]; sim = {}; dry = {}   # exact peak-day factor: at this allocation no day exceeds the inflow, so conventional service is a grid point with no dry day
            for f in fl:
                sf, E, dr = simulate_multi(W, f * mu, [sd * mu for sd in SDAYS], pen)
                E_opt = (dr * pen_opt[None, :]).sum(axis=1) * 365.0 / len(W); E_lo = (dr * pen_lo[None, :]).sum(axis=1) * 365.0 / len(W); E_cons = (dr * pen_cons[None, :]).sum(axis=1) * 365.0 / len(W)
                for j, sd in enumerate(SDAYS):
                    sim[(f, sd)] = (float(sf[j]), float(E[j]), float(E_opt[j]), float(E_lo[j]), float(E_cons[j]))
                    if f == 1.0: r[f"pen_f1.0_S{sd}_opt"] = float(E_opt[j]); r[f"pen_f1.0_S{sd}_lo"] = float(E_lo[j]); r[f"pen_f1.0_S{sd}_cons"] = float(E_cons[j])
                    if (f, sd) in ((1.0, 0), (1.0, 3), (1.0, 7), (1.0, 30), (1.25, 7), (1.25, 0)): dry[(f, sd)] = dr[j].copy()
                    if f == 1.0: r[f"short_f1.0_S{sd}"] = float(sf[j]); r[f"pen_f1.0_S{sd}"] = float(E[j])
                    if f in (0.6, 0.8, 1.25): r[f"short_f{f}_S{sd}"] = float(sf[j]); r[f"pen_f{f}_S{sd}"] = float(E[j])
                dry[("dryall", f)] = dr  # all S for the capacity-charge share (nS x NDAYS)
            for K in (7, 14, 30):
                sf, E, dr = simulate_multi(W, 1.0 * mu, [7 * mu, 30 * mu], pen, reserve_K=K)
                r[f"short_f1.0_S7_reserve{K}"] = float(sf[0]); r[f"pen_f1.0_S7_reserve{K}"] = float(E[0]); r[f"short_f1.0_S30_reserve{K}"] = float(sf[1]); r[f"pen_f1.0_S30_reserve{K}"] = float(E[1])
                dry[("reserve", K, 7)] = dr[0].copy(); dry[("reserve", K, 30)] = dr[1].copy()
            r["f_peak"] = fl[-1]; SIM[c] = sim; DRY[c] = dry
        rows.append(r)
    if i % 50 == 0: log("storage-yield", i, "/", len(cells))
sy = pd.DataFrame(rows); sy.to_csv(f"{RES}/storage_yield_cells_v5.csv", index=False); pd.DataFrame(curves).to_csv(f"{RES}/storage_curves_cells_v5.csv", index=False)
sy0 = sy[sy.design == "water_priority"].set_index("cell"); syh = sy[sy.design == "hybrid"].set_index("cell")
log("storage-yield and simulation done")

# ---------------------------------------------------------------- 3. utilities: attribution (EPA service areas) and supply (USGS reanalysis)
meta, idx, geom, fields = raw_read(f"{BASEW}/report_data/EPA_municipal_water_service_areas_CONUS_simplified.gpkg")
F = dict(zip(meta["fields"], fields)); sa = shapely.from_wkb(geom)
pop = np.array([float(x) if x not in (None, "") else 0.0 for x in F["Population_Served_Count"]]); pw = np.array(F["PWSID"]); nm = np.array(F["PWS_Name"]); stt = np.array(F["Primacy_Agency"])
U = sites[["xk", "yk"]].drop_duplicates().reset_index(drop=True); P = shapely.points(U.xk.values, U.yk.values)
for thr, tag in ((10000, ""), (3300, "_3300")):
    big = pop >= thr; tree = STRtree(sa[big]); near = tree.nearest(P)
    U["PWSID" + tag] = pw[big][near]; U["PWS_Name" + tag] = nm[big][near]; U["pws_state" + tag] = stt[big][near]; U["pws_pop" + tag] = pop[big][near]; U["dist_km" + tag] = shapely.distance(P, sa[big][near]) / 1000.0
sites = sites.merge(U, on=["xk", "yk"], how="left"); sites["attributed"] = sites.dist_km <= 5.0; sites["attributed_3300"] = sites.dist_km_3300 <= 5.0
ps = pd.read_csv(f"{EXT}/usgs_ps/PS_WSA_Tot_2000_2020.csv"); rec = ps[(ps.Year >= 2016) & (ps.Year <= 2020)]
need = set(sites.PWSID.dropna()) | set(sites.PWSID_3300.dropna()); use_ids = [p for p in ps.columns if p in need]
vals = rec[use_ids].astype(float); sup = pd.DataFrame({"avg_mgd": vals.mean(), "maxmonth_mgd": vals.groupby(rec.Month.values).mean().max()})
sup = sup[(sup.avg_mgd > 0) & np.isfinite(sup.avg_mgd)]; sup["mdd_mgd"] = MDD_MONTH * sup.maxmonth_mgd
popmap = dict(zip(pw, pop)); stmap = dict(zip(pw, stt)); sup["pop"] = [popmap.get(p, np.nan) for p in sup.index]; sup["gpcd"] = sup.avg_mgd * 1e6 / sup["pop"]
us15 = pd.read_csv(f"{EXT}/usgs/usco2015v2.0.csv", dtype=str, skiprows=1, low_memory=False); us15["w"] = pd.to_numeric(us15["PS-Wtotl"], errors="coerce"); us15["p"] = pd.to_numeric(us15["PS-TOPop"], errors="coerce") * 1000.0
st15 = us15.groupby("STATE").agg(w=("w", "sum"), p=("p", "sum")); st_gpcd = (st15.w * 1e6 / st15.p).to_dict(); nat_gpcd = float(st15.w.sum() * 1e6 / st15.p.sum())   # USGS 2015 public-supply withdrawals per person served, by state
pd.DataFrame(dict(gpcd=st_gpcd)).to_csv(f"{RES}/state_gpcd_usgs2015.csv")
sup.to_csv(f"{RES}/utility_supply_reanalysis.csv")
def supply_for(pwsid):
    if pwsid in sup.index and GPCD_MIN <= sup.loc[pwsid, "gpcd"] <= 600.0: r = sup.loc[pwsid]; return r.avg_mgd, r.mdd_mgd, "reanalysis"
    g = st_gpcd.get(stmap.get(pwsid), nat_gpcd); a = popmap.get(pwsid, np.nan) * g / 1e6; return a, MDD_ANNUAL * a, "population x state per-capita"
for tag in ("", "_3300"):
    S3 = [supply_for(p) for p in sites["PWSID" + tag]]
    sites["supply_avg_mgd" + tag] = [s[0] for s in S3]; sites["supply_mdd_mgd" + tag] = [s[1] for s in S3]; sites["supply_source" + tag] = [s[2] for s in S3]
for des, key in (("", "dW_wp"), ("_hy", "dW_hy")):
    sites["mean_mgd" + des] = sites.cell.map({c: D[c][key].mean() * MU * IT_KW / L_PER_MG for c in cells}); sites["peak_mgd" + des] = sites.cell.map({c: D[c][key].max() * MU * IT_KW / L_PER_MG for c in cells})
for tagc, muc in MU_SENS.items(): sites["mean_mgd_" + tagc] = sites.mean_mgd * muc / MU; sites["peak_mgd_" + tagc] = sites.peak_mgd * muc / MU
sites.to_csv(f"{RES}/sites_infra_v5.csv", index=False)
att = sites[sites.attributed]; log("supply attached | attributed share higher_50:", round(float(sites[sites.run == "higher_50"].attributed.mean()), 3), "| reanalysis coverage of attributed systems:", round(float((att.drop_duplicates("PWSID").supply_source == "reanalysis").mean()), 3))

# ---------------------------------------------------------------- 4. capacity-aware siting
pool = sites.drop_duplicates(["state", "xk", "yk"]).copy(); pool["score"] = pool.normalized_locational_cost.fillna(0.5)
alloc_rows = []; site_rows = []; DESIGN_LABEL = {"": "water_priority", "_hy": "hybrid", "_coc3": "water_priority_coc3", "_coc6": "water_priority_coc6"}
def run_siting(run, hdef, h, regime, tag="", design=""):
    tgt = sites[sites.run == run]; pw_col = "PWSID" + tag; att_col = "attributed" + tag; avg_col = "supply_avg_mgd" + tag; mdd_col = "supply_mdd_mgd" + tag
    base = sites.drop_duplicates(pw_col).set_index(pw_col); head = ((base[mdd_col] if hdef == "mdd" else base[avg_col]) * h).to_dict()
    dcol = ("peak_mgd" if regime == "no_storage" else "mean_mgd") + design
    for state, g in tgt.groupby("state"):
        N = len(g); own = set(zip(g.xk, g.yk)); cand = pool[pool.state == state].copy(); cand["own"] = [(x, y) in own for x, y in zip(cand.xk, cand.yk)]
        cand = cand.sort_values(["own", "score"], ascending=[False, True]); placed = kept = moved = 0
        for _, s in cand.iterrows():
            if placed >= N: break
            if not s[att_col] or not np.isfinite(s[avg_col]): continue
            pid = s[pw_col]; need = s[dcol]
            if head.get(pid, 0.0) >= need:
                head[pid] -= need; placed += 1; kept += int(s.own); moved += int(not s.own)
                if tag == "" and hdef == "mdd" and h == 0.25 and design == "": site_rows.append(dict(run=run, regime=regime, state=state, xk=s.xk, yk=s.yk, cell=s.cell, PWSID=pid, own=bool(s.own)))
        alloc_rows.append(dict(run=run, headroom_def=hdef, h=h, regime=regime, threshold=3300 if tag else 10000, design=DESIGN_LABEL.get(design, "water_priority"), state=state, sites=N, servable=placed, kept_own=kept, moved=moved, unservable=N - placed, unservable_mw=(N - placed) * IT_MW, own_attributed=int(g[att_col].sum())))
runs = ["low_50", "moderate_50", "high_50"] + sorted(r for r in sites.run.unique() if r.startswith("higher"))
for run in runs:
    for hdef, h in HEADROOM:
        for regime in ("no_storage", "storage"): run_siting(run, hdef, h, regime)
for regime in ("no_storage", "storage"):
    run_siting("higher_50", "mdd", 0.25, regime, tag="_3300"); run_siting("higher_50", "mdd", 0.25, regime, design="_hy"); run_siting("higher_50", "mdd", 0.25, regime, design="_coc3"); run_siting("higher_50", "mdd", 0.25, regime, design="_coc6")
alloc = pd.DataFrame(alloc_rows); alloc.to_csv(f"{RES}/siting_v5.csv", index=False); pd.DataFrame(site_rows).to_csv(f"{RES}/siting_sites_v5.csv", index=False)
log("siting done")

# ---------------------------------------------------------------- 5. utility exposure (withdrawals; relative to average production and to the existing maximum day)
urows = []
for run, g in att.groupby("run"):
    info = g.groupby("PWSID").agg(PWS_Name=("PWS_Name", "first"), pws_state=("pws_state", "first"), pws_pop=("pws_pop", "first"), supply_avg_mgd=("supply_avg_mgd", "first"), supply_mdd_mgd=("supply_mdd_mgd", "first"), supply_source=("supply_source", "first"), sites=("id", "size"))
    for pid, gg in g.groupby("PWSID"):
        n = gg.groupby("cell").size(); arr = np.zeros(NDAYS); arrh = np.zeros(NDAYS)
        for c, k in n.items(): arr += k * D[c]["dW_wp"]; arrh += k * D[c]["dW_hy"]
        arr *= MU * IT_KW / L_PER_MG; arrh *= MU * IT_KW / L_PER_MG; r = info.loc[pid].to_dict(); r["PWSID"] = pid; r["run"] = run
        r.update(mean_mgd=arr.mean(), peak_mgd=arr.max(), peak_mgd_hy=arrh.max(), mean_mgd_hy=arrh.mean(), ratio_mean_avg=arr.mean() / r["supply_avg_mgd"], ratio_peak_avg=arr.max() / r["supply_avg_mgd"],
                 ratio_mean_mdd=arr.mean() / r["supply_mdd_mgd"], ratio_peak_mdd=arr.max() / r["supply_mdd_mgd"], ratio_peak_mdd_hy=arrh.max() / r["supply_mdd_mgd"],
                 days_over_25pct_mdd=float(np.mean(arr > 0.25 * r["supply_mdd_mgd"])), days_over_100pct_mdd=float(np.mean(arr > r["supply_mdd_mgd"])), blowdown_mgd_mean=arr.mean() / COC)
        urows.append(r)
util = pd.DataFrame(urows); util.to_csv(f"{RES}/utilities_v5.csv", index=False)
usum = util.groupby("run").agg(systems=("PWSID", "size"), n_peak_ge25=("ratio_peak_mdd", lambda v: int((v >= 0.25).sum())), n_mean_ge25=("ratio_mean_mdd", lambda v: int((v >= 0.25).sum())), n_peak_ge100=("ratio_peak_mdd", lambda v: int((v >= 1.0).sum())), n_mean_ge100=("ratio_mean_mdd", lambda v: int((v >= 1.0).sum())),
                                 n_peak_ge25_hy=("ratio_peak_mdd_hy", lambda v: int((v >= 0.25).sum())), n_peak_ge25_avg=("ratio_peak_avg", lambda v: int((v >= 0.25).sum())), n_mean_ge25_avg=("ratio_mean_avg", lambda v: int((v >= 0.25).sum()))).reset_index()
u0_ = util[util.run == "higher_50"]
for tagc, muc in MU_SENS.items():
    k_ = muc / MU; usum = pd.concat([usum, pd.DataFrame([dict(run=f"higher_50 ({tagc})", systems=len(u0_), n_peak_ge25=int((u0_.ratio_peak_mdd * k_ >= 0.25).sum()), n_mean_ge25=int((u0_.ratio_mean_mdd * k_ >= 0.25).sum()), n_peak_ge100=int((u0_.ratio_peak_mdd * k_ >= 1).sum()), n_mean_ge100=int((u0_.ratio_mean_mdd * k_ >= 1).sum()),
                                                            n_peak_ge25_hy=int((u0_.ratio_peak_mdd_hy * k_ >= 0.25).sum()), n_peak_ge25_avg=int((u0_.ratio_peak_avg * k_ >= 0.25).sum()), n_mean_ge25_avg=int((u0_.ratio_mean_avg * k_ >= 0.25).sum()))])], ignore_index=True)
usum.to_csv(f"{RES}/utilities_summary_v5.csv", index=False); log("utilities done")

# ---------------------------------------------------------------- 6. water-power peak coincidence on UTC, by EIA-930 region, against the actual grid peak hours
rh = pd.read_csv(f"{EXT}/eia930/region_hourly_utc_2022_2023.csv"); rh["utc"] = pd.to_datetime(rh.utc); T0 = pd.Timestamp("2014-01-01")
us = rh.groupby("utc").agg(dem=("dem", "sum"), n=("region", "size")).reset_index(); us = us[us.n == rh.region.nunique()].assign(region="US48")[["region", "utc", "dem"]]
rh = pd.concat([rh, us], ignore_index=True); rh["h"] = ((rh.utc - pd.Timedelta(hours=1) - T0) / pd.Timedelta(hours=1)).astype(int)   # EIA-930 stamps are hour-ending: index of the beginning of the hour
SEASON = {"summer": [6, 7, 8, 9], "annual": list(range(1, 13)), "winter": [12, 1, 2]}; TOP = {k: {} for k in SEASON}; modal_rows = []
for reg, g in rh.groupby("region"):
    for sname, months in SEASON.items():
        gs = g[g.utc.dt.month.isin(months)]; hh = []; h50 = []
        for y, gy in gs.groupby(gs.utc.dt.year): hh.extend(gy.nlargest(20, "dem").h.tolist()); h50.extend(((gy.nlargest(50, "dem").utc - pd.Timedelta(hours=1)).dt.hour).tolist())
        TOP[sname][reg] = np.array(sorted(set(hh))); modal_rows.append(dict(region=reg, season=sname, modal_utc_hour=int(pd.Series(h50).mode().iloc[0]), n_hours=len(set(hh))))
modal = pd.DataFrame(modal_rows); modal.to_csv(f"{RES}/region_peak_hours_v5.csv", index=False); MODAL_S = modal[modal.season == "summer"].set_index("region").modal_utc_hour.to_dict()
top_hours = TOP["summer"]   # summer peak hours are the primary set; annual and winter sets are reported alongside
t_index = pd.date_range("2014-01-01", periods=HOURS, freq="h")
def cell_utc(c, key): return np.roll(HP[c][key], utc_shift(c))
def day_of_hour(c): return np.clip((np.arange(HOURS) - utc_shift(c)) // 24, 0, NDAYS - 1)
cell_region = sites.groupby("cell").eia_region.agg(lambda v: v.mode().iloc[0]).to_dict()
state_region = sites.groupby("state").eia_region.agg(lambda v: v.mode().iloc[0]).to_dict()
inc_cell = {}; inc_cell_v = {v: {} for v in ("opt", "lo", "cons")}; peakdays = {}
for c in cells:
    hh = top_hours.get(cell_region[c], np.array([], int)); wp_c = cell_utc(c, "wp"); inc = (cell_utc(c, "wf") - wp_c) * IT_KW
    inc_cell[c] = float(inc[hh].mean()) if len(hh) else float(np.quantile(inc, 0.99)); peakdays[c] = np.unique(day_of_hour(c)[hh]) if len(hh) else np.array([], int)
    for v, incv in (("opt", (cell_utc(c, "wf_opt") - wp_c) * IT_KW), ("lo", (cell_utc(c, "wf_lo") - wp_c) * IT_KW), ("cons", (cell_utc(c, "wf") - cell_utc(c, "wp_cons")) * IT_KW)):
        inc_cell_v[v][c] = float(incv[hh].mean()) if len(hh) else float(np.quantile(incv, 0.99))
MASKS = [("S0", (1.0, 0)), ("S3", (1.0, 3)), ("S7", (1.0, 7)), ("S30", (1.0, 30)), ("f125_S7", (1.25, 7)), ("reserve7", ("reserve", 7, 7)), ("reserve14", ("reserve", 14, 7)), ("reserve30_S30", ("reserve", 30, 30))]
peak_rows = []; prof = {}
for run in ("higher_50", "moderate_50", "high_50", "low_50"):
    g = sites[sites.run == run]
    for level, col in (("region", "eia_region"), ("state", "state"), ("nation", None)):
        groups = list(g.groupby(col)) if col else [("US", g)]
        for name, gs in groups:
            cnt = gs.groupby("cell").size(); wp = np.zeros(HOURS); wf = np.zeros(HOURS); hy = np.zeros(HOURS); adds = {m: np.zeros(HOURS) for m, _ in MASKS}
            VARS = ("opt", "lo", "cons"); adds_v = {v: {m: np.zeros(HOURS) for m in ("S0", "S7")} for v in VARS}; MK = dict(MASKS)
            for c, k in cnt.items():
                s_wp = cell_utc(c, "wp").astype(float); s_wf = cell_utc(c, "wf").astype(float); inc = (s_wf - s_wp) * k * IT_MW; dh = day_of_hour(c)
                wp += k * IT_MW * s_wp; wf += k * IT_MW * s_wf; hy += k * IT_MW * cell_utc(c, "hy").astype(float)
                for m, key in MASKS: adds[m] += inc * DRY[c][key][dh]
                incv = {"opt": (cell_utc(c, "wf_opt").astype(float) - s_wp) * k * IT_MW, "lo": (cell_utc(c, "wf_lo").astype(float) - s_wp) * k * IT_MW, "cons": (s_wf - cell_utc(c, "wp_cons").astype(float)) * k * IT_MW}
                for v in VARS:
                    for m in ("S0", "S7"): adds_v[v][m] += incv[v] * DRY[c][MK[m]][dh]
            i = int(np.argmax(wp)); it_mw = len(gs) * IT_MW
            r = dict(run=run, level=level, name=name, it_mw=it_mw, eia_region=(name if level == "region" else (state_region.get(name) if level == "state" else "US48")), cool_mean_mw=wp.mean(), cool_peak_mw=wp.max(), peak_time_utc=str(t_index[i]), peak_hour_utc=int(i % 24),
                     add_at_cool_peak_S0=wf[i] - wp[i], wf_peak_mw=wf.max(), hy_mean_mw=hy.mean(), hy_peak_mw=hy.max(), hy_at_cool_peak=hy[i], peak_per_gw_it=wp.max() / it_mw * 1000, add_per_gw_it=(wf[i] - wp[i]) / it_mw * 1000)
            for m, _ in MASKS: r[f"add_at_cool_peak_{m}"] = float(adds[m][i])
            top50c = np.argsort(wp)[-50:]; r["cool_modal_hour_utc"] = int(pd.Series(top50c % 24).mode().iloc[0])
            for sname, pre in (("summer", "grid_"), ("annual", "grida_"), ("winter", "gridw_")):
                hh = TOP[sname].get(r["eia_region"], np.array([], int))
                if not len(hh): continue
                r.update({pre + "hours": len(hh), pre + "cool_mw": float(wp[hh].mean()), pre + "add_S0": float((wf - wp)[hh].mean()), pre + "add_S0_max": float((wf - wp)[hh].max()), pre + "hy_mw": float(hy[hh].mean()), pre + "add_S0_per_gw": float((wf - wp)[hh].mean() / it_mw * 1000)})
                for m, _ in MASKS: r[pre + f"add_{m}"] = float(adds[m][hh].mean())
                if sname == "summer":
                    for v in VARS: r[f"grid_add_S0_{v}"] = float(adds_v[v]["S0"][hh].mean()); r[f"grid_add_S7_{v}"] = float(adds_v[v]["S7"][hh].mean())
                    window = set(int(h % 24) for h in hh); r["cool_top50_in_window"] = float(np.mean([(int(h % 24) in window) for h in top50c])); r["grid_window_hours"] = len(window)
            prof24 = wp[:(HOURS // 24) * 24].reshape(-1, 24).mean(axis=0); r["cool_diurnal_amp_pct"] = float((prof24.max() - prof24.min()) / prof24.mean() * 100)
            if r["eia_region"] in MODAL_S: r["grid_modal_hour_utc"] = int(MODAL_S[r["eia_region"]]); r["lag_h"] = int(((r["cool_modal_hour_utc"] - MODAL_S[r["eia_region"]] + 12) % 24) - 12)
            peak_rows.append(r)
            if level == "nation":
                day = i // 24; j = int(top_hours["US48"][np.argmax(wp[top_hours["US48"]])]); gday = j // 24
                prof[run] = dict(cool_peak_day=str(t_index[day * 24])[:10], wp=wp[day * 24:(day + 1) * 24].tolist(), wf=wf[day * 24:(day + 1) * 24].tolist(), S7=(wp + adds["S7"])[day * 24:(day + 1) * 24].tolist(), reserve7=(wp + adds["reserve7"])[day * 24:(day + 1) * 24].tolist(),
                                 grid_peak_day=str(t_index[gday * 24])[:10], g_wp=wp[gday * 24:(gday + 1) * 24].tolist(), g_wf=wf[gday * 24:(gday + 1) * 24].tolist(), g_S7=(wp + adds["S7"])[gday * 24:(gday + 1) * 24].tolist(), g_reserve7=(wp + adds["reserve7"])[gday * 24:(gday + 1) * 24].tolist(), grid_peak_hours_utc=[int(h % 24) for h in top_hours["US48"]])
    log("peak coincidence", run)
pc = pd.DataFrame(peak_rows); pc.to_csv(f"{RES}/peak_coincidence_v5.csv", index=False); json.dump(prof, open(f"{RES}/peak_day_profiles_v5.json", "w"))

# ---------------------------------------------------------------- 7. least-cost service with the extended grid, capacity charge and Monte Carlo
crf = COST["crf"]; rows = []
for c in cells:
    r = sy0.loc[c]; mu = r.mean_L_per_kW_day; fl = FC + [r.f_peak]; pdays = peakdays[c]; inc_kw = inc_cell[c]; inc_v = {v: inc_cell_v[v][c] for v in ("opt", "lo", "cons")}
    adder_c = COST["dry_capex_usd_per_kw_it"] + COST["dry_cooler_usd_per_kw_th"] * D[c]["dry_cooler_extra"]   # chiller option and crossover, plus the dry coolers the 6 K range needs beyond the free-cooling design
    adder_v = {v: COST["dry_capex_usd_per_kw_it"] + COST["dry_cooler_usd_per_kw_th"] * D[c][f"dry_cooler_extra_{v}"] for v in ("opt", "lo")}   # the 24 K bounds carry the dry coolers of their own range
    for f in fl:
        dall = DRY[c][("dryall", f)]
        for j, sd in enumerate(SDAYS):
            sf, E, E_opt, E_lo, E_cons = SIM[c][(f, sd)]; E_v = dict(opt=E_opt, lo=E_lo, cons=E_cons); cap_mgd = f * mu * MU * IT_KW / L_PER_MG; S_m3 = sd * mu * MU * IT_KW / 1000.0
            dryshare = float(dall[j][pdays].mean()) if len(pdays) else sf
            rows.append(dict(cell=c, f=f, S_days=sd, conventional=int(f == r.f_peak and sd == 0), cap_mgd=cap_mgd, S_m3=S_m3, shortfall_frac=sf, pen_kwh_per_kw=E, dryshare_peak=dryshare, inc_peak_kw=inc_kw,
                             cost_cap=cap_mgd * COST["cap_usd_per_mgd"] * crf, cost_tank=S_m3 * COST["tank_usd_per_m3"] * crf, cost_pond=S_m3 * COST["pond_usd_per_m3"] * crf, cost_e=E * IT_KW * COST["elec_usd_per_kwh"],
                             cost_cc=COST["cap_charge_usd_per_mw_yr"] * inc_kw / 1000.0 * dryshare, dry_unit=IT_KW * crf if sf > 0 else 0.0, adder=adder_c, cost_dry=(IT_KW * crf if sf > 0 else 0.0) * adder_c, adder_opt=adder_v["opt"], adder_lo=adder_v["lo"], cost_dry_opt=(IT_KW * crf if sf > 0 else 0.0) * adder_v["opt"], cost_dry_lo=(IT_KW * crf if sf > 0 else 0.0) * adder_v["lo"],
                             **{f"cost_e_{v}": E_v[v] * IT_KW * COST["elec_usd_per_kwh"] for v in E_v}, **{f"cost_cc_{v}": COST["cap_charge_usd_per_mw_yr"] * inc_v[v] / 1000.0 * dryshare for v in inc_v}))
lc = pd.DataFrame(rows); lc["total_tank"] = lc.cost_cap + lc.cost_tank + lc.cost_e + lc.cost_cc + lc.cost_dry; lc["total_pond"] = lc.cost_cap + lc.cost_pond + lc.cost_e + lc.cost_cc + lc.cost_dry
lc.to_csv(f"{RES}/least_cost_grid_v5.csv", index=False)
conv = lc[lc.conventional == 1].set_index("cell"); bt = lc.loc[lc.groupby("cell").total_tank.idxmin()].set_index("cell"); bp = lc.loc[lc.groupby("cell").total_pond.idxmin()].set_index("cell")
lcs = pd.DataFrame(dict(best_f=bt.f, best_S_days=bt.S_days, best_cost_tank=bt.total_tank, best_f_pond=bp.f, best_S_days_pond=bp.S_days, best_cost_pond=bp.total_pond, conventional_cap_mgd=conv.cap_mgd, conventional_cost=conv.total_tank))
lcs["saving_tank_frac"] = 1 - lcs.best_cost_tank / lcs.conventional_cost; lcs["saving_pond_frac"] = 1 - lcs.best_cost_pond / lcs.conventional_cost
def breakeven(cell):
    g = lc[lc.cell == cell]; cv = float(g[g.conventional == 1].total_tank.iloc[0]); best = g[g.conventional == 0]
    gaps = ((cv - (best.total_tank - best.cost_dry)) / best.dry_unit.replace(0, np.nan)).astype(float)   # dry-backup capital (per kW of IT) at which the best storage option costs the same as conventional service
    if ((best.dry_unit == 0) & (best.total_tank <= cv)).any(): return float("inf")   # an option with no dry backup already beats conventional service at any dry-backup capital
    return float(np.nanmax(gaps)) if np.isfinite(gaps).any() else 0.0
lcs["breakeven_dry_capex_usd_per_kw"] = [breakeven(c) for c in lcs.index]; lcs.to_csv(f"{RES}/least_cost_cells_v5.csv")
hi = sites[sites.run == "higher_50"]; w = hi.groupby("cell").size(); wc = w.index.values; wts = w.values.astype(float)
ncomb = (len(FC) + 1) * len(SDAYS); A = np.zeros((len(wc), ncomb, 12)); convidx = np.zeros(len(wc), int); Sgrid = np.zeros((len(wc), ncomb)); ADDER = np.zeros(len(wc)); ADDER_OPT = np.zeros(len(wc)); ADDER_LO = np.zeros(len(wc))
for k, c in enumerate(wc):
    g = lc[lc.cell == c].reset_index(drop=True); A[k, :, 0] = g.cost_cap; A[k, :, 1] = g.cost_tank; A[k, :, 2] = g.cost_pond; A[k, :, 3] = g.cost_e; A[k, :, 4] = g.cost_cc; A[k, :, 5] = g.dry_unit; A[k, :, 6] = g.cost_e_opt; A[k, :, 7] = g.cost_cc_opt; A[k, :, 8] = g.cost_e_lo; A[k, :, 9] = g.cost_cc_lo; A[k, :, 10] = g.cost_e_cons; A[k, :, 11] = g.cost_cc_cons; ADDER[k] = float(g.adder.iloc[0]); ADDER_OPT[k] = float(g.adder_opt.iloc[0]); ADDER_LO[k] = float(g.adder_lo.iloc[0]); convidx[k] = int(np.where(g.conventional == 1)[0][0]); Sgrid[k] = g.S_days
def wmed(v): o = np.argsort(v); cw = np.cumsum(wts[o]) / wts.sum(); return float(np.interp(0.5, cw, v[o]))
def wmed_inf(v):
    v = np.asarray(v, float); o = np.argsort(v); cw = np.cumsum(wts[o]) / wts.sum(); return float(v[o][int(np.searchsorted(cw, 0.5))])   # lower weighted median, infinities kept
def wmed_f(v):
    v = np.asarray(v, float); m = np.isfinite(v)
    if not m.any(): return float("nan")
    o = np.argsort(v[m]); cw = np.cumsum(wts[m][o]) / wts[m].sum(); return float(np.interp(0.5, cw, v[m][o]))
def evaluate(cap_m=1.0, tank_m=1.0, e_m=1.0, cc_m=1.0, mu_m=1.0, adder=None, adder_m=1.0, cop="base"):
    ie, ic = {"base": (3, 4), "opt": (6, 7), "lo": (8, 9), "cons": (10, 11)}[cop]; ad = ({"base": ADDER, "cons": ADDER, "opt": ADDER_OPT, "lo": ADDER_LO}[cop] * adder_m)[:, None] if adder is None else adder
    tot = A[:, :, 0] * cap_m * mu_m + A[:, :, 1] * tank_m * mu_m + A[:, :, ie] * e_m + A[:, :, ic] * cc_m + A[:, :, 5] * ad
    best = tot.min(axis=1); bi = tot.argmin(axis=1); cv = tot[np.arange(len(wc)), convidx]; sv = 1 - best / cv
    return wmed(sv) * 100, float((wts * (Sgrid[np.arange(len(wc)), bi] > 0)).sum() / wts.sum()) * 100, float((wts * (sv > 0)).sum() / wts.sum()) * 100
rng = np.random.default_rng(7); mc = []
for n in range(2000):
    coc = rng.uniform(3.0, 6.0); kw = dict(cap_m=rng.uniform(10e6, 40e6) / COST["cap_usd_per_mgd"], mu_m=(coc / (coc - 1)) / MU, tank_m=np.exp(rng.uniform(np.log(0.5), np.log(2.0))), e_m=np.exp(rng.uniform(np.log(0.6), np.log(2.0))), cc_m=rng.uniform(0, 150e3) / COST["cap_charge_usd_per_mw_yr"], adder_m=rng.uniform(0, 4.0))
    ms, ss, sp = evaluate(**kw); mc.append(dict(draw=n, coc=coc, **kw, median_saving_pct=ms, share_storage_pct=ss, share_positive_pct=sp))
mc = pd.DataFrame(mc); mc.to_csv(f"{RES}/least_cost_montecarlo_v5.csv", index=False)
oatrows = [dict(case="Base case", **dict(zip(("median_saving_pct", "share_storage_pct", "share_positive_pct"), evaluate())))]
for case, kw in (("Capacity cost 10 M$ per MGD", dict(cap_m=0.4)), ("Capacity cost 40 M$ per MGD", dict(cap_m=1.6)), ("Tank cost x 0.5", dict(tank_m=0.5)), ("Tank cost x 2", dict(tank_m=2.0)), ("Electricity x 0.6", dict(e_m=0.6)), ("Electricity x 2", dict(e_m=2.0)),
                 ("No capacity charge", dict(cc_m=0.0)), ("Capacity charge 150 k$ per MW-yr", dict(cc_m=1.25)), ("3 cycles of concentration", dict(mu_m=1.5 / MU)), ("6 cycles of concentration", dict(mu_m=1.2 / MU)), ("No dry-backup capital", dict(adder=0.0)), ("Dry-backup capital 100 $ per kW IT", dict(adder=100.0)), ("Dry-backup capital 300 $ per kW IT", dict(adder=300.0)), ("Dry mode at the 24 K range with the COP at the entering temperature and dry coolers sized for that range (optimistic bound)", dict(cop="opt")), ("Dry mode at the 24 K range with the COP at the leaving temperature and dry coolers sized for that range", dict(cop="lo")), ("Both chillers at a conventional 6 K range with the COP at the leaving temperature", dict(cop="cons"))):
    oatrows.append(dict(case=case, **dict(zip(("median_saving_pct", "share_storage_pct", "share_positive_pct"), evaluate(**kw)))))
pd.DataFrame(oatrows).to_csv(f"{RES}/least_cost_sensitivity_v5.csv", index=False); log("least cost done")

# ---------------------------------------------------------------- 8. headline numbers
syc = sy0.loc[wc]; syh2 = syh.loc[wc]
def wm(col, src=None):
    src = syc if src is None else src; v = src.loc[wc, col].values.astype(float); m = np.isfinite(v); return float((v[m] * wts[m]).sum() / wts[m].sum())
H = dict(sites=int(len(hi)), cells=len(cells), MU=MU, T_THR=T_THR, pk_wmean=wm("peak_day_factor"), pk_hy_wmean=wm("peak_day_factor", syh2), pk_hy_min=float(syh2.peak_day_factor.min()), pk_hy_max=float(syh2.peak_day_factor.max()),
         monthpk_wp=wm("month_peak_factor"), monthpk_hy=wm("month_peak_factor", syh2), hy_WUE_wmean=float(sum(D[c]["hy_WUE"] * k for c, k in zip(wc, wts)) / wts.sum()), hy_water_hours=float(sum(D[c]["hy_water_hours"] * k for c, k in zip(wc, wts)) / wts.sum()),
         s1_wmean=wm("S_days_f1.0"), s1_hy_wmean=wm("S_days_f1.0", syh2), short_S7_reserve7=wm("short_f1.0_S7_reserve7") * 365, short_S7=wm("short_f1.0_S7") * 365,
         siting=alloc[alloc.run == "higher_50"].groupby(["headroom_def", "h", "regime", "threshold", "design"]).agg(sites=("sites", "sum"), servable=("servable", "sum"), kept=("kept_own", "sum"), moved=("moved", "sum")).reset_index().to_dict(orient="records"),
         peaks=pc[(pc.level == "nation")].to_dict(orient="records"), lc_saving_tank_median=wmed(lcs.loc[wc].saving_tank_frac.values) * 100, lc_share_storage=float((wts * (lcs.loc[wc].best_S_days > 0)).sum() / wts.sum()) * 100,
         lc_share_positive=float((wts * (lcs.loc[wc].saving_tank_frac > 0)).sum() / wts.sum()) * 100, mc_median_saving_q=[float(mc.median_saving_pct.quantile(q)) for q in (0.05, 0.5, 0.95)], mc_share_storage_q=[float(mc.share_storage_pct.quantile(q)) for q in (0.05, 0.5, 0.95)],
         util=usum[usum.run == "higher_50"].to_dict(orient="records"))
L_ = lcs.loc[wc]; win = (L_.best_S_days > 0).values; wwin = wts[win]
combo = pd.Series([f"{f_:g}|{int(sd)}" for f_, sd in zip(L_.best_f.values[win], L_.best_S_days.values[win])]); cw = combo.groupby(combo).apply(lambda g_: float(wwin[g_.index].sum())).sort_values(ascending=False)
def wmedv(v, w_): o = np.argsort(v); cwv = np.cumsum(w_[o]) / w_.sum(); return float(np.interp(0.5, cwv, v[o]))
pcr_ = pc[(pc.run == "higher_50") & (pc.level == "region")]
H.update(dict(lc_win=dict(share=float(wwin.sum() / wts.sum()), f_mode=float(cw.index[0].split("|")[0]), S_mode=int(cw.index[0].split("|")[1]), mode_share=float(cw.iloc[0] / wwin.sum()), f_median=wmedv(L_.best_f.values[win], wwin), S_median=wmedv(L_.best_S_days.values[win].astype(float), wwin)),
              anomalies=dict(month_gt_week=pcr_[pcr_.grid_add_S30 > pcr_.grid_add_S7].name.tolist(), f125_gt_week=pcr_[pcr_.grid_add_f125_S7 > pcr_.grid_add_S7].name.tolist())))
H.update(dict(nat_note="withdrawals", dry_capex_wmean=float((ADDER * wts).sum() / wts.sum()), cells_extra_coolers=int(sum(1 for c in wc if D[c]["dry_cooler_extra"] > 0)), extra_cooler_max=float(max(D[c]["dry_cooler_extra"] for c in wc)), breakeven_median=wmed_inf(pd.to_numeric(lcs.loc[wc].breakeven_dry_capex_usd_per_kw, errors="coerce").values), breakeven_share_never=float((wts * np.isinf(pd.to_numeric(lcs.loc[wc].breakeven_dry_capex_usd_per_kw, errors="coerce").values)).sum() / wts.sum()),
              dry_capex_wmean_opt=float((ADDER_OPT * wts).sum() / wts.sum()), dry_capex_wmean_lo=float((ADDER_LO * wts).sum() / wts.sum()), cells_extra_coolers_opt=int(sum(1 for c in wc if D[c]["dry_cooler_extra_opt"] > 0)), extra_cooler_max_opt=float(max(D[c]["dry_cooler_extra_opt"] for c in wc)), extra_cooler_min=float(min(D[c]["dry_cooler_extra"] for c in wc)),
              gpcd_state=st_gpcd, gpcd_nat=nat_gpcd, gpcd_min=GPCD_MIN, mc_share_near_zero=float((mc.median_saving_pct < 0.5).mean()), lag=pc[(pc.run == "higher_50") & (pc.level == "region")][["name", "cool_modal_hour_utc", "grid_modal_hour_utc", "lag_h"]].to_dict(orient="records")))
json.dump(H, open(f"{RES}/headline_v5.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in H.items() if k not in ("siting", "peaks", "util")}, indent=1, default=float)); log("ALL DONE")
