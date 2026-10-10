"""Site, state, county, utility and grid analysis of the climate-resolved cooling model output.
Run after compute_hourly.py. --partial uses only cells already computed (for testing)."""
import os, sys, json, glob, time
import numpy as np, pandas as pd, shapely
from shapely.strtree import STRtree
from pyogrio.raw import read as raw_read
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ext as M

BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; MOD = EXT + "/model_power"; RES = EXT + "/results"
os.makedirs(RES, exist_ok=True)
PARTIAL = "--partial" in sys.argv
IT_KW = 36000.0; H_YR = 8760.0; L_PER_MG = 3.785411784e6; NDAYS = 3652
DESIGNS = ["water_priority", "water_free"]; DTS = [0, 1, 2]; G = ["low", "moderate", "high", "higher"]

t0 = time.time()
sites = pd.read_csv(f"{EXT}/sites_all_runs.csv")
summ = pd.read_csv(f"{MOD}/_summary.csv").set_index("cell")
if PARTIAL:
    sites = sites[sites.cell.isin(summ.index)].reset_index(drop=True); print("partial: site records", len(sites))
else:
    missing = set(sites.cell) - set(summ.index); print("cells missing:", len(missing))
    sites = sites[sites.cell.isin(summ.index)].reset_index(drop=True)
sites["run"] = sites.growth + "_" + sites.gravity.astype(int).astype(str)
sites["state"] = sites.region.str.replace("_", " ").str.title()
cnty = pd.read_csv(f"{BASEW}/report_data/IM3_all_projected_sites_with_county.csv", dtype={"GEOID": str})[["id", "growth_scenario", "gravity", "GEOID", "NAMELSAD", "STUSPS"]]
cnty = cnty.rename(columns={"growth_scenario": "growth"}); cnty["gravity"] = cnty.gravity.astype(int)
sites = sites.merge(cnty, on=["id", "growth", "gravity"], how="left")

# ---------------------------------------------------------------- 1. site-level annual metrics from cell summaries
for d in DESIGNS:
    for dT in DTS:
        tag = f"{d}_{dT}"
        wue = summ.loc[sites.cell, f"WUE_{tag}"].values; pue = summ.loc[sites.cell, f"PUE_{tag}"].values
        sites[f"cons_mgy_{tag}"] = wue * IT_KW * H_YR / L_PER_MG
        sites[f"cool_gwh_{tag}"] = (pue - 1.0) * IT_KW * H_YR / 1e6
        sites[f"pkday_{tag}"] = summ.loc[sites.cell, f"pkday_{tag}"].values
        sites[f"p99day_{tag}"] = summ.loc[sites.cell, f"p99day_{tag}"].values
        for m in range(4): sites[f"occ{m}_{tag}"] = summ.loc[sites.cell, f"occ{m}_{tag}"].values
sites["WUE_wp"] = summ.loc[sites.cell, "WUE_water_priority_0"].values; sites["PUE_wp"] = summ.loc[sites.cell, "PUE_water_priority_0"].values
sites["PUE_wf"] = summ.loc[sites.cell, "PUE_water_free_0"].values
sites["exchange_L_per_kWh"] = sites.WUE_wp / (sites.PUE_wf - sites.PUE_wp)
sites["im3_evap_cons_mgy"] = 34.815746 * sites.wfrac
sites.to_csv(f"{RES}/sites_metrics.csv", index=False)

# ---------------------------------------------------------------- 2. daily series per cell (L per kWh_IT per day) and hourly per-mode arrays
cells = sorted(sites.cell.unique())
D = {}
for c in cells:
    z = np.load(f"{MOD}/{c}.npz")
    D[c] = {k: z[k] for k in z.files if k.startswith("dW_") or k.startswith("dE_")}
print("cells loaded", len(cells), round(time.time() - t0), "s")

def group_daily(df, key_cols, design="water_priority", dT=0):
    """Coincident daily demand (MG/day) summed over the site records of each group, plus peak and mean."""
    out = []
    for keys, g in df.groupby(key_cols):
        n = g.groupby("cell").size()
        arr = np.zeros(NDAYS)
        for c, k in n.items(): arr += k * D[c][f"dW_{design}_{dT}"][:NDAYS]
        arr = arr * IT_KW / L_PER_MG
        if arr.mean() <= 0:
            out.append(dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)), sites=len(g), mean_mgd=0.0, peak_mgd=0.0, p99_mgd=0.0, pkfactor=np.nan, sum_site_peaks_mgd=0.0)); continue
        site_peaks = sum(k * D[c][f"dW_{design}_{dT}"][:NDAYS].max() for c, k in n.items()) * IT_KW / L_PER_MG
        out.append(dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)), sites=len(g), mean_mgd=arr.mean(), peak_mgd=arr.max(), p99_mgd=np.quantile(arr, 0.99),
                        pkfactor=arr.max() / arr.mean(), sum_site_peaks_mgd=site_peaks, exceed_days_frac=None))
    return pd.DataFrame(out)

# ---------------------------------------------------------------- 3. state and county totals (all runs, designs, warming)
rows = []
for d in DESIGNS:
    for dT in DTS:
        tag = f"{d}_{dT}"
        st = sites.groupby(["run", "growth", "gravity", "state"]).agg(sites=("id", "size"), cons_mgy=(f"cons_mgy_{tag}", "sum"), cool_gwh=(f"cool_gwh_{tag}", "sum"),
                                                                       im3_cons_mgy=("im3_cons_mgy", "sum"), im3_cool_gwh=("im3_cool_mwh", lambda v: v.sum() / 1000.0), im3_evap_cons_mgy=("im3_evap_cons_mgy", "sum"),
                                                                       mean_pkday=(f"pkday_{tag}", "mean")).reset_index()
        st["design"] = d; st["dT"] = dT; rows.append(st)
state_tot = pd.concat(rows, ignore_index=True)
# coincident state peaks for the main design and the 50 % gravity runs (all growth) and Higher at all gravities
pk = []
for (run, growth, grav), g in sites.groupby(["run", "growth", "gravity"]):
    if grav != 50 and growth != "higher": continue
    for dT in DTS:
        s = group_daily(g, ["state"], "water_priority", dT); s["run"] = run; s["dT"] = dT; pk.append(s)
state_peak = pd.concat(pk, ignore_index=True)
state_peak.to_csv(f"{RES}/state_coincident_peaks.csv", index=False); state_tot.to_csv(f"{RES}/state_totals.csv", index=False)
cty = sites[sites.gravity == 50].groupby(["run", "growth", "GEOID", "NAMELSAD", "STUSPS"]).agg(sites=("id", "size"), cons_mgy=("cons_mgy_water_priority_0", "sum"), im3_cons_mgy=("im3_cons_mgy", "sum"),
                                                                                             mean_pkday=("pkday_water_priority_0", "mean")).reset_index()
cty.to_csv(f"{RES}/county_totals_50gravity.csv", index=False)
print("state/county done", round(time.time() - t0), "s")

# ---------------------------------------------------------------- 4. national totals
nat = state_tot.groupby(["run", "growth", "gravity", "design", "dT"]).agg(sites=("sites", "sum"), cons_mgy=("cons_mgy", "sum"), cool_gwh=("cool_gwh", "sum"), im3_cons_mgy=("im3_cons_mgy", "sum"), im3_cool_gwh=("im3_cool_gwh", "sum"), im3_evap_cons_mgy=("im3_evap_cons_mgy", "sum")).reset_index()
natpk = []
for (run, growth, grav), g in sites.groupby(["run", "growth", "gravity"]):
    if grav != 50 and growth != "higher": continue
    for dT in DTS:
        g2 = g.copy(); g2["nation"] = "US"; s = group_daily(g2, ["nation"], "water_priority", dT); s["run"] = run; s["dT"] = dT; natpk.append(s)
natpk = pd.concat(natpk, ignore_index=True); nat.to_csv(f"{RES}/national_totals.csv", index=False); natpk.to_csv(f"{RES}/national_coincident_peaks.csv", index=False)

# ---------------------------------------------------------------- 5. utilities: nearest community water system serving >= 10,000 people within 5 km
meta, idx, geom, fields = raw_read(f"{BASEW}/report_data/EPA_municipal_water_service_areas_CONUS_simplified.gpkg")
F = dict(zip(meta["fields"], fields)); sa = shapely.from_wkb(geom)
pop = np.array([float(x) if x not in (None, "") else 0.0 for x in F["Population_Served_Count"]])
big = pop >= 10000; sa_big = sa[big]; pw_big = np.array(F["PWSID"])[big]; nm_big = np.array(F["PWS_Name"])[big]; st_big = np.array(F["Primacy_Agency"])[big]; pop_big = pop[big]
tree_all = STRtree(sa); tree_big = STRtree(sa_big)
upts = sites[["x", "y"]].round(0).drop_duplicates().reset_index(drop=True)
P = shapely.points(upts.x.values, upts.y.values)
near = tree_big.nearest(P); dist = shapely.distance(P, sa_big[near]) / 1000.0
inside_any = np.zeros(len(P), bool); q = tree_all.query(P, predicate="within"); inside_any[q[0]] = True
upts["PWSID"] = pw_big[near]; upts["PWS_Name"] = nm_big[near]; upts["pws_state"] = st_big[near]; upts["pws_pop"] = pop_big[near]; upts["dist_km"] = dist; upts["inside_any"] = inside_any
sites = sites.merge(upts, on=["x", "y"], how="left") if False else sites.merge(upts.assign(x=upts.x, y=upts.y), left_on=[sites.x.round(0), sites.y.round(0)], right_on=["x", "y"], how="left", suffixes=("", "_u"))
sites = sites.drop(columns=[c for c in sites.columns if c.endswith("_u")] + ["key_0", "key_1"], errors="ignore")
sites["attributed"] = sites.dist_km <= 5.0
# USGS county per-capita public supply
usgs = pd.read_csv(f"{EXT}/usgs/usco2015v2.0.csv", skiprows=1, dtype={"FIPS": str}, na_values=["--"])
usgs["gpcd"] = usgs["PS-Wtotl"] * 1e6 / (usgs["PS-TOPop"] * 1000.0)
usgs.loc[~np.isfinite(usgs.gpcd) | (usgs.gpcd < 40) | (usgs.gpcd > 500), "gpcd"] = np.nan  # purchased-water counties report near-zero withdrawals
state_gpcd = usgs.groupby("STATE").gpcd.median()
usgs["gpcd_f"] = usgs.gpcd.fillna(usgs.STATE.map(state_gpcd))
sites = sites.merge(usgs[["FIPS", "gpcd_f", "PS-Wtotl", "PS-TOPop", "TO-WFrTo"]].rename(columns={"FIPS": "GEOID", "PS-Wtotl": "county_ps_mgd", "PS-TOPop": "county_ps_pop_k", "TO-WFrTo": "county_fresh_mgd"}), on="GEOID", how="left")
sites.to_csv(f"{RES}/sites_metrics.csv", index=False)
urows = []
att = sites[sites.attributed]
for dT in [0, 2]:
    for run, g in att.groupby("run"):
        s = group_daily(g, ["PWSID"], "water_priority", dT)
        info = g.groupby("PWSID").agg(PWS_Name=("PWS_Name", "first"), pws_state=("pws_state", "first"), pws_pop=("pws_pop", "first"), gpcd=("gpcd_f", "median"), county_ps_mgd=("county_ps_mgd", "median"), inside_share=("inside_any", "mean")).reset_index()
        s = s.merge(info, on="PWSID"); s["run"] = run; s["dT"] = dT
        s["supply_usgs_mgd"] = s.pws_pop * s.gpcd / 1e6; s["supply_proxy_mgd"] = s.pws_pop * 380 / L_PER_MG
        s["ratio_mean"] = s.mean_mgd / s.supply_usgs_mgd; s["ratio_peak"] = s.peak_mgd / s.supply_usgs_mgd; s["ratio_mean_proxy"] = s.mean_mgd / s.supply_proxy_mgd
        # exceedance: fraction of days on which coincident demand exceeds 25 % and 100 % of supply
        ex25 = []; ex100 = []
        for _, r in s.iterrows():
            gg = g[g.PWSID == r.PWSID]; n = gg.groupby("cell").size(); arr = np.zeros(NDAYS)
            for c, k in n.items(): arr += k * D[c][f"dW_water_priority_{dT}"][:NDAYS]
            arr = arr * IT_KW / L_PER_MG; ex25.append(float(np.mean(arr > 0.25 * r.supply_usgs_mgd))); ex100.append(float(np.mean(arr > r.supply_usgs_mgd)))
        s["days_over_25pct"] = ex25; s["days_over_100pct"] = ex100
        urows.append(s)
util = pd.concat(urows, ignore_index=True); util.to_csv(f"{RES}/utilities_all_runs.csv", index=False)
usum = util.groupby(["run", "dT"]).agg(systems=("PWSID", "size"), n_mean_ge25=("ratio_mean", lambda v: int((v >= 0.25).sum())), n_peak_ge25=("ratio_peak", lambda v: int((v >= 0.25).sum())),
                                        n_peak_ge50=("ratio_peak", lambda v: int((v >= 0.5).sum())), n_peak_ge100=("ratio_peak", lambda v: int((v >= 1.0).sum())), n_mean_ge25_proxy=("ratio_mean_proxy", lambda v: int((v >= 0.25).sum()))).reset_index()
usum.to_csv(f"{RES}/utilities_summary.csv", index=False); print("utilities done", round(time.time() - t0), "s")

# ---------------------------------------------------------------- 6. grid water intensity by state (USGS thermoelectric consumption / EIA generation)
eia = pd.read_excel(f"{EXT}/eia/annual_generation_state.xlsx", header=1)
eia.columns = ["year", "st", "producer", "source", "mwh"]
tot = eia[(eia.producer == "Total Electric Power Industry")]
gen15 = tot[(tot.year == 2015) & (tot.source == "Total")].set_index("st").mwh
gen24 = tot[(tot.year == 2024) & (tot.source == "Total")].set_index("st").mwh
nonthermal = ["Hydroelectric Conventional", "Wind", "Solar Thermal and Photovoltaic", "Pumped Storage"]
th15 = gen15 - tot[(tot.year == 2015) & tot.source.isin(nonthermal)].groupby("st").mwh.sum().reindex(gen15.index).fillna(0)
th24 = gen24 - tot[(tot.year == 2024) & tot.source.isin(nonthermal)].groupby("st").mwh.sum().reindex(gen24.index).fillna(0)
pt = usgs.groupby("STATE").agg(cu_mgd=("PT-CUTot", "sum"), power_gwh=("PT-Power", "sum"))
grid = pd.DataFrame({"cu_mgd": pt.cu_mgd, "pt_gwh": pt.power_gwh, "gen15_mwh": gen15, "gen24_mwh": gen24, "th15_mwh": th15, "th24_mwh": th24})
grid["k_2015_L_per_kWh"] = grid.cu_mgd * L_PER_MG * 365 / (grid.gen15_mwh * 1000)
grid["k_thermo_L_per_kWh"] = grid.cu_mgd * L_PER_MG * 365 / (grid.pt_gwh * 1e6)
grid["k_2024_L_per_kWh"] = grid.k_thermo_L_per_kWh * grid.th24_mwh / grid.gen24_mwh
grid.to_csv(f"{RES}/grid_water_intensity.csv")
ST_ABBR = sites.dropna(subset=["STUSPS"]).drop_duplicates("state").set_index("state").STUSPS.to_dict()
# per-cell total-water-optimal design with the state k (2015 and 2024)
trows = []
kmap = {k: grid.k_2015_L_per_kWh.get(ST_ABBR[s], np.nan) for s, k in [(s, s) for s in sites.state.unique()]}
k24 = {s: grid.k_2024_L_per_kWh.get(ST_ABBR[s], np.nan) for s in sites.state.unique()}
cell_state = sites.groupby("cell").state.agg(lambda v: v.mode().iloc[0])
for c in cells:
    z = np.load(f"{MOD}/{c}.npz"); PUE = z["PUE_0"].astype(float); WUE = z["WUE_0"].astype(float)
    s = cell_state[c]; r = dict(cell=c, state=s)
    for lab, kk in (("k15", kmap.get(s)), ("k24", k24.get(s))):
        if kk is None or not np.isfinite(kk): continue
        mode, pue, wue = M.select(PUE, WUE, "total_water", kk)
        r[f"WUE_tw_{lab}"] = float(np.mean(wue)); r[f"PUE_tw_{lab}"] = float(np.mean(pue)); r[f"k_{lab}"] = kk
        for m in range(4): r[f"occ{m}_tw_{lab}"] = float(np.mean(mode == m))
    for d in ("water_priority", "water_free"):
        mode, pue, wue = M.select(PUE, WUE, d); r[f"WUE_{d}"] = float(np.mean(wue)); r[f"PUE_{d}"] = float(np.mean(pue))
    # total water per kWh_IT (direct + grid water of all electricity) for each design at k15
    kk = kmap.get(s)
    if kk is not None and np.isfinite(kk):
        for d in ("water_priority", "water_free"): r[f"total_{d}"] = r[f"WUE_{d}"] + kk * r[f"PUE_{d}"]
        r["total_tw_k15"] = r["WUE_tw_k15"] + kk * r["PUE_tw_k15"]
    # peak cooling electricity (kW per kW_IT) for the water-priority design
    modewp, puewp, _ = M.select(PUE, WUE, "water_priority"); r["peak_cool_frac_wp"] = float(np.nanmax(puewp) - 1.0); r["p99_cool_frac_wp"] = float(np.nanquantile(puewp, 0.99) - 1.0)
    trows.append(r)
tw = pd.DataFrame(trows); tw.to_csv(f"{RES}/cells_total_water.csv", index=False)
print("grid/total-water done", round(time.time() - t0), "s")

# ---------------------------------------------------------------- 7. headline JSON
H = {}
hi50 = sites[sites.run == "higher_50"]; mo50 = sites[sites.run == "moderate_50"]
for name, df in (("higher_50", hi50), ("moderate_50", mo50)):
    H[name] = dict(sites=len(df), cons_mgy_phys=float(df.cons_mgy_water_priority_0.sum()), cons_mgy_im3=float(df.im3_cons_mgy.sum()), cons_mgy_im3_allevap=float(df.im3_evap_cons_mgy.sum()),
                   cool_gwh_phys=float(df.cool_gwh_water_priority_0.sum()), cool_gwh_im3=float(df.im3_cool_mwh.sum() / 1000), WUE_mean=float(df.WUE_wp.mean()), WUE_min=float(df.WUE_wp.min()), WUE_max=float(df.WUE_wp.max()),
                   PUE_mean=float(df.PUE_wp.mean()), pkday_median=float(df.pkday_water_priority_0.median()), pkday_min=float(df.pkday_water_priority_0.min()), pkday_max=float(df.pkday_water_priority_0.max()),
                   exchange_median=float(df.exchange_L_per_kWh.median()), cons_mgy_plus2=float(df.cons_mgy_water_priority_2.sum()), occ_free=float(df.occ0_water_priority_0.mean()), occ_evap=float(df.occ1_water_priority_0.mean()), occ_chill=float(df.occ2_water_priority_0.mean()))
H["utilities"] = usum.to_dict(orient="records")
H["national_peaks"] = natpk.to_dict(orient="records")
json.dump(H, open(f"{RES}/headline.json", "w"), indent=1)
print(json.dumps(H["higher_50"], indent=1)); print("ALL DONE", round(time.time() - t0), "s")
