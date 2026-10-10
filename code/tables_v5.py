"""numbers_v5.json for the revised manuscript: every number quoted in the text, tables and legends is computed here from the
results of infra_v5.py, analysis.py and validate_v5.py. Also writes the state tables and the siting summary."""
import os, json, numpy as np, pandas as pd
import model_ext as ME; R = ME.R
def wq_w(vals, q, w):
    vals = np.asarray(vals, float); w = np.asarray(w, float); o = np.argsort(vals); cw = np.cumsum(w[o]) / w.sum(); return float(np.interp(q, cw, vals[o]))
def cop_check():
    prm = dict(R.DEFAULT); K = 273.15; Ta = 35.0; Tin = Ta + 6.0
    cop = lambda T: float(R.cop_fun(np.array([T]), np.array([T]), prm)[0])
    carnot = lambda Tcd: (prm['T_ite_supply'] - 2.0 + K) / (Tcd - (prm['T_ite_supply'] - 2.0))
    T24 = min(Tin + 24.0, 63.0); prm_l = dict(prm); prm_l['cop_model'] = 'lift'; copl = lambda T: float(R.cop_fun(np.array([T]), np.array([T]), prm_l)[0])
    out = dict(T_air=Ta, T_in=Tin, T_out=Tin + 6.0, T_out_24K=T24, cop_entering=cop(Tin), cop_leaving=cop(Tin + 6.0), cop_24K_leaving=cop(T24), cop_lift_6K=copl(Tin + 6.0), cop_lift_24K=copl(T24), T_chw=float(prm['T_ite_supply']))
    out['carnot_6K'] = carnot(Tin + 6.0 + 2.0); out['carnot_24K'] = carnot(T24 + 2.0)
    out['carnot_frac_6K'] = out['cop_leaving'] / out['carnot_6K']; out['carnot_frac_24K'] = out['cop_24K_leaving'] / out['carnot_24K']; out['carnot_frac_entering_6K'] = out['cop_entering'] / out['carnot_6K']
    Tw = float(prm['T4_design']); out.update(T_wet_in=Tw, cop_wet_entering=cop(Tw), cop_wet_6K_leaving=cop(Tw + 6.0), cop_wet_24K_leaving=cop(Tw + 24.0), T_wet_24K_out=Tw + 24.0, T_wet_6K_out=Tw + 6.0)
    out['carnot_frac_lo'] = 0.45; out['carnot_frac_hi'] = 0.55; out['cop_band_lo_6K'] = 0.45 * out['carnot_6K']; out['cop_band_hi_6K'] = 0.55 * out['carnot_6K']
    return out
BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; RES = EXT + "/results"; R3 = EXT + "/results_v5"; MOD = EXT + "/model_power"; MODE = EXT + "/model"
BUILD = os.path.dirname(os.path.abspath(__file__))
IT_KW = 36000.0; L_PER_MG = 3.785411784e6; ML_PER_MG = 3.785411784; MU = 1.25
sites = pd.read_csv(f"{R3}/sites_infra_v5.csv", dtype={"PWSID": str}); sy = pd.read_csv(f"{R3}/storage_yield_cells_v5.csv")
D = {d: sy[sy.design == d].set_index("cell") for d in sy.design.unique()}; sy0 = D["water_priority"]; syh = D["hybrid"]
summ = pd.read_csv(f"{MOD}/_summary.csv").set_index("cell"); alloc = pd.read_csv(f"{R3}/siting_v5.csv"); pc = pd.read_csv(f"{R3}/peak_coincidence_v5.csv")
util = pd.read_csv(f"{R3}/utilities_v5.csv", dtype={"PWSID": str}); usum = pd.read_csv(f"{R3}/utilities_summary_v5.csv"); lcs = pd.read_csv(f"{R3}/least_cost_cells_v5.csv").set_index("cell")
lcg = pd.read_csv(f"{R3}/least_cost_grid_v5.csv"); mc = pd.read_csv(f"{R3}/least_cost_montecarlo_v5.csv"); oat = pd.read_csv(f"{R3}/least_cost_sensitivity_v5.csv")
nat = pd.read_csv(f"{RES}/national_totals.csv"); natpk = pd.read_csv(f"{RES}/national_coincident_peaks.csv"); H = json.load(open(f"{R3}/headline_v5.json"))
isd = pd.read_csv(f"{R3}/validation_isd_summary.csv"); isdst = pd.read_csv(f"{R3}/validation_isd_stations.csv"); fac = pd.read_csv(f"{R3}/validation_facilities.csv").set_index("case")
hi = sites[sites.run == "higher_50"].copy(); w = hi.groupby("cell").size(); cells = w.index.values; wts = w.values.astype(float)
def wq(vals, q):
    vals = np.asarray(vals, float); keep = np.isfinite(vals); vals = vals[keep]; ww = wts[keep]; o = np.argsort(vals); cw = np.cumsum(ww[o]) / ww.sum(); return float(np.interp(q, cw, vals[o]))
def wm(vals):
    v = np.asarray(vals, float); m = np.isfinite(v); return float((v[m] * wts[m]).sum() / wts[m].sum())
def col(src, c): return src.loc[cells, c].values
# ---- state tables (Higher 50 %)
pcs = pc[(pc.run == "higher_50") & (pc.level == "state")].set_index("name"); a25 = alloc[(alloc.run == "higher_50") & (alloc.headroom_def == "mdd") & (alloc.h == 0.25) & (alloc.threshold == 10000) & (alloc.design == "water_priority")]
rows = []
for st, g in hi.groupby("state"):
    sn = a25[(a25.regime == "no_storage") & (a25.state == st)]; ss = a25[(a25.regime == "storage") & (a25.state == st)]; p = pcs.loc[st] if st in pcs.index else None
    rows.append(dict(state=st, sites=len(g), cons_mgy=float(summ.loc[g.cell, "WUE_water_priority_0"].sum() * IT_KW * 8760 / L_PER_MG), im3_cons_mgy=float(g.im3_cons_mgy.sum()),
                     pkday=float(sy0.loc[g.cell, "peak_day_factor"].mean()), pk_hy=float(syh.loc[g.cell, "peak_day_factor"].mean()), coincident=float(sy0.loc[g.cell, "coincident_factor"].mean()),
                     storage_days=float(sy0.loc[g.cell, "S_days_f1.0"].mean()), storage_ml_100mw=float(np.mean(sy0.loc[g.cell, "S_days_f1.0"].values * sy0.loc[g.cell, "mean_L_per_kW_day"].values * MU * 1e5 / 1e6)),
                     serv_kept_pct=100 * float(sn.kept_own.sum()) / len(g) if len(sn) else np.nan, serv_nostorage_pct=100 * float(sn.servable.sum()) / len(g) if len(sn) else np.nan, serv_storage_pct=100 * float(ss.servable.sum()) / len(g) if len(ss) else np.nan,
                     grid_cool_per_gw=float(p.grid_cool_mw / p.it_mw * 1000) if p is not None and np.isfinite(p.grid_cool_mw) else np.nan, grid_add_per_gw=float(p.grid_add_S0 / p.it_mw * 1000) if p is not None and np.isfinite(p.grid_add_S0) else np.nan,
                     grid_add_S7_per_gw=float(p.grid_add_S7 / p.it_mw * 1000) if p is not None and np.isfinite(p.grid_add_S7) else np.nan, eia_region=p.eia_region if p is not None else "", peak_hour_utc=int(p.peak_hour_utc) if p is not None else -1,
                     wue=float(summ.loc[g.cell, "WUE_water_priority_0"].mean()), wue_hy=float(syh.loc[g.cell, "annual_L_per_kW"].mean() / 8760), pue=float(summ.loc[g.cell, "PUE_water_priority_0"].mean()), storage_days_plus2=float(D["water_priority_plus2"].loc[g.cell, "S_days_f1.0"].mean()), pkday_plus2=float(D["water_priority_plus2"].loc[g.cell, "peak_day_factor"].mean())))
st_all = pd.DataFrame(rows).sort_values("sites", ascending=False); st_all.to_csv(f"{R3}/state_table_all_v5.csv", index=False); st_all.head(20).to_csv(f"{R3}/state_table_v5.csv", index=False)
ss_ = alloc.groupby(["run", "headroom_def", "h", "regime", "threshold", "design"]).agg(sites=("sites", "sum"), kept_own=("kept_own", "sum"), moved=("moved", "sum"), unservable=("unservable", "sum"), unservable_mw=("unservable_mw", "sum")).reset_index(); ss_.to_csv(f"{R3}/siting_summary_v5.csv", index=False)
def sit(run="higher_50", hdef="mdd", h=0.25, thr=10000, des="water_priority"):
    g = ss_[(ss_.run == run) & (ss_.headroom_def == hdef) & (ss_.h == h) & (ss_.threshold == thr) & (ss_.design == des)].set_index("regime"); return {k: dict(sites=int(g.loc[k, "sites"]), kept=int(g.loc[k, "kept_own"]), moved=int(g.loc[k, "moved"]), unserv=int(g.loc[k, "unservable"])) for k in g.index}
# ---- numbers for the text
nt = lambda g, d, dT: nat[(nat.run == g + "_50") & (nat.design == d) & (nat.dT == dT)].iloc[0]; pkn = lambda g, dT: natpk[(natpk.run == g + "_50") & (natpk.dT == dT)].iloc[0]
pcn = pc[(pc.run == "higher_50") & (pc.level == "nation")].iloc[0]; pcm = pc[(pc.run == "moderate_50") & (pc.level == "nation")].iloc[0]; pcr = pc[(pc.run == "higher_50") & (pc.level == "region")].set_index("name")
u0 = util[util.run == "higher_50"]; usr = usum[usum.run.str.startswith("higher")]
exch = col(summ, "WUE_water_priority_0") / (col(summ, "PUE_water_free_0") - col(summ, "PUE_water_priority_0"))
erows = []
if os.path.exists(f"{MODE}/_summary.csv"):
    se = pd.read_csv(f"{MODE}/_summary.csv").set_index("cell"); link = pd.read_csv(f"{EXT}/sites_all_runs.csv")[["cell", "cell_era"]].drop_duplicates()
    link = link[link.cell_era.isin(se.index) & link.cell.isin(summ.index)].drop_duplicates("cell_era")
    for metric, c_ in (("WUE (L per kWh)", "WUE_water_priority_0"), ("Cooling PUE", "PUE_water_priority_0"), ("Peak-day factor", "pkday_water_priority_0"), ("99th-percentile-day factor", "p99day_water_priority_0"), ("Free-cooling share", "occ0_water_priority_0"), ("Evaporative share", "occ1_water_priority_0"), ("Chiller share", "occ2_water_priority_0"), ("PUE, water-free design", "PUE_water_free_0")):
        a = summ.loc[link.cell, c_].values; b = se.loc[link.cell_era, c_].values; m = np.isfinite(a) & np.isfinite(b)
        erows.append(dict(metric=metric, mean_power=float(a[m].mean()), mean_era=float(b[m].mean()), mae=float(np.mean(np.abs(a[m] - b[m]))), r=float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan, n=int(m.sum())))
era = pd.DataFrame(erows); era.to_csv(f"{R3}/validation_era5_v5.csv", index=False)
isdd = isd.set_index("metric")
N = dict(
    sites_higher=int(len(hi)), cells=int(len(cells)), locations=int(sites.drop_duplicates(["xk", "yk"]).shape[0]), MU=MU, COC=5, T_THR=29.4,
    cons_higher=float(nt("higher", "water_priority", 0).cons_mgy), cons_higher_im3=float(nt("higher", "water_priority", 0).im3_cons_mgy), cons_low=float(nt("low", "water_priority", 0).cons_mgy), cons_moderate=float(nt("moderate", "water_priority", 0).cons_mgy), cons_high=float(nt("high", "water_priority", 0).cons_mgy),
    cons_higher_plus2=float(nt("higher", "water_priority", 2).cons_mgy), cons_higher_plus1=float(nt("higher", "water_priority", 1).cons_mgy), cool_higher=float(nt("higher", "water_priority", 0).cool_gwh), cool_higher_im3=float(nt("higher", "water_priority", 0).im3_cool_gwh), cool_higher_wf=float(nt("higher", "water_free", 0).cool_gwh),
    wue_wmean=wm(col(summ, "WUE_water_priority_0")), wue_p10=wq(col(summ, "WUE_water_priority_0"), 0.1), wue_p90=wq(col(summ, "WUE_water_priority_0"), 0.9), wue_min=float(col(summ, "WUE_water_priority_0").min()), wue_max=float(col(summ, "WUE_water_priority_0").max()),
    pue_wmean=wm(col(summ, "PUE_water_priority_0")), pue_wf_wmean=wm(col(summ, "PUE_water_free_0")), free_share=wm(col(summ, "occ0_water_priority_0")), evap_share=wm(col(summ, "occ1_water_priority_0")), chill_share=wm(col(summ, "occ2_water_priority_0")),
    wet_intensity=wm(col(summ, "WUE_water_priority_0") / np.maximum(col(summ, "occ1_water_priority_0") + col(summ, "occ2_water_priority_0"), 1e-6)),
    hy_wue_wmean=wm(col(syh, "annual_L_per_kW") / 8760), hy_wue_p10=wq(col(syh, "annual_L_per_kW") / 8760, 0.1), hy_wue_p90=wq(col(syh, "annual_L_per_kW") / 8760, 0.9), hy_water_hours=wm(col(syh, "water_days_per_year")) * 24 if False else H["hy_water_hours"], hy_water_days=wm(col(syh, "water_days_per_year")),
    hy25_wue=wm(col(D["hybrid25"], "annual_L_per_kW") / 8760), hy35_wue=wm(col(D["hybrid35"], "annual_L_per_kW") / 8760),
    pk_wmean=wm(col(sy0, "peak_day_factor")), pk_p10=wq(col(sy0, "peak_day_factor"), 0.1), pk_p90=wq(col(sy0, "peak_day_factor"), 0.9), pk_min=float(col(sy0, "peak_day_factor").min()), pk_max=float(col(sy0, "peak_day_factor").max()), p99_wmean=wm(col(sy0, "p99_day_factor")),
    pk_hy_wmean=wm(col(syh, "peak_day_factor")), pk_hy_p10=wq(col(syh, "peak_day_factor"), 0.1), pk_hy_p90=wq(col(syh, "peak_day_factor"), 0.9), pk_hy_min=float(np.nanmin(col(syh, "peak_day_factor"))), pk_hy_max=float(np.nanmax(col(syh, "peak_day_factor"))), pk_hy_median=wq(col(syh, "peak_day_factor"), 0.5),
    pk_hy25_wmean=wm(col(D["hybrid25"], "peak_day_factor")), pk_hy35_wmean=wm(col(D["hybrid35"], "peak_day_factor")), monthpk_wp=wm(col(sy0, "month_peak_factor")), monthpk_hy=wm(col(syh, "month_peak_factor")), pk_hy_share_3to10=wm(((col(syh, "peak_day_factor") >= 3) & (col(syh, "peak_day_factor") <= 10)).astype(float)), pk_hy_share_ge3=wm((col(syh, "peak_day_factor") >= 3).astype(float)),
    coincident_wmean=wm(col(sy0, "coincident_factor")), coincident_share=wm(col(sy0, "peak_on_hottest_day_share")),
    nat_pk_higher=float(pkn("higher", 0).pkfactor), nat_mean_mgd_high=float(pkn("high", 0).mean_mgd) * MU, nat_mean_mgd_moderate=float(pkn("moderate", 0).mean_mgd) * MU, nat_peak_mgd_high=float(pkn("high", 0).peak_mgd) * MU, nat_peak_mgd=float(pkn("higher", 0).peak_mgd) * MU, nat_mean_mgd=float(pkn("higher", 0).mean_mgd) * MU, nat_sum_site_peaks=float(pkn("higher", 0).sum_site_peaks_mgd) * MU, nat_sum_site_peaks_hy=float(hi.peak_mgd_hy.sum()), nat_mean_hy_mgd=float(hi.mean_mgd_hy.sum()), nat_sum_site_peaks_evap_mgd=float(pkn("higher", 0).sum_site_peaks_mgd), nat_peak_evap_mgd=float(pkn("higher", 0).peak_mgd), nat_mean_evap_mgd=float(pkn("higher", 0).mean_mgd),
    s1_wmean=wm(col(sy0, "S_days_f1.0")), s1_p10=wq(col(sy0, "S_days_f1.0"), 0.1), s1_p90=wq(col(sy0, "S_days_f1.0"), 0.9), s1_min=float(col(sy0, "S_days_f1.0").min()), s1_max=float(col(sy0, "S_days_f1.0").max()), s12_wmean=wm(col(sy0, "S_days_f1.2")), s15_wmean=wm(col(sy0, "S_days_f1.5")),
    s1_ml100_wmean=wm(col(sy0, "S_days_f1.0") * col(sy0, "mean_L_per_kW_day") * MU * 1e5 / 1e6), s1_ml100_p90=wq(col(sy0, "S_days_f1.0") * col(sy0, "mean_L_per_kW_day") * MU * 1e5 / 1e6, 0.9), s1_hy_wmean=wm(col(syh, "S_days_f1.0")),
    s1_plus2_wmean=wm(col(D["water_priority_plus2"], "S_days_f1.0")), pk_plus2_wmean=wm(col(D["water_priority_plus2"], "peak_day_factor")), s1_plus1_wmean=wm(col(D["water_priority_plus1"], "S_days_f1.0")), pk_plus1_wmean=wm(col(D["water_priority_plus1"], "peak_day_factor")),
    short_f1_S0=wm(col(sy0, "short_f1.0_S0")) * 365, short_f1_S3=wm(col(sy0, "short_f1.0_S3")) * 365, short_f1_S7=wm(col(sy0, "short_f1.0_S7")) * 365, short_f1_S30=wm(col(sy0, "short_f1.0_S30")) * 365, short_f1_S0_med=wq(col(sy0, "short_f1.0_S0") * 365, 0.5), short_f1_S7_med=wq(col(sy0, "short_f1.0_S7") * 365, 0.5),
    short_f08_S30=wm(col(sy0, "short_f0.8_S30")) * 365, short_f125_S0=wm(col(sy0, "short_f1.25_S0")) * 365, short_f125_S7=wm(col(sy0, "short_f1.25_S7")) * 365, short_res7=wm(col(sy0, "short_f1.0_S7_reserve7")) * 365, short_res14=wm(col(sy0, "short_f1.0_S7_reserve14")) * 365, short_res30_S30=wm(col(sy0, "short_f1.0_S30_reserve30")) * 365,
    pen_f1_S0_pct=wm(col(sy0, "pen_f1.0_S0")) / 8760 * 100, pen_f1_S7_pct=wm(col(sy0, "pen_f1.0_S7")) / 8760 * 100, pen_f1_S7_med=wq(col(sy0, "pen_f1.0_S7") / 8760 * 100, 0.5), pen_f1_S0_med=wq(col(sy0, "pen_f1.0_S0") / 8760 * 100, 0.5), pen_f08_S30_pct=wm(col(sy0, "pen_f0.8_S30")) / 8760 * 100, pen_res7_pct=wm(col(sy0, "pen_f1.0_S7_reserve7")) / 8760 * 100,
    exch_median=wq(exch, 0.5), exch_p10=wq(exch, 0.1), exch_p90=wq(exch, 0.9),
)
S = sit(); S10 = sit(h=0.10); S50 = sit(h=0.50); SA = sit(hdef="avg"); S33 = sit(thr=3300); SH = sit(des="hybrid")
def shares(s): n = s["no_storage"]["sites"]; return dict(ns_kept=s["no_storage"]["kept"], ns_moved=s["no_storage"]["moved"], ns_unserv=s["no_storage"]["unserv"], s_kept=s["storage"]["kept"], s_moved=s["storage"]["moved"], s_unserv=s["storage"]["unserv"], n=n)
N.update(sit25=shares(S), sit10=shares(S10), sit50=shares(S50), sit_avg25=shares(SA), sit3300=shares(S33), sit_hybrid=shares(SH),
         sit_mod=shares(sit(run="moderate_50")), sit_low=shares(sit(run="low_50")), sit_high=shares(sit(run="high_50")),
         attributed_share=float(hi.attributed.mean()), attributed_share_3300=float(hi.attributed_3300.mean()), reanalysis_cov=float((hi[hi.attributed].drop_duplicates("PWSID").supply_source == "reanalysis").mean()), n_screened=int((sites.drop_duplicates("PWSID").supply_source != "reanalysis").sum()),
         n_systems=int(u0.PWSID.nunique()), sys_peak25=int((u0.ratio_peak_mdd >= 0.25).sum()), sys_mean25=int((u0.ratio_mean_mdd >= 0.25).sum()), sys_peak100=int((u0.ratio_peak_mdd >= 1).sum()), sys_mean100=int((u0.ratio_mean_mdd >= 1).sum()), sys_peak25_hy=int((u0.ratio_peak_mdd_hy >= 0.25).sum()),
         sys_peak25_runs=[int(v) for v in usr.n_peak_ge25], sys_mean25_runs=[int(v) for v in usr.n_mean_ge25], sys_peak25_hy_runs=[int(v) for v in usr.n_peak_ge25_hy], sys_peak25_avgdef=int((u0.ratio_peak_avg >= 0.25).sum()), sys_mean25_avgdef=int((u0.ratio_mean_avg >= 0.25).sum()),
         sys_pop_q=[float(u0.pws_pop.quantile(q)) for q in (0.1, 0.5, 0.9)], top15=u0.sort_values("ratio_peak_mdd", ascending=False).head(15)[["PWS_Name", "pws_state", "pws_pop", "sites", "supply_avg_mgd", "supply_mdd_mgd", "supply_source", "mean_mgd", "peak_mgd", "ratio_mean_mdd", "ratio_peak_mdd", "days_over_25pct_mdd"]].to_dict(orient="records"),
         mdd_over_avg_median=float((u0.supply_mdd_mgd / u0.supply_avg_mgd).median()), gpcd_median=float((u0.supply_avg_mgd * 1e6 / u0.pws_pop).median()),
         us_it_mw=float(pcn.it_mw), us_cool_mean=float(pcn.cool_mean_mw), us_cool_peak=float(pcn.cool_peak_mw), us_peak_time=str(pcn.peak_time_utc), us_add_S0=float(pcn.add_at_cool_peak_S0), us_wf_peak=float(pcn.wf_peak_mw), us_peak_per_gw=float(pcn.peak_per_gw_it), us_add_per_gw=float(pcn.add_per_gw_it),
         us_grid_cool=float(pcn.grid_cool_mw), us_grid_add_S0=float(pcn.grid_add_S0), us_grid_add_S0_max=float(pcn.grid_add_S0_max), us_grid_add_S3=float(pcn.grid_add_S3), us_grid_add_S7=float(pcn.grid_add_S7), us_grid_add_S30=float(pcn.grid_add_S30), us_grid_add_res7=float(pcn.grid_add_reserve7), us_grid_add_res14=float(pcn.grid_add_reserve14), us_grid_add_res30=float(pcn.grid_add_reserve30_S30), us_grid_add_f125=float(pcn.grid_add_f125_S7),
         us_grid_hy=float(pcn.grid_hy_mw), us_grid_add_S0_per_gw=float(pcn.grid_add_S0_per_gw), us_hy_peak=float(pcn.hy_peak_mw), usm_grid_add_S0_per_gw=float(pcm.grid_add_S0_per_gw), usm_peak_per_gw=float(pcm.peak_per_gw_it), usm_add_per_gw=float(pcm.add_per_gw_it),
         regions=pcr[["it_mw", "grid_cool_mw", "grid_add_S0", "grid_add_S7", "grid_add_reserve7", "grid_add_S30", "grid_hy_mw", "grid_add_S0_per_gw", "peak_hour_utc", "cool_peak_mw", "add_at_cool_peak_S0", "grida_add_S0_per_gw", "gridw_add_S0_per_gw", "grida_cool_mw", "gridw_cool_mw", "grida_add_S0", "gridw_add_S0", "lag_h", "cool_modal_hour_utc", "grid_modal_hour_utc", "grid_add_S0_opt", "grid_add_S0_lo", "grid_add_S0_cons", "cool_top50_in_window", "cool_diurnal_amp_pct", "grid_add_f125_S7"]].to_dict(orient="index"),
         us_grida_cool=float(pcn.grida_cool_mw), us_grida_add_S0=float(pcn.grida_add_S0), us_grida_add_S7=float(pcn.grida_add_S7), us_gridw_cool=float(pcn.gridw_cool_mw), us_gridw_add_S0=float(pcn.gridw_add_S0), us_gridw_add_S7=float(pcn.gridw_add_S7),
         **{f"us_grid_add_S0_{v}": float(pcn[f"grid_add_S0_{v}"]) for v in ("opt", "lo", "cons")}, **{f"us_grid_add_S7_{v}": float(pcn[f"grid_add_S7_{v}"]) for v in ("opt", "lo", "cons")},
         us_cool_window_share=float(pcn.cool_top50_in_window), us_cool_diurnal_amp=float(pcn.cool_diurnal_amp_pct), lc_win=H["lc_win"], anomalies=H["anomalies"],
         wue_min_state=str(hi.groupby("cell").state.first()[summ.loc[cells, "WUE_water_priority_0"].idxmin()]), wue_max_state=str(hi.groupby("cell").state.first()[summ.loc[cells, "WUE_water_priority_0"].idxmax()]),
         isd_timing=dict(within2h=int((np.abs(((isdst.cooltop50_hour_station - isdst.cooltop50_hour_power + 12) % 24) - 12) <= 2).sum()), n=int(len(isdst)), amp_station_mean=float(isdst.cool_diurnal_amp_pct_station.mean()), amp_power_mean=float(isdst.cool_diurnal_amp_pct_power.mean()), amp_station_max=float(isdst.cool_diurnal_amp_pct_station.max()), amp_station_min=float(isdst.cool_diurnal_amp_pct_station.min())),
         cop_check=cop_check(),
         phoenix_diff_pct=float(100 * (fac.loc["V-Phoenix", "monthpk_wp"] - 1.46) / 1.46), extra_cooler_min=float(H["extra_cooler_min"]), 
         sit_coc3=shares(sit(des="water_priority_coc3")), sit_coc6=shares(sit(des="water_priority_coc6")), sys_coc={k: {c: int(usum[usum.run == f"higher_50 ({k})"].iloc[0][c]) for c in ("n_peak_ge25", "n_mean_ge25", "n_peak_ge100", "n_mean_ge100")} for k in ("coc3", "coc6")},
         lag=H["lag"], n_regions=int(len(pcr)), n_lead=int(sum(1 for r in H["lag"] if r["lag_h"] < 0)), n_lag=int(sum(1 for r in H["lag"] if r["lag_h"] > 0)), n_same=int(sum(1 for r in H["lag"] if r["lag_h"] == 0)),
         dry_capex_wmean=float(H["dry_capex_wmean"]), cells_extra_coolers=int(H["cells_extra_coolers"]), extra_cooler_max=float(H["extra_cooler_max"]), breakeven_median=float(H["breakeven_median"]), breakeven_share_never=float(H["breakeven_share_never"]) * 100, dry_capex_wmean_opt=float(H["dry_capex_wmean_opt"]), dry_capex_wmean_lo=float(H["dry_capex_wmean_lo"]), cells_extra_coolers_opt=int(H["cells_extra_coolers_opt"]), extra_cooler_max_opt=float(H["extra_cooler_max_opt"]), mc_share_near_zero=float(H["mc_share_near_zero"]) * 100,
         gpcd_state=H["gpcd_state"], gpcd_nat=float(H["gpcd_nat"]), gpcd_min=float(H["gpcd_min"]), dry_capex_base=25.0, dry_cooler_cost=150.0,
         **{f"pen_f1_S7_{v}_pct": wm(col(sy0, f"pen_f1.0_S7_{v}")) / 8760 * 100 for v in ("opt", "lo", "cons")}, **{f"pen_f1_S0_{v}_pct": wm(col(sy0, f"pen_f1.0_S0_{v}")) / 8760 * 100 for v in ("opt", "lo", "cons")},
         thr={k: dict(wue=wm(col(D[d], "annual_L_per_kW") / 8760), water_hours=wm(col(D[d], "water_hours_per_year")), pk=wm(col(D[d], "peak_day_factor")), pk_med=wq(col(D[d], "peak_day_factor"), 0.5), pk_p10=wq(col(D[d], "peak_day_factor"), 0.1), pk_p90=wq(col(D[d], "peak_day_factor"), 0.9), monthpk=wm(col(D[d], "month_peak_factor")), s1=wm(col(D[d], "S_days_f1.0")), share_3to10=wm(((col(D[d], "peak_day_factor") >= 3) & (col(D[d], "peak_day_factor") <= 10)).astype(float))) for k, d in (("25", "hybrid25"), ("29.4", "hybrid"), ("35", "hybrid35"))},
         lc_saving_tank_median=float(H["lc_saving_tank_median"]), lc_saving_pond_median=wq(lcs.loc[cells].saving_pond_frac.values, 0.5) * 100, lc_saving_tank_p10=wq(lcs.loc[cells].saving_tank_frac.values, 0.1) * 100, lc_saving_tank_p90=wq(lcs.loc[cells].saving_tank_frac.values, 0.9) * 100,
         lc_win_nodry_share=(lambda lcg, win: float((wts[win] * (np.array([lcg.get((c, round(float(lcs.loc[c].best_f), 6), int(lcs.loc[c].best_S_days)), np.nan) for c in cells[win]]) == 0)).sum() / wts[win].sum()) * 100 if win.any() else 0.0)(
             pd.read_csv(f"{R3}/least_cost_grid_v5.csv").assign(f=lambda d: d.f.round(6)).set_index(["cell", "f", "S_days"]).shortfall_frac.to_dict(), (lcs.loc[cells].best_S_days > 0).values),
         lc_win_saving_median=wq_w(lcs.loc[cells].saving_tank_frac.values[(lcs.loc[cells].best_S_days > 0).values], 0.5, wts[(lcs.loc[cells].best_S_days > 0).values]) * 100 if (lcs.loc[cells].best_S_days > 0).any() else 0.0,
         lc_share_storage=float(H["lc_share_storage"]), lc_share_positive=float(H["lc_share_positive"]), lc_share_storage_pond=float((wts * (lcs.loc[cells].best_S_days_pond > 0)).sum() / wts.sum()) * 100, lc_best_f_median=wq(lcs.loc[cells].best_f.values, 0.5), lc_best_S_median=wq(lcs.loc[cells].best_S_days.values, 0.5),
         lc_share_f_le1=float((wts * (lcs.loc[cells].best_f <= 1.0)).sum() / wts.sum()) * 100, lc_share_conventional=float((wts * ((lcs.loc[cells].best_S_days == 0) & (lcs.loc[cells].saving_tank_frac <= 1e-9))).sum() / wts.sum()) * 100, lc_breakeven_median=wq(lcs.loc[cells].breakeven_dry_capex_usd_per_kw.values, 0.5),
         mc_median_saving_q=[float(mc.median_saving_pct.quantile(q)) for q in (0.05, 0.5, 0.95)], mc_share_storage_q=[float(mc.share_storage_pct.quantile(q)) for q in (0.05, 0.5, 0.95)], mc_share_positive_q=[float(mc.share_positive_pct.quantile(q)) for q in (0.05, 0.5, 0.95)], mc_draws=int(len(mc)), mc_share_median_negative=float((mc.median_saving_pct < 0).mean()) * 100,
         oat=oat.set_index("case").to_dict(orient="index"), isd=isdd.to_dict(orient="index"), isd_n=int(isdd.loc["WUE_wp", "n"]), isd_cov_min=float(isdst.coverage.min()),
         fac={k: {c: (float(v) if isinstance(v, (int, float, np.floating)) else v) for c, v in r.items()} for k, r in fac.to_dict(orient="index").items()},
         era_n=int(era.n.iloc[0]) if len(era) else 0, era_requested=101, era_modeled=int(pd.read_csv(f"{MODE}/_summary.csv").shape[0]) if os.path.exists(f"{MODE}/_summary.csv") else 0, era=era.to_dict(orient="records"),
         cons_rh2=float(np.nan), s1_rh2_wmean=float(np.nan), pk_rh2_wmean=float(np.nan),
    top_states=st_all.head(5).state.tolist(), st=st_all.set_index("state").to_dict(orient="index"), elec_price=0.0813, elec_price_peak=0.0813 * 1.5, cap_charge=120e3)
if os.path.exists(f"{EXT}/results_v3/warming_constRH.csv"):
    rh = pd.read_csv(f"{EXT}/results_v3/warming_constRH.csv").set_index("cell")
    N["cons_rh2"] = float(nt("higher", "water_priority", 0).cons_mgy * wm(col(rh, "WUE_rh2")) / wm(col(summ, "WUE_water_priority_0"))); N["s1_rh2_wmean"] = wm(col(rh, "S1_rh2")); N["pk_rh2_wmean"] = wm(col(rh, "pk_rh2"))
json.dump(N, open(f"{BUILD}/numbers_v5.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if k not in ("st", "era", "top_states", "oat", "isd", "fac", "regions", "top15")}, indent=1, default=float)[:6000])
