"""Main figures of the revised manuscript (v5). Reads data_ext/results_v5 (and results); writes submission_v5/figures/Fig1-6.png (300 dpi)."""
import os, sys, json, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import shapely, pyproj
from pyogrio.raw import read as raw_read
BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; RES = EXT + "/results"; R3 = EXT + "/results_v5"; MOD = EXT + "/model_power"
OUT = BASEW + "/submission_v5/figures"; os.makedirs(OUT, exist_ok=True)
NE = "C:/Users/workt/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_50m_admin_1_states_provinces_lakes.shp"
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})
C = dict(blue="#2b6cb0", lblue="#9ecae1", orange="#e66101", purple="#5e3c99", teal="#1b9e77", gray="#8c8c8c", lgray="#d9d9d9", dark="#222222", gold="#e6ab02", red="#c0392b")
IT_KW = 36000.0; L_PER_MG = 3.785411784e6; NDAYS = 3652; MU = 1.25; ML_PER_MG = 3.785411784
fwd = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:102003", always_xy=True).transform
N = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers_v5.json")))
sites = pd.read_csv(f"{R3}/sites_infra_v5.csv", dtype={"PWSID": str}); sy = pd.read_csv(f"{R3}/storage_yield_cells_v5.csv"); curves = pd.read_csv(f"{R3}/storage_curves_cells_v5.csv")
alloc = pd.read_csv(f"{R3}/siting_v5.csv"); asites = pd.read_csv(f"{R3}/siting_sites_v5.csv"); pc = pd.read_csv(f"{R3}/peak_coincidence_v5.csv"); prof = json.load(open(f"{R3}/peak_day_profiles_v5.json"))
lcs = pd.read_csv(f"{R3}/least_cost_cells_v5.csv").set_index("cell"); lcg = pd.read_csv(f"{R3}/least_cost_grid_v5.csv"); util = pd.read_csv(f"{R3}/utilities_v5.csv", dtype={"PWSID": str}); mc = pd.read_csv(f"{R3}/least_cost_montecarlo_v5.csv")
sy0 = sy[sy.design == "water_priority"].set_index("cell"); syh = sy[sy.design == "hybrid"].set_index("cell"); sy2 = sy[sy.design == "water_priority_plus2"].set_index("cell"); sy1 = sy[sy.design == "water_priority_plus1"].set_index("cell")
hi = sites[sites.run == "higher_50"].copy(); hi = hi[hi.cell.isin(sy0.index)]; w = hi.groupby("cell").size(); cells = w.index.values; wts = w.values.astype(float)
dates = np.arange("2014-01-01", "2024-01-01", dtype="datetime64[D]")[:NDAYS]
def wq(vals, q):
    vals = np.asarray(vals, float); k = np.isfinite(vals); v = vals[k]; ww = wts[k]; o = np.argsort(v); cw = np.cumsum(ww[o]) / ww.sum(); return float(np.interp(q, cw, v[o]))
meta, idx, geom, fields = raw_read(NE); F = dict(zip(meta["fields"], fields)); st = shapely.from_wkb(geom)
keep = np.array([(i == "US") and (p not in ("AK", "HI", "PR")) for i, p in zip(F["iso_a2"], F["postal"])]); states = shapely.transform(st[keep], lambda c: np.column_stack(fwd(c[:, 0], c[:, 1])))
def paths(geoms):
    out = []
    for g in geoms:
        for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            verts = []; codes = []
            for ring in [p.exterior, *p.interiors]:
                c = np.asarray(ring.coords); verts.append(c); codes += [Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY]
            out.append(Path(np.vstack(verts), codes))
    return out
SP = paths(states); SB = shapely.total_bounds(states)
def basemap(ax):
    ax.add_collection(PathCollection(SP, facecolor="#f5f5f5", edgecolor="#9a9a9a", linewidth=0.4, zorder=1))
    ax.set_xlim(SB[0] - 4e4, SB[2] + 4e4); ax.set_ylim(SB[1] - 4e4, SB[3] + 4e4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
def label(ax, letter, x=-0.04, y=1.06): ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
def title(ax, t): ax.set_title(t, fontsize=7.5, loc="left", pad=3)
def site_xy(df): x, y = fwd(df.lon.values, df.lat.values); return x, y
def cell_label(c):
    g = hi[hi.cell == c].iloc[0]; return f"{g.NAMELSAD}, {g.STUSPS}" if isinstance(g.NAMELSAD, str) else c
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{OUT}/{name}.eps", format="eps", bbox_inches="tight", facecolor="white", pad_inches=0.04); plt.close(fig); print(name)

# ================================================================ Fig 1: storage-yield framework
S1 = sy0.loc[cells, "S_days_f1.0"].values; order = np.argsort(S1); ex_cells = [cells[order[0]], cells[order[len(order) // 2]], cells[order[-1]]]; cols = [C["blue"], C["orange"], C["purple"]]
fig = plt.figure(figsize=(7.2, 6.8)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.40, wspace=0.30)
ax = fig.add_subplot(gs[0, 0]); yr = (dates >= np.datetime64("2023-01-01"))
for c, col in zip(ex_cells, cols):
    W = np.load(f"{MOD}/{c}.npz")["dW_water_priority_0"][:NDAYS] * MU * IT_KW / 1e6
    ax.plot(dates[yr], W[yr], lw=0.5, color=col, alpha=0.75); mm = pd.Series(W[yr], index=pd.DatetimeIndex(dates[yr])).resample("MS").mean()
    ax.step(mm.index, mm.values, where="post", lw=1.4, color=col, label=f"{cell_label(c)} (mean {W.mean():.2f} ML per day, peak {W.max()/W.mean():.1f} x mean)")
ax.set_ylabel("Withdrawal demand of a 36 MW site (ML per day)"); ax.set_xlabel("2023"); ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.legend(loc="upper left", frameon=False, handlelength=1.4, fontsize=5.6)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); label(ax, "a"); title(ax, "Daily demand, one year (thin) and monthly means (steps)")
ax = fig.add_subplot(gs[1, 0]); fr = sorted(curves.f.unique()); cv = curves[curves.design == "water_priority"]
lo = [wq(cv[cv.f == f].set_index("cell").loc[cells, "S_days"].values, 0.1) for f in fr]; hi_ = [wq(cv[cv.f == f].set_index("cell").loc[cells, "S_days"].values, 0.9) for f in fr]; md = [wq(cv[cv.f == f].set_index("cell").loc[cells, "S_days"].values, 0.5) for f in fr]
ax.fill_between(fr, lo, hi_, color=C["lgray"], label="10th to 90th percentile of projected sites"); ax.plot(fr, md, color=C["dark"], lw=1.0, label="Median site")
for c, col in zip(ex_cells, cols):
    v = cv[cv.cell == c].sort_values("f"); ax.plot(v.f, v.S_days, color=col, lw=1.2, marker="o", ms=2.5, label=cell_label(c))
ax.set_xlabel("Allocated capacity (multiple of the mean day)"); ax.set_ylabel("Storage required (days of mean demand)"); ax.set_ylim(0, None); ax.set_xlim(1.0, 2.0); ax.legend(frameon=False, loc="upper right", handlelength=1.2); label(ax, "c"); title(ax, "Storage-yield curves (sequent peak, 2014 to 2023)")
ax = fig.add_subplot(gs[1, 1]); basemap(ax); x, y = site_xy(hi); v = sy0.loc[hi.cell, "S_days_f1.0"].values * sy0.loc[hi.cell, "mean_L_per_kW_day"].values * MU * 1e5 / 1e6
sc = ax.scatter(x, y, c=v, s=5, cmap="viridis", vmin=0, vmax=np.nanpercentile(v, 98), edgecolor="none", zorder=4)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8); cb.set_label("Storage at a mean-day allocation (ML of withdrawal per 100 MW of IT)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
label(ax, "d", x=0.0, y=1.02); title(ax, f"      Higher growth, 50% market gravity (n = {len(hi)} sites)")
ax = fig.add_subplot(gs[0, 1]); YM = 1.0 / 1.8; ax.axvspan(3, 10, ymax=YM, color=C["lgray"], alpha=0.6, label="Operating facilities, Han et al.: 3 to 10"); ax.axvline(4.5, ymax=YM, color=C["orange"], lw=1.2, ls="--", label="Planning factor 4.5, Han et al.")
for src, col, ls, colname, lab in ((sy0, C["blue"], "-", "peak_day_factor", "Always-on evaporative: peak day / mean day"), (sy0, C["purple"], ":", "p99_day_factor", "Always-on: 99th-percentile day / mean day"), (syh, C["teal"], "-", "peak_day_factor", "Hybrid (evaporates above 29.4 °C): peak day / mean day")):
    pf = src.loc[cells, colname].values; k = np.isfinite(pf); o = np.argsort(pf[k]); ax.step(pf[k][o], np.cumsum(wts[k][o]) / wts[k].sum(), where="post", color=col, lw=1.3, ls=ls, label=lab)
ax.axvline(2.21, ymax=YM, color=C["red"], lw=0.8, ls="-.", label="Measured, The Dalles tower zone: 2.21"); ax.axvline(6.45, ymax=YM, color=C["red"], lw=0.8, ls=":", label="Measured, Iowa hybrid campus: 6.45 or more")
ax.set_xscale("log"); ax.set_xlim(1, 40); ax.set_xticks([1, 2, 3, 4.5, 6, 10, 20, 40]); ax.set_xticklabels(["1", "2", "3", "4.5", "6", "10", "20", "40"]); ax.set_ylim(0, 1.8); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlabel("Daily peaking factor (maximum day / mean day)"); ax.set_ylabel("Cumulative share of projected sites")
ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.4, fontsize=5.3); label(ax, "b"); title(ax, "Peaking factors of the two designs against measured values")
save(fig, "Fig1")

# ================================================================ Fig 2: reliability of a capacity allocation with storage and dry backup
SD = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
for f, col in zip((0.8, 1.0, 1.25), (C["orange"], C["blue"], C["purple"])):
    for k, (pre, scale) in enumerate((("short", 365.0), ("pen", 100.0 / 8760.0))):
        md = [wq(sy0.loc[cells, f"{pre}_f{f}_S{s}"].values * scale, 0.5) for s in SD]; lo = [wq(sy0.loc[cells, f"{pre}_f{f}_S{s}"].values * scale, 0.25) for s in SD]; hi_ = [wq(sy0.loc[cells, f"{pre}_f{f}_S{s}"].values * scale, 0.75) for s in SD]
        axs[k + 1].fill_between(SD, lo, hi_, color=col, alpha=0.15); axs[k + 1].plot(SD, md, color=col, marker="o", ms=2.5, lw=1.2, label=f"Capacity = {f:.2f} x mean day")
for k in (1, 2): axs[k].set_xscale("symlog", linthresh=1); axs[k].set_xlabel("On-site storage (days of mean demand)")
axs[1].set_ylabel("Days per year on dry backup (median site)"); axs[2].set_ylabel("Extra electricity of dry backup (% of IT energy)"); axs[1].legend(frameon=False, loc="upper right", fontsize=6)
label(axs[1], "b"); label(axs[2], "c")
ax = axs[0]; s0 = sy0.loc[cells, "S_days_f1.0"].values
for src, col, lab in ((sy1, C["gold"], "+1 °C"), (sy2, C["teal"], "+2 °C")):
    s2 = src.loc[cells, "S_days_f1.0"].values; ax.scatter(s0, s2, s=np.sqrt(wts) * 2.0, c=col, alpha=0.6, edgecolor="none", label=lab)
m = max(s0.max(), sy2.loc[cells, "S_days_f1.0"].max()) * 1.05; ax.plot([0, m], [0, m], color=C["gray"], lw=0.7, ls="--"); ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("Storage at a mean-day allocation,\n2014 to 2023 climate (days)"); ax.set_ylabel("With uniform warming (days)"); ax.legend(frameon=False, loc="upper left", fontsize=6); label(ax, "a")
save(fig, "Fig2")

# ================================================================ Fig 3: capacity-aware siting (headroom 25 % of the existing maximum day)
a = alloc[(alloc.run == "higher_50") & (alloc.headroom_def == "mdd") & (alloc.h == 0.25) & (alloc.threshold == 10000) & (alloc.design == "water_priority")]
fig = plt.figure(figsize=(7.2, 6.1)); gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.1], hspace=0.40, wspace=0.3)
for k, (regime, ttl) in enumerate((("no_storage", "Peak-day demand (no storage)"), ("storage", "Mean-day demand (storage with dry backup)"))):
    ax = fig.add_subplot(gs[0, k]); d = a[a.regime == regime].sort_values("sites", ascending=False).head(20); yy = np.arange(len(d))[::-1]
    ax.barh(yy, d.kept_own, color=C["blue"], label="Served at the IM3 location"); ax.barh(yy, d.moved, left=d.kept_own, color=C["teal"], label="Served after relocation within the state"); ax.barh(yy, d.unservable, left=d.kept_own + d.moved, color=C["orange"], label="Not servable by existing systems")
    ax.set_yticks(yy); ax.set_yticklabels(d.state, fontsize=6); ax.set_xlabel("Projected sites, Higher growth (36 MW each)"); title(ax, ttl)
    if k == 0: ax.legend(frameon=False, loc="lower right", fontsize=6)
    label(ax, "ab"[k])
ax = fig.add_subplot(gs[1, :]); basemap(ax); pl = asites[(asites.run == "higher_50") & (asites.regime == "storage")]
own = hi.copy(); kept = set(zip(pl[pl.own].xk.round(0), pl[pl.own].yk.round(0))); own["kept"] = [(x, y) in kept for x, y in zip(own.xk.round(0), own.yk.round(0))]
x, y = site_xy(own[own.kept]); ax.scatter(x, y, s=4, c=C["blue"], edgecolor="none", zorder=4, label=f"IM3 site served (n = {int(own.kept.sum())})")
x, y = site_xy(own[~own.kept]); ax.scatter(x, y, s=9, marker="x", c=C["orange"], linewidths=0.6, zorder=5, label=f"IM3 site displaced (n = {int((~own.kept).sum())})")
mv = pl[~pl.own].merge(sites.drop_duplicates(["xk", "yk"])[["xk", "yk", "lon", "lat"]], on=["xk", "yk"], how="left")
if len(mv): x, y = site_xy(mv); ax.scatter(x, y, s=4, c=C["teal"], edgecolor="none", zorder=4, label=f"Relocated to another feasible location (n = {len(mv)})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=6.5); label(ax, "c", x=0.0, y=1.02); title(ax, "      Higher growth, 50% market gravity; mean-day demand with storage; headroom 25% of each system's existing maximum day")
save(fig, "Fig3")

# ================================================================ Fig 4: utility exposure
u = util[util.run == "higher_50"].copy()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
ax = axs[0]
for colname, col, ls, lab in (("ratio_peak_mdd", C["orange"], "-", "Peak day, no storage (always-on design)"), ("ratio_mean_mdd", C["blue"], "-", "Mean day, storage and dry backup"), ("ratio_peak_mdd_hy", C["teal"], "--", "Peak day, hybrid design")):
    v = np.sort(u[colname].values * 100); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.3, ls=ls, label=lab)
for xv in (25, 100): ax.axvline(xv, color=C["gray"], lw=0.7, ls=":")
ax.set_xscale("log"); ax.set_xlim(0.1, 1000); ax.set_xlabel("Demand of the attributed sites (% of the system's existing maximum day)"); ax.set_ylabel(f"Cumulative share of {len(u)} systems"); ax.legend(frameon=False, loc="lower right", fontsize=5.5); label(ax, "a"); title(ax, "Community water systems serving the Higher-growth sites")
ax = axs[1]; s = util.groupby("run").agg(peak25=("ratio_peak_mdd", lambda v: int((v >= 0.25).sum())), mean25=("ratio_mean_mdd", lambda v: int((v >= 0.25).sum())), hy25=("ratio_peak_mdd_hy", lambda v: int((v >= 0.25).sum()))); s["growth"] = s.index.str.split("_").str[0]
pos = {g: i for i, g in enumerate(["low", "moderate", "high", "higher"])}
for colname, off, c_ in (("peak25", -0.25, C["orange"]), ("mean25", 0.0, C["blue"]), ("hy25", 0.25, C["teal"])):
    for g, grp in s.groupby("growth"): ax.boxplot(grp[colname].values, positions=[pos[g] + off], widths=0.2, patch_artist=True, boxprops=dict(facecolor=c_, alpha=0.6, lw=0.6), medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), flierprops=dict(ms=2))
ax.set_xticks(range(4)); ax.set_xticklabels(["Low", "Moderate", "High", "Higher"]); ax.set_ylabel("Systems at or above 25% of their maximum day"); ax.set_xlabel("Growth scenario (box over the five siting runs)")
ax.legend(handles=[Patch(color=C["orange"], alpha=0.6, label="Peak day, no storage"), Patch(color=C["blue"], alpha=0.6, label="Mean day, with storage"), Patch(color=C["teal"], alpha=0.6, label="Peak day, hybrid design")], frameon=False, fontsize=6, loc="upper left"); label(ax, "b")
save(fig, "Fig4")

# ================================================================ Fig 5: water-power coincidence on UTC by EIA-930 region
p = prof["higher_50"]; hrs = np.arange(24); pr = pc[(pc.run == "higher_50") & (pc.level == "region")].set_index("name").sort_values("it_mw", ascending=False); pn = pc[(pc.run == "higher_50") & (pc.level == "nation")].iloc[0]
fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32)
ax = fig.add_subplot(gs[0, 0])
for h in sorted(set(p["grid_peak_hours_utc"])): ax.axvspan(h - 0.5, h + 0.5, color=C["lgray"], alpha=0.5, lw=0)
ax.plot(hrs, p["g_wf"], color=C["orange"], lw=1.2, ls="--", label="Water-free (dry) design"); ax.plot(hrs, p["g_S7"], color=C["purple"], lw=1.2, ls="-.", label="Mean-day allocation, one week of storage, dry backup"); ax.plot(hrs, p["g_wp"], color=C["blue"], lw=1.5, label="Always-on evaporative design")
ax.set_xlabel(f"Hour of day (UTC), national grid peak day {p['grid_peak_day']}"); ax.set_ylabel("Cooling electricity, all projected sites (MW)"); ax.set_xlim(0, 23); ax.set_ylim(0, None); ax.legend(frameon=False, loc="lower left", fontsize=5.8); label(ax, "a"); title(ax, "Shaded: hours of day of the 20 highest summer demands of 2022 and 2023")
ax = fig.add_subplot(gs[0, 1]); yy = np.arange(len(pr))[::-1]; h = 0.26
ax.barh(yy + h, pr.grid_add_S0 / pr.it_mw * 1000, height=h, color=C["orange"], label="No storage"); ax.barh(yy, pr.grid_add_S7 / pr.it_mw * 1000, height=h, color=C["purple"], label="Mean day + one week of storage"); ax.barh(yy - h, pr.grid_add_f125_S7 / pr.it_mw * 1000, height=h, color=C["blue"], label="1.25 x mean day + one week")
ax.set_yticks(yy); ax.set_yticklabels([f"{r} ({v/1000:.1f} GW)" for r, v in zip(pr.index, pr.it_mw)], fontsize=6); ax.set_xlabel("Increment at the 40 summer peak hours (MW per GW IT)"); ax.legend(frameon=False, fontsize=6, loc="lower right"); label(ax, "b"); title(ax, "By EIA-930 region, Higher growth")
ax = fig.add_subplot(gs[1, 0]); vals = [pn.grid_cool_mw, pn.grid_cool_mw + pn.grid_add_S0, pn.grid_cool_mw + pn.grid_add_S7, pn.grid_cool_mw + pn.grid_add_f125_S7, pn.grid_hy_mw]
labs = ["Evaporative\ndesign", "Dry backup,\nno storage", "Dry backup,\n1 week", "Dry backup,\n1.25 x mean\n+ 1 week", "Hybrid\ndesign"]
ax.bar(range(5), np.array(vals) / 1000, color=[C["blue"], C["orange"], C["purple"], C["lblue"], C["teal"]])
for i, v in enumerate(vals): ax.text(i, v / 1000, f"{v/1000:.1f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=5.6); ax.set_ylabel("Cooling electricity at the US-48 summer peak hours (GW)"); label(ax, "c"); title(ax, f"Mean over the 40 summer peak hours; {pn.it_mw/1000:.1f} GW of IT load")
ax = fig.add_subplot(gs[1, 1]); mod = pd.read_csv(f"{R3}/region_peak_hours_v5.csv"); gh = mod[mod.season == "summer"].set_index("region").modal_utc_hour
wrap = lambda h: h + 24 if h < 6 else h
for r in pr.index:
    g_, c_ = wrap(int(gh[r])), wrap(int(pr.loc[r, "cool_modal_hour_utc"])); ax.plot([g_, c_], [r, r], color=C["gray"], lw=0.8); ax.plot(g_, r, "o", color=C["dark"], ms=4); ax.plot(c_, r, "s", color=C["blue"], ms=4)
ax.set_xlim(5.5, 29.5); ax.set_xticks([6, 9, 12, 15, 18, 21, 24, 27]); ax.set_xticklabels(["6", "9", "12", "15", "18", "21", "0", "3"]); ax.set_xlabel("Hour of day (UTC), axis from 06:00 to 03:00")
ax.legend(handles=[Line2D([0], [0], marker="o", color=C["dark"], ls="none", label="Grid demand, modal hour of the 50 highest summer hours"), Line2D([0], [0], marker="s", color=C["blue"], ls="none", label="Modeled cooling electricity, modal hour of its 50 highest hours")], frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1); ax.tick_params(axis="y", labelsize=6); label(ax, "d"); title(ax, "Timing of the summer grid peak and of the cooling peak")
fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)
save(fig, "Fig5")

# ================================================================ Fig 6: least-cost service
c0 = ex_cells[1]; g = lcg[lcg.cell == c0]; fp = sy0.loc[c0, "f_peak"]
fig = plt.figure(figsize=(7.2, 3.1)); gs = fig.add_gridspec(1, 3, wspace=0.5)
ax = fig.add_subplot(gs[0, 0])
for sdv, col in zip((0, 2, 7, 30), plt.cm.viridis(np.linspace(0, 0.9, 4))):
    d = g[g.S_days == sdv].sort_values("f"); ax.plot(d.f, d.total_tank / 1e6, color=col, marker="o", ms=2.5, lw=1.0, label=f"{sdv} days of storage")
r = lcs.loc[c0]; ax.scatter([fp], [r.conventional_cost / 1e6], marker="*", s=70, c=C["orange"], zorder=5, label="Conventional (peak-day capacity)")
ax.set_xlabel("Allocated capacity (multiple of the mean day)"); ax.set_ylabel("Annualized cost, one 36 MW site (M$ per year)"); ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=5.3, loc="lower left"); label(ax, "a"); title(ax, cell_label(c0))
ax = fig.add_subplot(gs[0, 1]); L = lcs.loc[cells]; fcat = np.where(L.best_f.values >= L.index.map(lambda c: sy0.loc[c, "f_peak"]).values - 1e-6, 6, np.searchsorted([0.7, 0.9, 1.1, 1.3, 1.7, 2.5], L.best_f.values))
sedges = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12, 17.5, 25.5, 100]; H2, _, _ = np.histogram2d(fcat, L.best_S_days.values, bins=[np.arange(-0.5, 7.5, 1), sedges], weights=wts)
im = ax.pcolormesh(np.arange(len(sedges)), np.arange(8), H2 / wts.sum() * 100, cmap="Blues"); ax.set_yticks(np.arange(7) + 0.5); ax.set_yticklabels(["0.6", "0.8", "1.0", "1.25", "1.5", "2.0", "peak day"]); ax.set_xticks(np.arange(len(sedges) - 1) + 0.5); ax.set_xticklabels(["0", "1", "2", "3", "5", "7", "10", "14", "21", "30+"], fontsize=5.5)
ax.set_xlabel("Least-cost storage (days)"); ax.set_ylabel("Least-cost capacity (x mean day)"); cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("% of projected sites", fontsize=6); cb.ax.tick_params(labelsize=6); label(ax, "b")
ax = fig.add_subplot(gs[0, 2]); sv = L.saving_tank_frac.values * 100; sv2 = L.saving_pond_frac.values * 100
for v, col, lab in ((sv, C["blue"], "Steel tanks"), (sv2, C["teal"], "Lined ponds")):
    o = np.argsort(v); ax.step(v[o], np.cumsum(wts[o]) / wts.sum(), where="post", color=col, lw=1.3, label=lab)
q = mc.median_saving_pct.quantile([0.05, 0.95]); ax.axvspan(q.iloc[0], q.iloc[1], color=C["lgray"], alpha=0.5, lw=0, label="Monte Carlo band")
ax.axvline(0, color=C["gray"], lw=0.6); ax.set_xlabel("Saving vs peak-day capacity (%)"); ax.set_ylabel("Cumulative share of projected sites"); ax.legend(frameon=False, loc="lower right", fontsize=5.5); label(ax, "c")
fig.subplots_adjust(left=0.07, right=0.97)
save(fig, "Fig6")
