"""Supplementary figures of the revised manuscript (v5). Writes submission_v5/figures/FigS*.png."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from matplotlib.patches import Patch
import shapely, pyproj
from pyogrio.raw import read as raw_read
BASEW = "F:/2025/Publications/Water_DataCenters"; EXT = BASEW + "/data_ext"; RES = EXT + "/results"; R3 = EXT + "/results_v5"; MOD = EXT + "/model_power"
OUT = BASEW + "/submission_v5/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})
C = dict(blue="#2b6cb0", lblue="#9ecae1", orange="#e66101", purple="#5e3c99", teal="#1b9e77", gray="#8c8c8c", lgray="#d9d9d9", dark="#222222", gold="#e6ab02")
fwd = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:102003", always_xy=True).transform
NE = "C:/Users/workt/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_50m_admin_1_states_provinces_lakes.shp"
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
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{OUT}/{name}.eps", format="eps", bbox_inches="tight", facecolor="white", pad_inches=0.04); plt.close(fig); print(name)
sites = pd.read_csv(f"{R3}/sites_infra_v5.csv", dtype={"PWSID": str}); summ = pd.read_csv(f"{MOD}/_summary.csv").set_index("cell"); sy = pd.read_csv(f"{R3}/storage_yield_cells_v5.csv")
sy0 = sy[sy.design == "water_priority"].set_index("cell"); syh = sy[sy.design == "hybrid"].set_index("cell"); sy1 = sy[sy.design == "water_priority_plus1"].set_index("cell"); sy2 = sy[sy.design == "water_priority_plus2"].set_index("cell")
hi = sites[sites.run == "higher_50"].copy(); hi = hi[hi.cell.isin(sy0.index)]; w = hi.groupby("cell").size(); cells = w.index.values; wts = w.values.astype(float)
IT_KW = 36000.0; L_PER_MG = 3.785411784e6; MU = 1.25

# ---- S1 station observations versus the reanalysis cell, 46 airports
v = pd.read_csv(f"{R3}/validation_isd_stations.csv")
fig, axs = plt.subplots(1, 5, figsize=(7.8, 1.8), constrained_layout=True); fig.get_layout_engine().set(rect=(0.0, 0.0, 0.955, 1.0))
for ax, (a, b, t, letter) in zip(axs, (("WUE_wp", "WUE_wp", "WUE, evaporative (L per kWh)", "a"), ("pk_wp", "pk_wp", "Peak-day factor, evaporative", "b"), ("S1_wp", "S1_wp", "Storage, mean-day allocation (d)", "c"), ("monthpk_hy", "monthpk_hy", "Monthly factor, hybrid", "d"), ("pk_hy", "pk_hy", "Daily factor, hybrid", "e"))):
    xs = v[a + "_station"].values; ys = v[b + "_power"].values; ax.scatter(xs, ys, c=C["blue"], s=12, alpha=0.8, edgecolor="none", zorder=3)
    lo = min(xs.min(), ys.min()) * 0.95; hi_ = max(xs.max(), ys.max()) * 1.05; ax.plot([lo, hi_], [lo, hi_], color=C["gray"], lw=0.7, ls="--"); ax.set_xlim(lo, hi_); ax.set_ylim(lo, hi_)
    r = np.corrcoef(xs, ys)[0, 1]; bias = 100 * (ys - xs).mean() / xs.mean(); ax.text(0.04, 0.96, f"r = {r:.3f}\nbias {bias:+.1f}%", transform=ax.transAxes, fontsize=6, va="top")
    ax.set_xlabel("Station observations (ISD)"); ax.set_ylabel("Reanalysis cell (POWER)"); ax.set_title(t, fontsize=5.4, loc="center", pad=3); label(ax, letter, x=-0.14, y=1.18)
    if a == "pk_hy": ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi_); ax.set_ylim(lo, hi_); ax.set_xlabel("Station observations (ISD)", loc="right")
save(fig, "FigS1")

# ---- S2 annual water intensity and mode occupancy
fig = plt.figure(figsize=(7.2, 3.3)); gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.18)
ax = fig.add_subplot(gs[0, 0]); basemap(ax); x, y = fwd(hi.lon.values, hi.lat.values); vv = summ.loc[hi.cell, "WUE_water_priority_0"].values
sc = ax.scatter(x, y, c=vv, s=5, cmap="plasma", edgecolor="none", zorder=4); cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8); cb.set_label("Annual evaporative WUE, always-on evaporative design (L per kWh of IT)", fontsize=6.5); cb.ax.tick_params(labelsize=6); label(ax, "a", x=0.0, y=1.02)
ax = fig.add_subplot(gs[0, 1]); occ = hi.groupby("state").apply(lambda g: pd.Series({m: summ.loc[g.cell, f"occ{m}_water_priority_0"].mean() for m in range(3)})).sort_values(2)
yy = np.arange(len(occ)); ax.barh(yy, occ[0], color=C["lblue"], label="Free cooling"); ax.barh(yy, occ[1], left=occ[0], color=C["blue"], label="Evaporative"); ax.barh(yy, occ[2], left=occ[0] + occ[1], color=C["orange"], label="Chiller + evaporative")
ax.set_yticks(yy); ax.set_yticklabels(occ.index, fontsize=5.5); ax.set_xlabel("Share of hours (site mean by state)"); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3); label(ax, "b")
save(fig, "FigS2")

# ---- S3 climate-resolved water and energy versus the fixed assignments of the siting dataset
phys = summ.loc[hi.cell, "WUE_water_priority_0"].values * IT_KW * 8760 / L_PER_MG; im3e = 34.815746
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
ax = axs[0]; ax.hist(phys / im3e, bins=40, color=C["blue"], alpha=0.8); ax.axvline(1, color=C["orange"], lw=1); ax.set_xlabel("Modeled annual consumption / IM3 all-evaporative constant (34.8 MG per year)"); ax.set_ylabel("Projected sites"); label(ax, "a")
ax = axs[1]; cool = (summ.loc[hi.cell, "PUE_water_priority_0"].values - 1) * IT_KW * 8760 / 1e6; coolwf = (summ.loc[hi.cell, "PUE_water_free_0"].values - 1) * IT_KW * 8760 / 1e6
ax.hist(cool, bins=40, color=C["blue"], alpha=0.7, label="Always-on evaporative design"); ax.hist(coolwf, bins=40, color=C["orange"], alpha=0.7, label="Water-free design"); ax.axvline(94.608, color=C["purple"], lw=1, ls="--", label="IM3 mechanical-cooling energy (94.6 GWh per year)")
ax.set_xlabel("Cooling electricity per 36 MW site (GWh per year)"); ax.set_ylabel("Projected sites"); ax.legend(frameon=False, fontsize=6); label(ax, "b")
save(fig, "FigS3")

# ---- S4 warming sensitivity by state (+1 and +2 C)
g = hi.groupby("state"); w0 = g.apply(lambda d: summ.loc[d.cell, "WUE_water_priority_0"].mean()); w1 = g.apply(lambda d: summ.loc[d.cell, "WUE_water_priority_1"].mean()); w2 = g.apply(lambda d: summ.loc[d.cell, "WUE_water_priority_2"].mean())
p0 = g.apply(lambda d: sy0.loc[d.cell, "peak_day_factor"].mean()); p1 = g.apply(lambda d: sy1.loc[d.cell, "peak_day_factor"].mean()); p2 = g.apply(lambda d: sy2.loc[d.cell, "peak_day_factor"].mean())
s0 = g.apply(lambda d: sy0.loc[d.cell, "S_days_f1.0"].mean()); s1_ = g.apply(lambda d: sy1.loc[d.cell, "S_days_f1.0"].mean()); s2_ = g.apply(lambda d: sy2.loc[d.cell, "S_days_f1.0"].mean())
fig, axs = plt.subplots(1, 3, figsize=(7.2, 4.2), constrained_layout=True); o = (w2 / w0 - 1).sort_values().index; yy = np.arange(len(o)); hh = 0.38
for ax, (a1, a2, xl, letter) in zip(axs, (((w1 / w0 - 1) * 100, (w2 / w0 - 1) * 100, "Change in annual water (%)", "a"), (p1 - p0, p2 - p0, "Change in the peak-day factor", "b"), ((s1_ / s0 - 1) * 100, (s2_ / s0 - 1) * 100, "Change in storage, mean-day allocation (%)", "c"))):
    ax.barh(yy + hh / 2, a1.loc[o].values, height=hh, color=C["gold"], label="+1 °C"); ax.barh(yy - hh / 2, a2.loc[o].values, height=hh, color=C["teal"], label="+2 °C"); ax.set_yticks(yy); ax.set_yticklabels(o, fontsize=5.2); ax.set_xlabel(xl, fontsize=6.5); ax.axvline(0, color=C["gray"], lw=0.6); label(ax, letter)
axs[0].legend(frameon=False, fontsize=6, loc="lower right"); save(fig, "FigS4")

# ---- S5 cost sensitivity: one-at-a-time tornado and Monte Carlo distributions
oat = pd.read_csv(f"{R3}/least_cost_sensitivity_v5.csv"); mc = pd.read_csv(f"{R3}/least_cost_montecarlo_v5.csv"); base = oat.iloc[0]
fig = plt.figure(figsize=(7.2, 3.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.45)
ax = fig.add_subplot(gs[0, 0]); cs = oat.iloc[1:]; yy = np.arange(len(cs))[::-1]
ax.barh(yy + 0.2, cs.median_saving_pct - base.median_saving_pct, left=base.median_saving_pct, height=0.38, color=[C["orange"] if v < base.median_saving_pct else C["blue"] for v in cs.median_saving_pct]); ax.axvline(base.median_saving_pct, color=C["dark"], lw=0.8)
ax2 = ax.twiny(); ax2.barh(yy - 0.2, cs.share_storage_pct - base.share_storage_pct, left=base.share_storage_pct, height=0.38, color=[C["lgray"] if v < base.share_storage_pct else C["teal"] for v in cs.share_storage_pct], alpha=0.9); ax2.axvline(base.share_storage_pct, color=C["teal"], lw=0.8, ls="--")
SHORT = {"Dry mode at the 24 K range with the COP at the entering temperature and dry coolers sized for that range (optimistic bound)": "Dry mode: 24 K range, COP at entering temperature (optimistic)", "Dry mode at the 24 K range with the COP at the leaving temperature and dry coolers sized for that range": "Dry mode: 24 K range, COP at leaving temperature", "Both chillers at a conventional 6 K range with the COP at the leaving temperature": "Both chillers: 6 K range, COP at leaving temperature"}
ax.set_yticks(yy); ax.set_yticklabels([SHORT.get(c, c) for c in cs.case], fontsize=5.8); ax.set_xlabel("Median saving versus peak-day capacity (%, upper bars)", fontsize=6.5); ax2.set_xlabel("Sites where storage is least-cost (%, lower bars)", fontsize=6.5); ax2.tick_params(labelsize=6); label(ax, "a", y=1.14)
ax = fig.add_subplot(gs[0, 1]); ax.hist(mc.median_saving_pct, bins=40, color=C["blue"], alpha=0.8); ax.axvline(base.median_saving_pct, color=C["dark"], lw=0.8); ax.set_xlabel("Median saving (%)"); ax.set_ylabel(f"Draws (n = {len(mc)})"); label(ax, "b")
ax = fig.add_subplot(gs[0, 2]); ax.hist(mc.share_storage_pct, bins=40, color=C["teal"], alpha=0.8); ax.axvline(base.share_storage_pct, color=C["dark"], lw=0.8); ax.set_xlabel("Share with storage least-cost (%)"); ax.set_ylabel("Draws"); label(ax, "c")
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.2, top=0.86)
save(fig, "FigS5")

# ---- S6 utilities map by exposure class (relative to the existing maximum day)
u = pd.read_csv(f"{R3}/utilities_v5.csv", dtype={"PWSID": str}); u = u[u.run == "higher_50"]
loc = hi.groupby("PWSID").agg(lon=("lon", "mean"), lat=("lat", "mean")); u = u.merge(loc, left_on="PWSID", right_index=True, how="left").dropna(subset=["lon"])
fig, ax = plt.subplots(figsize=(7.2, 4.4)); basemap(ax); x, y = fwd(u.lon.values, u.lat.values)
cat = np.where(u.ratio_mean_mdd >= 0.25, 2, np.where(u.ratio_peak_mdd >= 0.25, 1, 0)); sz = np.clip(np.sqrt(u.pws_pop) / 25, 3, 60)
for k, (cc, lab) in enumerate(((C["lblue"], "Peak-day demand below 25% of the system's maximum day"), (C["orange"], "Peak day above 25%, mean day below (storage resolves)"), (C["purple"], "Mean day above 25% of the maximum day (expansion, relocation or dry cooling)"))):
    m = cat == k; ax.scatter(x[m], y[m], s=sz[m], c=cc, edgecolor="white", linewidth=0.3, zorder=4 + k, label=f"{lab} (n = {int(m.sum())})")
ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=6.5); title(ax, "      Community water systems serving projected sites, Higher growth, 50% market gravity (symbol size: population served)")
save(fig, "FigS6")

# ---- S7 hybrid design: annual water intensity and daily peaking factor
fig = plt.figure(figsize=(7.2, 3.1)); gs = fig.add_gridspec(1, 2, wspace=0.08)
for k, (colname, cmap, lab, vmax) in enumerate((("annual_L_per_kW", "plasma", "Annual WUE of the hybrid design (L per kWh of IT)", None), ("peak_day_factor", "viridis", "Daily peaking factor of the hybrid design (maximum day / mean day)", 40))):
    ax = fig.add_subplot(gs[0, k]); basemap(ax); x, y = fwd(hi.lon.values, hi.lat.values); vv = syh.loc[hi.cell, colname].values / (8760 if colname == "annual_L_per_kW" else 1)
    sc = ax.scatter(x, y, c=vv, s=5, cmap=cmap, vmin=0, vmax=vmax if vmax else np.nanpercentile(vv, 98), edgecolor="none", zorder=4); cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.85); cb.set_label(lab, fontsize=6.5); cb.ax.tick_params(labelsize=6); label(ax, "ab"[k], x=0.0, y=1.0)
save(fig, "FigS7")
print("SI figures written")
