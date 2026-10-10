"""Extension of the revised cooling model (revmodel.py, Sanchez et al., Energy Nexus revision) with a fourth,
water-free mode: chiller with a dry cooler on the condenser loop. Everything else is the published model.
Modes: 0 free cooling (dry cooler), 1 evaporative tower, 2 chiller + evaporative tower, 3 chiller + dry cooler."""
import sys, numpy as np
sys.path.insert(0, "F:/2025/Publications/Raul_DataCenters/Reviewers/Revision_code")
import revmodel as R


def run_modes(T2, RH2, p, **kw):
    prm = dict(R.DEFAULT); prm.update(kw); pr = R.Props(prm['f_cpL'], prm['f_hfg'], prm['f_cpa'])
    T2 = np.asarray(T2, float); RH2 = np.clip(np.asarray(RH2, float), 0.5, 100.0); n = len(T2)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).astype(float)
    Ts, Tr = prm['T_ite_supply'], prm['T_ite_return']; one = np.ones(n)
    Twb = R.wet_bulb(T2, RH2, p, pr)
    free_ok = T2 <= Ts - prm['a_dry']; evap_ok = Twb <= Ts - prm['a_ev']
    fr = R.tower_states(Tr*one, Ts*one, T2, RH2, one, p, prm, pr, False)
    ev = R.tower_states(Tr*one, Ts*one, T2, RH2, one, p, prm, pr, True)
    T4c = np.maximum(prm['T4_design'], Twb + prm['a_ev']); T1c = T4c + prm['dT_chiller']; copc = R.cop_fun(T4c, T1c, prm)
    ch = R.tower_states(T1c, T4c, T2, RH2, one, p, prm, pr, True)
    # dry condenser mode inside a stated chiller envelope: the condenser-water leaving temperature T1 is capped at T1max
    # (63 C, the high-condensing option of standard water-cooled screw chillers; None = uncapped, the model's 24 K range
    # at any ambient, which needs a high-lift machine above 33 C). The chiller range is reduced to fit the cap (at least
    # R_min K), which raises the dry-cooler air flow; the COP correlation is evaluated at the entering condenser-water
    # temperature T4 (the model's convention, cop_T='entering'), or at the leaving temperature ('leaving') as a sensitivity.
    T1max = prm.get('T1max', 63.0); cop_T = prm.get('cop_T', 'entering'); R_min = prm.get('R_min', 3.0)
    T4d = np.maximum(prm['T4_design'], T2 + prm['a_dry'])
    R0 = float(prm['dT_chiller']) if prm.get('R_dry') is None else float(prm['R_dry'])   # condenser range of the dry mode: the model's 24 K, or a conventional range (R_dry, e.g. 6 K)
    Rd = np.full(n, R0) if T1max is None else np.maximum(np.minimum(R0, T1max - T4d), R_min)
    T1d = T4d + Rd; Tcop = {'entering': T4d, 'leaving': T1d, 'mid': T4d + 0.5*Rd}[cop_T]; copd = R.cop_fun(Tcop, T1d, prm)
    dr = R.tower_states(T1d, T4d, T2, RH2, one, p, prm, pr, False)
    aux = lambda st: st['VA5']*prm['dp_fan']/1000.0 + (st['mw']/1000.0)*prm['dp_pump']/1000.0
    PUE = np.vstack([1 + aux(fr), 1 + aux(ev), 1 + aux(ch) + 1.0/copc, 1 + aux(dr) + 1.0/copd])
    PUE[0, ~free_ok] = np.nan; PUE[1, ~evap_ok] = np.nan
    WUE = np.vstack([np.zeros(n), np.where(evap_ok, ev['WUE'], np.nan), ch['WUE'], np.zeros(n)])
    return dict(PUE=PUE, WUE=WUE, free_ok=free_ok, evap_ok=evap_ok, Twb=Twb, cop_evap=copc, cop_dry=copd, T1_dry=T1d, R_dry=Rd, VA5_dry=dr['VA5'], VA5_free=fr['VA5'])


def rh_const_w(T, RH, dT):
    """Relative humidity after a uniform warming dT at constant vapor pressure (constant specific humidity)."""
    pv = RH/100*R.psat(T)
    return np.clip(100*pv/R.psat(T+dT), 0.5, 100.0)


def select(PUE, WUE, design, k=None):
    """Hourly mode choice. design: 'water_priority' (free > evap > chiller+evap; the published rule),
    'water_free' (free > chiller+dry), 'total_water' (argmin WUE + k*PUE over feasible modes, k in L/kWh)."""
    n = PUE.shape[1]
    if design == 'water_priority':
        mode = np.where(~np.isnan(PUE[0]), 0, np.where(~np.isnan(PUE[1]), 1, 2))
    elif design == 'water_free':
        mode = np.where(~np.isnan(PUE[0]), 0, 3)
    elif design == 'total_water':
        J = WUE + k*PUE; J = np.where(np.isnan(J), np.inf, J); mode = np.argmin(J, axis=0)
    else:
        raise ValueError(design)
    idx = np.arange(n)
    return mode, PUE[mode, idx], WUE[mode, idx]
