"""Build the dashboard data file from the Ausgrid dataset and a CG-SFDRLF forecast export.

usage: python prepare_dashboard_data.py dataset.npz cg_dash_export.npz metrics_all_seeds.csv
Writes data/ausgrid_cg_dashboard.npz and data/model_comparison.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H = 6
ds_path, ex_path, metrics_path = sys.argv[1:4]
out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)

d = np.load(ds_path)
ex = np.load(ex_path)
L, PV, CS, cap, split = d["L"], d["PV"], d["CS"], d["cap"], d["split"]
s = L[split == 0].mean(0).astype(np.float32)          # same node scale as common.Data
kw = 2 * s                                             # normalised -> kW (kWh per half hour x 2)
day = (CS / cap) > 0.01

t = ex["test_t"].astype(np.int64)
assert np.allclose(ex["s"], s, rtol=1e-4), "dataset and export use different scales"
fut = t[:, None] + np.arange(1, H + 1)
muL = ex["test_muL"].astype(np.float32) * kw
sL = ex["test_sL"].astype(np.float32) * kw
muP = ex["test_muP"].astype(np.float32) * kw
sP = ex["test_sP"].astype(np.float32) * kw
rho = ex["test_rho"].astype(np.float32) if "test_rho" in ex.files else np.zeros_like(muL)

# check against the report: MAE in kW per node over all test windows and horizons
yL, yP = 2 * L[fut], 2 * PV[fut]
m = day[fut]
muN = muL - muP
sN = np.sqrt(np.maximum(sL ** 2 + sP ** 2 - 2 * rho * sL * sP, 1e-6))
print(f"windows {len(t)}  load MAE {np.abs(yL - muL).mean():.3f}  PV MAE (day) {np.abs(yP - muP)[m].mean():.3f}"
      f"  net MAE {np.abs(yL - yP - muN).mean():.3f}  net RMSE {np.sqrt(((yL - yP - muN) ** 2).mean()):.3f} kW")
print(f"net 90% coverage {(np.abs(yL - yP - muN) <= 1.645 * sN).mean():.3f}"
      f"  95% reserve coverage {((yL - yP) <= muN + 1.645 * sN).mean():.3f}")

f16 = lambda a: np.asarray(a, dtype=np.float16)
np.savez_compressed(
    out / "ausgrid_cg_dashboard.npz",
    start=np.array("2012-07-01T00:00"), L_kw=f16(2 * L), PV_kw=f16(2 * PV), CS_kw=f16(2 * CS), split=split.astype(np.int8),
    lat=d["lat"], lon=d["lon"], nh=d["nh"], cap=cap, has_cl=d["has_cl"],
    t=t.astype(np.int32), muL=f16(muL), sL=f16(sL), muP=f16(muP), sP=f16(sP), rho=f16(rho),
)

# model comparison table (means over the 3 seeds of the full experiment)
mt = pd.read_csv(metrics_path)
names = {"cg": "CG-SFDRLF (proposed)", "sfdrlf_mt": "SFDRLF-MT", "sfdrlf_load": "SFDRLF (load only)",
         "lstm": "LSTM-MT", "gcngru": "GCN-GRU", "Seasonal naive (24 h)": "Seasonal naive (24 h)", "Persistence (smart PV)": "Persistence (smart PV)"}
print(sorted(mt["model"].unique()))
mt = mt[mt["model"].isin(names)]
cols = [c for c in ["L_MAE", "PV_MAE", "N_MAE", "N_RMSE", "N_CRPS", "R_conf_kW", "R_conf_cov"] if c in mt.columns]
tab = mt.groupby("model")[cols].mean().round(3)
tab.index = [names[i] for i in tab.index]
tab = tab.rename(columns={"L_MAE": "Load MAE (kW)", "PV_MAE": "PV MAE (kW)", "N_MAE": "Net MAE (kW)",
                          "N_RMSE": "Net RMSE (kW)", "N_CRPS": "Net CRPS (kW)", "R_conf_kW": "95% reserve (kW)", "R_conf_cov": "Reserve coverage"})
tab = tab.sort_values("Net MAE (kW)", na_position="last")
tab.to_csv(out / "model_comparison.csv", index_label="Model")
print(tab)


# ---------------- all-regions total: cross-node error correlation (cross-fitted) ----------------
# Regional forecast errors are partly correlated (shared weather). For the 21-region total we use
# sigma_tot = sqrt(s^T C s), with the correlation matrix C estimated on the *other* half of the test
# blocks (even/odd 2-day blocks), so the displayed coverage is out-of-sample.
z = np.load(out / "ausgrid_cg_dashboard.npz")
store = {k: z[k] for k in z.files}
t = store["t"].astype(int); H6 = store["muL"].shape[1]
L2, P2 = store["L_kw"].astype(float), store["PV_kw"].astype(float)
mL, mP = store["muL"].astype(float), store["muP"].astype(float)
sL2, sP2, r2 = store["sL"].astype(float), store["sP"].astype(float), store["rho"].astype(float)
sN2 = np.sqrt(np.maximum(sL2 ** 2 + sP2 ** 2 - 2 * r2 * sL2 * sP2, 1e-6))
block = np.cumsum(np.r_[1, np.diff(t) > 1]) % 2          # alternate test blocks: 0/1
tot = {}
for key, y_fn, mu, sd in [("L", lambda h: L2[t + h], mL, sL2), ("P", lambda h: P2[t + h], mP, sP2),
                          ("N", lambda h: L2[t + h] - P2[t + h], mL - mP, sN2)]:
    out_sd = np.zeros((len(t), H6), np.float32)
    for h in range(1, H6 + 1):
        zres = (y_fn(h) - mu[:, h - 1]) / np.maximum(sd[:, h - 1], 1e-3)
        for part in (0, 1):
            C = np.corrcoef(zres[block != part].T)
            C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
            s = sd[block == part, h - 1]
            out_sd[block == part, h - 1] = np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', s, C, s), 1e-6))
    tot[key] = out_sd
yN = np.stack([L2[t + h].sum(1) - P2[t + h].sum(1) for h in range(1, H6 + 1)], 1)
muN = (mL - mP).sum(2)
print("all-regions 95%% reserve coverage by horizon (cross-fitted):",
      np.round(np.mean(yN <= muN + 1.645 * tot["N"], 0), 3), " mean reserve kW:", np.round((1.645 * tot["N"]).mean(0), 1))
store.update(sL_tot=tot["L"].astype(np.float16), sP_tot=tot["P"].astype(np.float16), sN_tot=tot["N"].astype(np.float16))
np.savez_compressed(out / "ausgrid_cg_dashboard.npz", **store)
