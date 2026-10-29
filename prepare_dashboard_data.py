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
