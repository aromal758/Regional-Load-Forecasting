# Real-data dashboard (add-on)

These files are added next to the original dashboard. They do not replace or change anything.

- `app_ausgrid.py`: dashboard on the real Ausgrid Solar Home data with CG-SFDRLF forecasts
- `data/ausgrid_cg_dashboard.npz`: measurements + CG-SFDRLF test forecasts (seed 0)
- `data/model_comparison.csv`: 3-seed comparison table from the full experiment
- `prepare_dashboard_data.py`: rebuilds the data file from `dataset.npz` and `cg_dash_export.npz`

Run locally: `streamlit run app_ausgrid.py`
Deploy on Streamlit Community Cloud as a separate app with main file path `app_ausgrid.py`.
The original `app.py` keeps running at its own link.
