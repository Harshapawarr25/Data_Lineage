# Oberva

Oberva is a local Streamlit workspace for staged dataset quality checks,
reference-versus-current statistical drift analysis, and searchable project PDFs.

## Run locally on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

Run Streamlit as a Python module so PowerShell does not need the standalone
`streamlit` command on `PATH`. If you already created the virtual environment,
activate it first with `.\.venv\Scripts\Activate.ps1`.

## Analysis workflow

1. **Intake:** Upload a reference and current CSV. Review missing/extra fields,
   inferred feature types, and previews, then select **Run Analysis - Create run**.
2. **Data Quality:** Compare both datasets for completeness, duplicates, IQR and
   z-score outliers, configured domain violations, unexpected categories, and
   categorical-group missingness. Adjust amber/red thresholds and numeric bounds.
3. **Drift Detection:** Compare shared features using numeric KS and Mann-Whitney
   tests, Wasserstein distance and PSI; categorical chi-square, Jensen-Shannon
   distance and PSI; and new-category detection. Valid p-values are corrected
   together using Benjamini-Hochberg. Adjust significance and PSI thresholds.
4. **Dashboard:** Review overall quality/drift signals, alerts, the plain-English
   summary, full statistical tables, and run timeline; download a self-contained
   HTML report.

Each step is gated by completion of the previous analysis stage. Project PDFs are
an optional side page and are not needed for CSV analysis. Text-based PDFs can be
searched and their extracted text downloaded; scanned PDFs need OCR first.

## Local data and privacy

Run inputs, SQLite results, and the run timeline are saved under `data/` by default:

- `data/oberva.db` contains the `runs`, `quality_results`, and `drift_results`
  tables.
- `data/runs/<run_id>/reference.csv` and `current.csv` preserve each run's inputs.

Set the `OBERVA_DATA_DIR` environment variable before starting the app to use a
different local data directory. The previous `OBSERVA_DATA_DIR` variable is still
accepted for existing setups. Uploaded datasets are stored locally and are not
sent to an external service. PDF files and extracted PDF text remain in Streamlit
session state and are not saved to disk. Do not commit `data/`; it is excluded by
the project `.gitignore`.

Automatic non-negative bounds are suggested only for clearly named numeric
fields such as charges, costs, fees, counts, and quantities. Review or override
these bounds for each run. Inferred feature types treat two-valued numeric fields
as categorical. Statistical flags help prioritize review and do not prove a cause.

## Tests

Install pytest in the active environment if needed, then run:

```powershell
python -m pip install pytest
python -m pytest -q tests
```
