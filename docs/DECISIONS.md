# Design Decisions

Log of architectural decisions and the reasoning behind each.
Evidence for data decisions: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Column-level classification: [`COLUMN_AUDIT.md`](COLUMN_AUDIT.md).

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Dataset | Lending Club 2007–2020Q3 (Kaggle: ethon0426) | Real issuance dates enable genuine temporal drift; immature late vintages serve as a drift-replay stream. |
| D2 | Label + population | 36-month, individual applications, issued **2013-01 … 2017-04**, resolved only (953,889 loans, 14.7% default). Charged Off/Default = 1, Fully Paid = 0 | ≥98% of loans issued ≤ 2017-04 are resolved; after that, right-censoring distorts the label. 2013-01 start = every loan has the bureau fields added in 2012. Joint apps are rare and carry their own field set. |
| D3 | Validation | Out-of-time split: train 2013-01…2015-12 (546k) · val 2016-01…2016-06 (168k) · test 2016-07…2017-04 (240k) | Random K-fold leaks future information and hides the drift visible in the data (default rate 12% → 16%). |
| D4 | LC decision columns | Excluded from features: `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt(_inv)`, `initial_list_status`, `verification_status`. Grade ablation only | Outputs of Lending Club's own underwriting. `installment` reconstructs `int_rate` (Spearman 0.9997). LC verifies applicants it already considers risky ("Verified" defaults 18% vs 11%). |
| D4b | Other exclusions | Post-origination (32 cols, leakage), bureau wave 3 (14 cols, only from 2015-12), joint-only (15), proxies (`zip_code`, `addr_state`, `emp_title`) | See `COLUMN_AUDIT.md`. 61 candidate features remain. |
| D5 | Cost function | Loss = PD × funded_amnt × LGD. Missed profit = contractual interest × prepayment haircut. Global threshold minimizes total $ cost. LGD & haircut estimated on train split only | EDA preview: LGD ≈ 0.52, haircut ≈ 0.80, break-even PD ≈ 23% for an average loan — far from the naive 0.5. |
| D6 | Class imbalance | No resampling; calibrate probabilities; cost-based threshold | Resampling distorts probabilities that expected-loss math depends on. 14.7% default rate is mild imbalance. |
| D7 | Algorithm | LightGBM | Native categorical handling; fast on ~1M rows. |
| D8 | Tracking + data remote | DagsHub (hosted MLflow + DVC remote) | Free, publicly browsable runs; enables CI retraining. |
| D9 | Model promotion | MLflow registry aliases (`@challenger` → `@champion`) | Registry stages are deprecated since MLflow 2.9. |
| D10 | Model packaging | Model baked into Docker image at build time | Image tag = model version; rollback = redeploy old image; no runtime MLflow dependency. |
| D11 | Deployment | GCP Cloud Run | Runs containers directly, scales to zero, free tier. |
| D12 | Additions | pandera data validation; SHAP reason codes in API | Catch bad data before training; US lenders must give reasons for denials (adverse action). |
| D13 | Tooling | Python 3.11, uv, ruff, pytest | Fast, reproducible (lockfile), standard. |
| D14 | Ingest | Stream CSV → parquet, every column as string | 1.77 GB CSV doesn't fit in pandas on 8 GB RAM; type inference from early rows is wrong for columns empty before 2012. Parsing is explicit in the clean stage. |

## Known corners cut
- **No reject inference:** trained only on approved loans (selection bias).
- **Simulated label arrival:** the OOT split assumes labels exist immediately; real credit labels arrive months to years later.
- **Simplified economics:** fixed LGD; ignores interest collected before default, cost of funds, servicing fees, capital charges.
- **Bureau wave 3 unused:** 14 informative fields dropped because they only exist from 2015-12.
- **Bureau fields assumed as-of application** (per LC data dictionary) — not verifiable from the data.
- **No feature store:** a shared Python module replaces Feast/Tecton.
- **No shadow or canary rollout:** promotion goes straight to production.
- **Limited fairness analysis:** no protected attributes in the data.
